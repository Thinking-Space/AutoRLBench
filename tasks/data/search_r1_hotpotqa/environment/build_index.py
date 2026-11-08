"""Build CPU-compatible e5 FlatIP indexes from the staged retrieval catalogs."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time

def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b''):
            value.update(chunk)
    return value.hexdigest()

def main():
    import faiss
    import numpy as np
    workspace = Path(os.environ['AUTORLBENCH_WORKSPACE'])
    sys.path.insert(0, str(workspace))
    from encoder import Encoder
    folder = workspace / 'shared/data'
    config = json.loads((workspace / 'config.json').read_text())
    model = workspace / 'shared/models/encoder'
    model_hash = digest(model / 'model.safetensors')
    cache_root = Path(os.environ.get('AUTORLBENCH_DATASETS') or workspace / '.index-cache') / 'hotpotqa_e5'
    cache_root.mkdir(parents=True, exist_ok=True)
    encoder = None
    previous_texts = previous_vectors = None
    for name in dict.fromkeys([config['training_catalog'], config['validation_catalog']]):
        corpus = folder / (name + '.jsonl')
        docs = [json.loads(line) for line in corpus.read_text().splitlines()]
        texts = [row['contents'] for row in docs]
        specification = {'corpus_sha256': digest(corpus), 'model_sha256': model_hash,
            'prefix': 'passage: ', 'max_length': 256, 'pooling': 'masked_mean_l2',
            'index': 'FlatIP', 'dtype': 'float32', 'rows': len(docs), 'dim': 768}
        key = hashlib.sha256(json.dumps(specification, sort_keys=True).encode()).hexdigest()
        cached, marker = cache_root / (key + '.index'), cache_root / (key + '.complete.json')
        with (cache_root / (key + '.lock')).open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if marker.is_file() and cached.is_file() and json.loads(marker.read_text()) == specification:
                index = faiss.read_index(str(cached))
                assert index.ntotal == len(docs) and index.d == 768
                print('[prepare] index cache verified', name, flush=True)
            else:
                if encoder is None:
                    encoder = Encoder(model, 'cpu', threads=int(os.environ.get('INDEX_CPU_THREADS', '8')))
                if previous_texts is not None and texts == previous_texts[:len(texts)]:
                    vectors = previous_vectors[:len(texts)]
                else:
                    pieces = []
                    start_time = time.time()
                    for offset in range(0, len(texts), 128):
                        pieces.append(encoder.encode(texts[offset:offset + 128], 'passage: '))
                        if offset % 1024 == 0:
                            print('[prepare] encoding', name, min(offset + 128, len(texts)), '/', len(texts), round(time.time() - start_time, 1), flush=True)
                    vectors = np.concatenate(pieces)
                assert vectors.shape == (len(docs), 768) and np.isfinite(vectors).all()
                assert np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=.002)
                index = faiss.IndexFlatIP(768)
                index.add(vectors)
                descriptor, temporary = tempfile.mkstemp(prefix=key + '.', dir=cache_root)
                os.close(descriptor)
                try:
                    faiss.write_index(index, temporary)
                    os.replace(temporary, cached)
                finally:
                    if os.path.exists(temporary):
                        os.unlink(temporary)
                pending = marker.with_suffix('.new')
                pending.write_text(json.dumps(specification, indent=2))
                pending.replace(marker)
                previous_texts, previous_vectors = texts, vectors
            target = folder / (name + '.index')
            temporary = target.with_suffix('.new')
            if temporary.exists():
                temporary.unlink()
            try:
                os.link(cached, temporary)
            except OSError:
                import shutil
                shutil.copyfile(cached, temporary)
            temporary.replace(target)
        print('[prepare] index ready', name, len(docs), flush=True)

if __name__ == '__main__':
    main()
