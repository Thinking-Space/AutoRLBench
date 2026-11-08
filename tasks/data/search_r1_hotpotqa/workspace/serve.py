"""Local e5 / FAISS HTTP service."""
import argparse
import json
import threading
import time
from pathlib import Path
import faiss
from fastapi import FastAPI, Request
import uvicorn
from encoder import Encoder

ROOT = Path(__file__).resolve().parent

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--trace', type=Path, required=True)
    args = parser.parse_args()
    faiss.omp_set_num_threads(1)
    encoder = Encoder(ROOT / 'shared/models/encoder', 'cpu', threads=4)
    indexes, catalogs = {}, {}
    for name in ['catalog_001', 'catalog_002']:
        folder = ROOT / 'shared/data'
        indexes[name] = faiss.read_index(str(folder / (name + '.index')))
        catalogs[name] = [json.loads(line) for line in (folder / (name + '.jsonl')).read_text().splitlines()]
        assert indexes[name].ntotal == len(catalogs[name])
    app, lock = FastAPI(), threading.Lock()
    args.trace.parent.mkdir(parents=True, exist_ok=True)

    @app.get('/health')
    def health():
        return {'ready': True, 'retriever': 'e5/FlatIP', 'device': 'cpu'}

    @app.post('/{catalog}/{phase}/retrieve')
    async def retrieve(catalog: str, phase: str, request: Request):
        if catalog not in indexes:
            raise ValueError('Unknown catalog')
        payload = await request.json()
        queries, topk = payload['queries'], int(payload.get('topk') or 3)
        assert 1 <= topk <= 10
        if not queries:
            return {'result': []}
        result = []
        with lock:
            vectors = encoder.encode(queries, 'query: ')
            scores, ids = indexes[catalog].search(vectors, topk)
            with args.trace.open('a', encoding='utf-8') as stream:
                for query, values, positions in zip(queries, scores, ids):
                    entries = [{'document': catalogs[catalog][int(i)], 'score': float(s)} for s, i in zip(values, positions)]
                    stream.write(json.dumps({'time': time.time(), 'catalog': catalog, 'phase': phase, 'query': query, 'results': entries}, ensure_ascii=False) + '\n')
                    result.append(entries if payload.get('return_scores') else [entry['document'] for entry in entries])
        return {'result': result}

    uvicorn.run(app, host='127.0.0.1', port=args.port)

if __name__ == '__main__':
    main()
