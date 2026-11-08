"""Verify the submitted catalog path with fresh CPU production retrieval."""
import argparse
import ast
from collections import Counter
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import secrets
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request

HERE = Path(__file__).resolve().parent.parent


def sha(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            value.update(block)
    return value.hexdigest()


def regular(path):
    if path.is_symlink() or not path.is_file():
        raise ValueError('Expected a regular file: ' + str(path))
    return path


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        with os.fdopen(descriptor, 'w') as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write('\n')
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def fingerprint(documents):
    if not isinstance(documents, list):
        raise ValueError('Producer output must be a list')
    for doc in documents:
        if not isinstance(doc, dict) or set(doc) != {'id', 'contents'}:
            raise ValueError('Each document must contain id and contents')
        if not all(isinstance(doc[k], str) for k in doc):
            raise ValueError('Document identifiers and contents must be strings')
    return Counter(json.dumps(doc, sort_keys=True, ensure_ascii=False) for doc in documents)


def producer(source, rows, passages, scratch):
    inputs, output = scratch / (secrets.token_hex(8) + '.json'), scratch / (secrets.token_hex(8) + '.json')
    inputs.write_text(json.dumps({'rows': rows, 'passages': passages}, ensure_ascii=False))
    call = subprocess.run([sys.executable, '-I', str(HERE / 'tests/catalog_worker.py'),
                           str(source), str(inputs), str(output)],
                          cwd=scratch, capture_output=True, text=True, timeout=25)
    if call.returncode or not output.is_file() or output.stat().st_size > 32 * 1024**2:
        raise ValueError('Producer execution failed: ' + call.stderr[-700:])
    return json.loads(output.read_text())


def wrapper(source):
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == 'build':
            node.body = [ast.Pass()]
    return ast.dump(tree, include_attributes=False)


def original_index(spec, catalog, cache):
    import faiss
    details = {'corpus_sha256': spec['data_files']['shared/data/' + catalog + '.jsonl']['sha256'],
               'model_sha256': spec['encoder']['sha256'], 'prefix': 'passage: ',
               'max_length': 256, 'pooling': 'masked_mean_l2', 'index': 'FlatIP',
               'dtype': 'float32', 'rows': spec['catalog_rows'][catalog], 'dim': 768}
    key = hashlib.sha256(json.dumps(details, sort_keys=True).encode()).hexdigest()
    index = cache / 'hotpotqa_e5' / (key + '.index')
    marker = index.with_suffix('.complete.json')
    regular(index)
    if json.loads(regular(marker).read_text()) != details:
        raise ValueError('Prepared index specification mismatch')
    identities = json.loads(regular(Path(os.environ['AUTORLBENCH_PREPARED']) / 'indexes.json').read_text())
    identity = next((record for record in identities if record['catalog'] == catalog), None)
    if (identity is None or identity['cache_key'] != key or identity['sha256'] != sha(index)
            or identity['bytes'] != index.stat().st_size):
        raise ValueError('Prepared index byte identity mismatch')
    result = faiss.read_index(str(index))
    if result.ntotal != details['rows'] or result.d != 768 or result.metric_type != faiss.METRIC_INNER_PRODUCT:
        raise ValueError('Prepared index shape or metric mismatch')
    return result, index


def evaluate(workspace, results, spec):
    import faiss
    import numpy as np
    import pyarrow.parquet as pq
    checks = []
    report = regular(workspace / 'report.md')
    if not 0 < report.stat().st_size <= 1024**2:
        raise ValueError('A nonempty report.md of at most 1 MiB is required')
    for name, expected in spec['pinned_workspace_sha256'].items():
        if sha(regular(workspace / name)) != expected:
            raise ValueError('Unsupported workspace change: ' + name)
    config = json.loads(regular(workspace / 'config.json').read_text())
    expected = dict(spec['config'])
    selected = config.get('training_catalog')
    if selected not in ['catalog_001', 'catalog_002']:
        raise ValueError('Unsupported training_catalog')
    expected['training_catalog'] = selected
    if config != expected:
        raise ValueError('Only config.training_catalog may change')
    source = regular(workspace / 'catalog.py')
    if source.stat().st_size > 65536:
        raise ValueError('Producer source exceeds its budget')
    if wrapper(source.read_text()) != wrapper((HERE / 'resources/catalog.py').read_text()):
        raise ValueError('Only the build function body may change in catalog.py')
    identities = json.loads(regular(Path(os.environ['AUTORLBENCH_PREPARED']) / 'indexes.json').read_text())
    if {item['catalog'] for item in identities} != {'catalog_001', 'catalog_002'}:
        raise ValueError('Prepared index identities are incomplete')
    for identity in identities:
        path = regular(workspace / 'shared/data' / (identity['catalog'] + '.index'))
        if path.stat().st_size != identity['bytes'] or sha(path) != identity['sha256']:
            raise ValueError('Prepared workspace index changed: ' + identity['catalog'])
    checks.append('submission_and_runtime_integrity')
    data_cache = Path(os.environ['AUTORLBENCH_DATASETS'])
    model_cache = Path(os.environ['AUTORLBENCH_MODELS'])
    snapshot = data_cache / (spec['source']['repo_id'].replace('/', '__') + '--' + spec['source']['revision'])
    for name, item in spec['data_files'].items():
        for path in [snapshot / name, workspace / name]:
            regular(path)
            if path.stat().st_size != item['bytes'] or sha(path) != item['sha256']:
                raise ValueError('Pinned input changed: ' + str(path))
    encoder = spec['encoder']
    encoder_cache = model_cache / (encoder['repo'].replace('/', '__') + '--' + encoder['revision'])
    model = workspace / 'shared/models/encoder'
    for name in encoder['files']:
        reference, candidate = regular(encoder_cache / name), regular(model / name)
        if sha(reference) != encoder['file_sha256'][name] or sha(candidate) != encoder['file_sha256'][name]:
            raise ValueError('Encoder asset changed: ' + name)
    weight = model / 'model.safetensors'
    if weight.stat().st_size != encoder['bytes'] or sha(weight) != encoder['sha256']:
        raise ValueError('Encoder weights do not match the pinned configuration')
    actor = spec['actor']
    model_identities = json.loads(regular(Path(os.environ['AUTORLBENCH_PREPARED']) / 'models.json').read_text())
    identity = next((record for record in model_identities if record.get('repo') == actor['repo']), None)
    if (identity is None or any(identity.get(key) != actor[key] for key in ['revision', 'bytes', 'sha256'])
            or set(identity.get('file_sha256', {})) != set(actor['files'])
            or identity['file_sha256']['model.safetensors'] != actor['sha256']):
        raise ValueError('Prepared actor identity mismatch')
    actor_path = workspace / 'shared/models/actor'
    for name, expected_digest in identity['file_sha256'].items():
        if sha(regular(actor_path / name)) != expected_digest:
            raise ValueError('Actor asset changed: ' + name)
    checks.append('immutable_inputs')
    base = jsonl(snapshot / 'shared/data/catalog_002.jsonl')
    expected_documents = fingerprint(base)
    rows = pq.read_table(snapshot / 'shared/data/train.parquet').to_pylist()
    with tempfile.TemporaryDirectory(prefix='catalog-check-', dir=results) as name:
        scratch = Path(name)
        if selected == 'catalog_001':
            active = producer(source, rows, base, scratch)
            if fingerprint(active) != expected_documents:
                raise ValueError('Effective training catalog does not preserve exactly the source passages')
            replaced = json.loads(json.dumps(rows))
            for row in replaced:
                row['reward_model']['ground_truth']['target'] = [secrets.token_hex(24)]
            if fingerprint(producer(source, replaced, base, scratch)) != expected_documents:
                raise ValueError('Effective catalog depends on scoring targets')
            for count in [0, 3]:
                probes = [{'extra_info': {'question': 'Question ' + secrets.token_hex(10)},
                           'reward_model': {'ground_truth': {'target': [secrets.token_hex(24)]}}}
                          for _ in range(count)]
                if fingerprint(producer(source, probes, base[:7], scratch)) != fingerprint(base[:7]):
                    raise ValueError('Producer fails additional valid inputs')
        else:
            active = base
        checks.append('effective_catalog_provenance')
        baseline, baseline_path = original_index(spec, 'catalog_002', data_cache)
        vectors = baseline.reconstruct_n(0, baseline.ntotal)
        if not np.isfinite(vectors).all() or not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=.002):
            raise ValueError('Prepared source vectors are not finite unit vectors')
        by_id = {doc['id']: i for i, doc in enumerate(base)}
        if len(by_id) != len(base):
            raise ValueError('Prepared source identifiers are not unique')
        service = scratch / 'workspace'
        data = service / 'shared/data'
        data.mkdir(parents=True)
        for filename in ['serve.py', 'encoder.py']:
            shutil.copyfile(workspace / filename, service / filename)
        # Model assets remain immutable inputs, referenced by real files in a fresh service tree.
        def link_or_copy(source, destination):
            try:
                os.link(source, destination)
            except OSError:
                shutil.copyfile(source, destination)
        service_model = service / 'shared/models/encoder'
        service_model.mkdir(parents=True)
        for filename in encoder['files']:
            link_or_copy(model / filename, service_model / filename)
        (data / 'catalog_002.jsonl').write_text(''.join(json.dumps(doc, ensure_ascii=False) + '\n' for doc in base))
        shutil.copyfile(baseline_path, data / 'catalog_002.index')
        if selected == 'catalog_001':
            (data / 'catalog_001.jsonl').write_text(''.join(json.dumps(doc, ensure_ascii=False) + '\n' for doc in active))
            fresh_index = faiss.IndexFlatIP(768)
            fresh_index.add(np.ascontiguousarray(vectors[[by_id[doc['id']] for doc in active]]))
            faiss.write_index(fresh_index, str(data / 'catalog_001.index'))
        else:
            _, inactive_path = original_index(spec, 'catalog_001', data_cache)
            shutil.copyfile(snapshot / 'shared/data/catalog_001.jsonl', data / 'catalog_001.jsonl')
            shutil.copyfile(inactive_path, data / 'catalog_001.index')
        checks.append('fresh_index_mapping')
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            port = sock.getsockname()[1]
        env = dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_DATASETS_OFFLINE='1',
                   PYTHONDONTWRITEBYTECODE='1', TOKENIZERS_PARALLELISM='false',
                   OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1', NO_PROXY='127.0.0.1,localhost',
                   no_proxy='127.0.0.1,localhost')
        for key in ['PYTHONPATH', 'PYTHONHOME', 'HF_TOKEN', 'HUGGING_FACE_HUB_TOKEN']:
            env.pop(key, None)
        launch = ('import runpy,sys;sys.path.insert(0,sys.argv[1]);'
                  'sys.argv=sys.argv[2:];runpy.run_path(sys.argv[0],run_name="__main__")')
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with (scratch / 'service.log').open('w') as log:
            child = subprocess.Popen([sys.executable, '-I', '-c', launch, str(service), str(service / 'serve.py'),
                                      '--port', str(port), '--trace', str(scratch / 'retrieval.jsonl')],
                                     cwd=service, env=env, stdout=log, stderr=subprocess.STDOUT,
                                     start_new_session=True)
            try:
                for _ in range(90):
                    if child.poll() is not None:
                        raise ValueError('Production service exited: ' + (scratch / 'service.log').read_text()[-1000:])
                    try:
                        opener.open(f'http://127.0.0.1:{port}/health', timeout=1).close()
                        break
                    except OSError:
                        time.sleep(1)
                else:
                    raise ValueError('Production service readiness timeout')
                query_rows = secrets.SystemRandom().sample(rows, 8)
                queries = []
                for row in query_rows:
                    question = (row.get('extra_info') or {}).get('question')
                    if not question:
                        matches = re.findall(r'(?:Question:|question:)\s*(.*)', row['prompt'][-1]['content'], re.S)
                        if not matches:
                            raise ValueError('Prepared row has no question text')
                        question = matches[-1]
                    queries.append(question.strip())
                query_encoder = module(HERE / 'resources/encoder.py', 'reference_encoder').Encoder(model, 'cpu', threads=4)
                query_vectors = query_encoder.encode(queries, 'query: ')
                cutoffs = baseline.search(query_vectors, 3)[0][:, -1]
                for catalog in dict.fromkeys([selected, config['validation_catalog']]):
                    endpoint = f'http://127.0.0.1:{port}/{catalog}/verification/retrieve'
                    for inputs in [[], queries]:
                        request = urllib.request.Request(endpoint, data=json.dumps({
                            'queries': inputs, 'topk': 3, 'return_scores': True}).encode(),
                            headers={'Content-Type': 'application/json'})
                        with opener.open(request, timeout=60) as response:
                            body = json.load(response)
                        if not isinstance(body, dict) or len(body.get('result', [])) != len(inputs):
                            raise ValueError('Retrieval cardinality mismatch')
                        for number, hits in enumerate(body['result']):
                            if len(hits) != 3 or len({hit['document']['id'] for hit in hits}) != 3:
                                raise ValueError('Retrieval must return three distinct sources')
                            previous = float('inf')
                            for hit in hits:
                                doc, score = hit['document'], float(hit['score'])
                                if doc.get('id') not in by_id or doc != base[by_id[doc['id']]] or not np.isfinite(score):
                                    raise ValueError('Tool observation is not a valid source passage')
                                expected_score = float(vectors[by_id[doc['id']]] @ query_vectors[number])
                                if abs(score - expected_score) > 1e-4 or score < float(cutoffs[number]) - 1e-4 or score > previous + 1e-4:
                                    raise ValueError('Tool observation does not match actual e5 top-k retrieval')
                                previous = score
                checks.append('production_http_retrieval')
            finally:
                if child.poll() is None:
                    os.killpg(child.pid, signal.SIGTERM)
                child.wait(timeout=20)
        return {'checks': checks, 'selected_catalog': selected, 'source_documents': len(base),
                'retrieval_queries': len(queries), 'training_rerun': False, 'gpu_count': 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, default=Path(os.environ.get('AUTORLBENCH_WORKSPACE', '/workspace')))
    parser.add_argument('--results', type=Path, default=Path(os.environ.get('AUTORLBENCH_RESULTS', '/results')))
    args = parser.parse_args()
    results = args.results.absolute()
    if results.is_symlink():
        raise ValueError('Results root must not be a symlink')
    results.mkdir(parents=True, exist_ok=True)
    metrics = {'task_id': 'search_r1_hotpotqa', 'errors': []}
    try:
        spec = json.loads((HERE / 'resources/grader_spec.json').read_text())
        if spec['schema_version'] != 1 or spec['required_score'] != 1.0:
            raise ValueError('Invalid grader specification')
        metrics.update(evaluate(args.workspace.absolute(), results, spec))
        metrics['passed'] = len(metrics['checks']) == 5
    except Exception as error:
        metrics['passed'] = False
        metrics['errors'].append(type(error).__name__ + ': ' + str(error))
    score = float(metrics['passed'])
    result = {'passed': metrics['passed'], 'score': score, 'metrics': metrics}
    for name, value in [('metrics.json', metrics), ('result.json', result),
                        ('experiment_metrics.json', {'metrics': {'functional_repair': score}})]:
        write_json(results / name, value)
    print(json.dumps(result, allow_nan=False))
    return 0 if metrics['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
