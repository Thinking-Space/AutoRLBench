#!/usr/bin/env python3
"""Reconstruct the frozen math runtime store from four pinned public shards.

Only prompt/answer content comes from the public dataset. Historical accuracies
and easy/hard assignments are explicitly frozen, author-supplied metadata.
"""
import argparse
import collections
import hashlib
import json
import math
from pathlib import Path
import re
import urllib.request


HERE = Path(__file__).resolve().parent
DEFAULT_SELECTION = HERE / 'selection.json'
DEFAULT_LOCK = HERE / 'sources.lock.json'


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _pretty(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True,
                       indent=2, allow_nan=False) + '\n').encode('utf-8')


def _content_hash(value):
    return _sha(json.dumps(value, ensure_ascii=False, sort_keys=True,
                           separators=(',', ':'), allow_nan=False).encode('utf-8'))


def _digest(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def _empty_output(out):
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise FileExistsError('Output must be absent or empty: %s' % out)


def _load_inputs(selection_path, lock_path):
    selection_raw = selection_path.read_bytes()
    lock_raw = lock_path.read_bytes()
    selection = json.loads(selection_raw)
    lock = json.loads(lock_raw)
    for name, value in [('selection', selection), ('source lock', lock)]:
        _require(isinstance(value, dict) and type(value.get('schema_version')) is int
                 and value['schema_version'] == 1, 'Invalid %s schema_version' % name)
    identity = {key: lock.get(key) for key in ('repo', 'revision', 'config', 'split')}
    _require(isinstance(identity['repo'], str) and re.fullmatch(r'[\w.-]+/[\w.-]+', identity['repo']),
             'Invalid source repository')
    _require(isinstance(identity['revision'], str) and re.fullmatch('[0-9a-f]{40}', identity['revision']),
             'Source revision must be a full commit SHA')
    _require(identity['config'] == 'Math' and identity['split'] == 'train', 'Expected Math/train source')
    _require(selection.get('source') == identity, 'Selection and lock source identity mismatch')
    files = lock.get('files')
    _require(isinstance(files, list) and len(files) == 4, 'Expected exactly four public Math shards')
    for index, item in enumerate(files, 1):
        _require(isinstance(item, dict) and item.get('path') == 'data/Math/Math_part-%d-of-4.jsonl' % index,
                 'Expected four public Math shards in numbered file order')
        _require(type(item.get('size')) is int and item['size'] > 0, 'Invalid shard size')
        _require(type(item.get('rows')) is int and item['rows'] > 0, 'Invalid shard record count')
        _require(_digest(item.get('sha256')), 'Invalid shard SHA256')
    strata = selection.get('historical_strata')
    _require(isinstance(strata, dict) and strata.get('kind') == 'frozen_author_supplied',
             'Historical strata must be frozen_author_supplied')
    _require(isinstance(strata.get('description'), str)
             and 'not public dataset measurements' in strata['description'],
             'Historical strata must explain these are not public dataset measurements')
    _require(strata.get('easy_min') == 0.875 and strata.get('hard_max') == 0.125,
             'Historical stratum thresholds must retain the original contract')
    counts = selection.get('expected_counts')
    _require(isinstance(counts, dict) and set(counts) == {'train', 'eval'}, 'Invalid expected split counts')
    for split in ('train', 'eval'):
        group = counts[split]
        _require(isinstance(group, dict) and set(group) == {'easy', 'hard'}
                 and all(type(v) is int and v > 0 for v in group.values()), 'Invalid expected group counts')
        _require(group['easy'] == group['hard'], 'Expected equal easy/hard group counts')
    rows = selection.get('rows')
    _require(isinstance(rows, list) and len(rows) == sum(sum(g.values()) for g in counts.values()),
             'Insufficient or unexpected selected record count')
    seen_ids, seen_uuids, observed = set(), set(), collections.Counter()
    for row in rows:
        _require(isinstance(row, dict), 'Invalid selected record')
        required = {'id', 'uuid', 'split', 'difficulty', 'source_line', 'historical_accuracy',
                    'file', 'row_index_0based', 'public_record_sha256',
                    'prompt_sha256', 'label_sha256', 'content_sha256'}
        _require(required <= row.keys(), 'Selected record lacks required fields')
        _require(isinstance(row['id'], str) and isinstance(row['uuid'], str), 'Invalid selected identity')
        _require(row['id'] not in seen_ids and row['uuid'] not in seen_uuids, 'Duplicate selected record')
        seen_ids.add(row['id'])
        seen_uuids.add(row['uuid'])
        _require(row['split'] in ('train', 'eval') and row['difficulty'] in ('easy', 'hard'),
                 'Invalid selected split/difficulty')
        line = row['source_line']
        _require(type(line) is int and line > 0, 'Invalid selected source_line')
        _require(row['id'] == '%s-%08d' % (row['difficulty'], line), 'Selected ID/source_line mismatch')
        _require(row['uuid'] == 'Math_%05d' % line, 'Selected UUID/global order mismatch')
        _require(row['file'] in {item['path'] for item in files}, 'Selected public file is not locked')
        _require(type(row['row_index_0based']) is int and row['row_index_0based'] >= 0,
                 'Invalid selected file row offset')
        accuracy = row['historical_accuracy']
        _require(type(accuracy) in (int, float) and math.isfinite(accuracy) and 0 <= accuracy <= 1,
                 'Invalid frozen historical_accuracy')
        _require((row['difficulty'] == 'easy' and accuracy >= 0.875)
                 or (row['difficulty'] == 'hard' and accuracy <= 0.125),
                 'Frozen historical_accuracy does not match difficulty')
        for key in ('public_record_sha256', 'prompt_sha256', 'label_sha256', 'content_sha256'):
            _require(_digest(row[key]), 'Invalid selected %s' % key)
        observed[(row['split'], row['difficulty'])] += 1
    _require(observed == collections.Counter({(split, difficulty): n for split, groups in counts.items()
                                              for difficulty, n in groups.items()}), 'Selected stratum count mismatch')
    expected = selection.get('expected_runtime_sha256')
    paths = {'manifest.json', 'eval.jsonl'} | {
        'live/%s/%s.json' % (row['difficulty'], row['id']) for row in rows if row['split'] == 'train'}
    _require(isinstance(expected, dict) and set(expected) == paths, 'Expected runtime file set mismatch')
    _require(all(_digest(value) for value in expected.values()), 'Invalid expected runtime SHA256')
    return selection, lock, {'selection_sha256': _sha(selection_raw), 'sources_lock_sha256': _sha(lock_raw)}


def _verify_asset(raw, item):
    _require(len(raw) == item['size'], 'Public shard size mismatch: %s' % item['path'])
    _require(_sha(raw) == item['sha256'], 'Public shard SHA256 mismatch: %s' % item['path'])


def _read_shards(source_dir, lock, download):
    shards = {}
    global_line = 0
    for item in lock['files']:
        path = source_dir / Path(item['path']).name
        if not path.exists() and download:
            url = 'https://huggingface.co/datasets/{}/resolve/{}/{}'.format(
                lock['repo'], lock['revision'], item['path'])
            request = urllib.request.Request(url, headers={'User-Agent': 'AutoRLBench-public-math/1'})
            with urllib.request.urlopen(request, timeout=60) as response:
                raw = response.read()
            _verify_asset(raw, item)
            source_dir.mkdir(parents=True, exist_ok=True)
            with path.open('xb') as stream:
                stream.write(raw)
        else:
            raw = path.read_bytes()
            _verify_asset(raw, item)
        lines = raw.splitlines(keepends=True)
        _require(len(lines) == item['rows'] and all(line.strip() for line in lines),
                 'Public shard record count mismatch: %s' % item['path'])
        values = []
        for line in lines:
            record = json.loads(line)
            global_line += 1
            _require(isinstance(record, dict) and record.get('uuid') == 'Math_%05d' % global_line,
                     'Public record UUID/global order mismatch')
            _require(record.get('domain') == 'Math' and isinstance(record.get('query'), str)
                     and isinstance(record.get('ground_truth'), str), 'Invalid public Math record fields')
            values.append((record, _sha(line), global_line))
        shards[item['path']] = values
    return shards


def _reconstruct(selection, shards):
    runtime, manifest, evaluation, provenance = {}, [], [], []
    for selected in selection['rows']:
        values = shards[selected['file']]
        offset = selected['row_index_0based']
        _require(offset < len(values), 'Insufficient public records for selected file offset')
        record, record_hash, global_line = values[offset]
        _require(selected['source_line'] == global_line and selected['uuid'] == record['uuid'],
                 'Selected public global order mismatch: %s' % selected['id'])
        _require(selected['public_record_sha256'] == record_hash,
                 'Public record SHA256 mismatch: %s' % selected['id'])
        prompt = [{'role': 'user', 'content': record['query']}]
        label = record['ground_truth']
        content = {'prompt': prompt, 'label': label}
        for field, value in [('prompt', prompt), ('label', label), ('content', content)]:
            key = field + '_sha256'
            _require(_content_hash(value) == selected[key], '%s mismatch: %s' % (key, selected['id']))
        payload = {'id': selected['id'], 'prompt': prompt, 'label': label,
                   'metadata': {'difficulty': selected['difficulty'], 'source_line': global_line,
                                'historical_accuracy': selected['historical_accuracy']}}
        if selected['split'] == 'train':
            path = 'live/%s/%s.json' % (selected['difficulty'], selected['id'])
            runtime[path] = _pretty(payload)
            manifest.append({'id': selected['id'], 'difficulty': selected['difficulty'], 'path': path,
                             'sha256': _sha(runtime[path]), 'source_line': global_line, 'split': 'train'})
        else:
            evaluation.append(json.dumps(payload, ensure_ascii=False, sort_keys=True, allow_nan=False) + '\n')
        row = {key: selected[key] for key in ('id', 'split', 'difficulty', 'source_line',
                                             'prompt_sha256', 'label_sha256', 'content_sha256')}
        row.update(source_row_id=record['uuid'], public_file=selected['file'],
                   row_index_0based=offset, public_record_sha256=record_hash)
        provenance.append(row)
    runtime['manifest.json'] = _pretty({'schema_version': 1, 'rows': manifest})
    runtime['eval.jsonl'] = ''.join(evaluation).encode('utf-8')
    _require(set(runtime) == set(selection['expected_runtime_sha256']), 'Reconstructed runtime file set mismatch')
    for name, raw in runtime.items():
        _require(_sha(raw) == selection['expected_runtime_sha256'][name], 'Runtime SHA256 mismatch: %s' % name)
    return runtime, provenance


def prepare(source_dir, out, *, selection_path=DEFAULT_SELECTION, lock_path=DEFAULT_LOCK, download=False):
    """Validate inputs, write an absent/empty store, and return its public receipt.

    Input/identity/hash validation finishes before any output file is created.
    Downloading is opt-in; existing local shards must always match their pins.
    """
    source_dir, out = Path(source_dir), Path(out)
    _empty_output(out)
    selection, lock, inputs = _load_inputs(Path(selection_path), Path(lock_path))
    shards = _read_shards(source_dir, lock, download)
    runtime, provenance = _reconstruct(selection, shards)
    public_manifest = dict(selection['source'], files=[
        {key: item[key] for key in ('path', 'size', 'sha256', 'rows')} for item in lock['files']])
    source = dict(public_manifest, sha256=_content_hash(public_manifest),
                  sha256_meaning='SHA256 of the canonical public source manifest: repo, revision, config, split, files.',
                  fileshashes={item['path']: item['sha256'] for item in lock['files']})
    case = {'schema_version': 1, 'synthetic_fault_injection': True, 'source': source,
            'difficulty_basis': selection['historical_strata'], 'rows': provenance,
            'train_per_group': selection['expected_counts']['train']['easy'],
            'eval_per_group': selection['expected_counts']['eval']['easy']}
    case_raw = _pretty(case)
    receipt = {
        'schema_version': 1, 'synthetic_fault_injection': True,
        'source': source, 'inputs': inputs,
        'preparer': {'file': 'prepare_public.py', 'sha256': _sha(Path(__file__).read_bytes())},
        'historical_strata': selection['historical_strata'],
        'runtime_sha256': {name: _sha(raw) for name, raw in runtime.items()},
        'runtime_sizes': {name: len(raw) for name, raw in runtime.items()},
        'case_sha256': _sha(case_raw), 'case_is_new_public_provenance': True,
        'verification': {'runtime_files': len(runtime), 'selected_pairs': len(provenance),
                         'content_hashes': 3 * len(provenance),
                         'counts': selection['expected_counts'],
                         'question_normalization': 'none', 'answer_normalization': 'none',
                         'scope': 'Runtime bytes match the frozen author-supplied expected hashes; CASE is new public provenance.'},
    }
    # Recheck every pinned input before publishing the completed store.
    for item in lock['files']:
        _verify_asset((source_dir / Path(item['path']).name).read_bytes(), item)
    _empty_output(out)
    out.mkdir(parents=True, exist_ok=True)
    for name, raw in list(runtime.items()) + [('CASE.json', case_raw), ('PUBLIC_PROVENANCE.json', _pretty(receipt))]:
        path = out / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as stream:
            stream.write(raw)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, required=True, help='Directory containing the four Math_part-*.jsonl shards')
    parser.add_argument('--out', type=Path, required=True, help='Absent or empty output directory')
    parser.add_argument('--download', action='store_true', help='Download missing shards from their pinned public revisions')
    parser.add_argument('--selection', type=Path, default=DEFAULT_SELECTION)
    parser.add_argument('--lock', type=Path, default=DEFAULT_LOCK)
    args = parser.parse_args()
    try:
        receipt = prepare(args.source_dir, args.out, selection_path=args.selection,
                          lock_path=args.lock, download=args.download)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps({'runtime_files': receipt['verification']['runtime_files'],
                      'selected_pairs': receipt['verification']['selected_pairs'],
                      'content_hashes': receipt['verification']['content_hashes'],
                      'source_sha256': receipt['source']['sha256'],
                      'manifest_sha256': receipt['runtime_sha256']['manifest.json'],
                      'eval_sha256': receipt['runtime_sha256']['eval.jsonl'],
                      'case_sha256': receipt['case_sha256'],
                      'receipt_sha256': _sha((args.out / 'PUBLIC_PROVENANCE.json').read_bytes())}, sort_keys=True))


if __name__ == '__main__':
    main()
