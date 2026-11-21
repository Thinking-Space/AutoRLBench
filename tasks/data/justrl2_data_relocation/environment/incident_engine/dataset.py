"""Validate the small, pinned dataset without loading model checkpoints."""
from collections import Counter
import hashlib
import json
from pathlib import Path, PurePosixPath
import stat


def regular(path):
    path = Path(path)
    if path.is_symlink() or not stat.S_ISREG(path.lstat().st_mode):
        raise ValueError('Expected a regular file: ' + path.name)
    return path


def inside(root, name):
    if not isinstance(name, str) or not name:
        raise ValueError('Expected a relative path string')
    relative = PurePosixPath(name)
    if (relative.is_absolute() or '..' in relative.parts or '\\' in name
            or relative.as_posix() != name):
        raise ValueError('Expected a canonical relative path')
    path = root
    for part in relative.parts:
        path /= part
        if path.is_symlink():
            raise ValueError('Dataset symlinks are forbidden')
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError('Dataset path escapes its root')
    return regular(path)


def read_json(path):
    return json.loads(regular(path).read_text(encoding='utf-8'))


def verify_dataset(scene, snapshot):
    scene = Path(scene)
    if scene.is_symlink() or not scene.is_dir():
        raise ValueError('Expected a real scene directory')
    pin = read_json(snapshot)
    if pin.get('schema_version') != 1 or pin.get('mode') != 'data-contract':
        raise ValueError('Expected a data-contract snapshot')
    expected = pin['files']
    if not isinstance(expected, dict) or not expected:
        raise ValueError('Empty dataset inventory')
    actual = {}
    for path in scene.rglob('*'):
        if path.is_symlink():
            raise ValueError('Dataset symlinks are forbidden')
        if path.is_dir():
            continue
        regular(path)
        raw = path.read_bytes()
        actual[path.relative_to(scene).as_posix()] = {
            'size': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}
    if actual != expected:
        raise ValueError('Dataset file inventory/hash mismatch')
    root = scene / 'dataset'
    rows = read_json(root / 'manifest.json')['rows']
    catalog = read_json(root / 'catalog.json')['rows']
    if len(rows) != 64 or len({row['id'] for row in rows}) != 64:
        raise ValueError('Expected 64 distinct training records')
    counts = dict(Counter(row['difficulty'] for row in rows))
    if counts != {'easy': 32, 'hard': 32} or counts != pin['train_counts']:
        raise ValueError('Unexpected training mixture')
    locations = {row['id']: row for row in catalog}
    if len(catalog) != len(rows) or set(locations) != {row['id'] for row in rows}:
        raise ValueError('Catalog does not cover original IDs uniquely')
    paths = set()
    for row in rows:
        current = locations[row['id']]
        if current['sha256'] != row['sha256'] or current['difficulty'] != row['difficulty']:
            raise ValueError('Catalog changes content or difficulty')
        path = inside(root, current['path'])
        if path in paths:
            raise ValueError('Duplicate backing file')
        paths.add(path)
        raw = path.read_bytes()
        value = json.loads(raw)
        if hashlib.sha256(raw).hexdigest() != row['sha256']:
            raise ValueError('Backing content hash differs')
        if value['id'] != row['id'] or value['metadata']['difficulty'] != row['difficulty']:
            raise ValueError('Backing identity/difficulty differs')
        if not isinstance(value.get('prompt'), list) or 'label' not in value:
            raise ValueError('Missing math prompt/answer')
    evaluation = [json.loads(line) for line in (root / 'eval.jsonl').read_text().splitlines() if line.strip()]
    if len(evaluation) != 16 or len({row['id'] for row in evaluation}) != 16:
        raise ValueError('Expected 16 distinct evaluation records')
    if {row['id'] for row in rows} & {row['id'] for row in evaluation}:
        raise ValueError('Training/evaluation IDs overlap')
    if dict(Counter(row['metadata']['difficulty'] for row in evaluation)) != pin['eval_counts']:
        raise ValueError('Unexpected evaluation mixture')
    return {'rows': rows, 'catalog': catalog, 'train_counts': counts,
            'eval_count': len(evaluation), 'file_count': len(actual),
            'size_bytes': sum(value['size'] for value in actual.values())}
