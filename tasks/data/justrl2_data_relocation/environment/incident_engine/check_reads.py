"""Fresh CPU behavior checks for submitted data queries; no model or training."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import secrets
import shutil
import sys
import tempfile

# -I -S removes the workspace and startup hooks from import search paths.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import regular, verify_dataset
from loader_contract import read_resolved
from resolve_worker import MAX_SOURCE, Query


def _check_records(name, root, rows, source):
    counts = Counter()
    identifiers = []
    errors = []
    for record in rows:
        try:
            payload, _ = read_resolved(root, record,
                lambda directory, row: Query(directory).run(source, row))
            identifiers.append(payload['id'])
            counts[record['difficulty']] += 1
        except Exception as error:
            if len(errors) < 4:
                errors.append(record['id'] + ': ' + type(error).__name__)
    expected = [record['id'] for record in rows]
    return {'name': name, 'passed': identifiers == expected,
            'readable_count': len(identifiers), 'easy': counts['easy'], 'hard': counts['hard'],
            'ids_sha256': hashlib.sha256(json.dumps(identifiers).encode()).hexdigest(), 'errors': errors}


def _relocate(root, rows, catalog, destination):
    """Change locations only; keep original IDs, hashes and payload bytes."""
    shutil.copytree(root, destination)
    originals = {row['id']: (root / row['path']).read_bytes() for row in catalog}
    locations = []
    for row in catalog:
        (destination / row['path']).unlink()
        name = 'storage/' + secrets.token_hex(8) + '/' + secrets.token_hex(8) + '.json'
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(originals[row['id']])
        locations.append(dict(row, path=name))
    # Existing stale paths contain a different real sample, so a path-exists
    # shortcut cannot avoid consulting current metadata.
    for index, record in enumerate(rows):
        decoy = destination / record['path']
        decoy.parent.mkdir(parents=True, exist_ok=True)
        decoy.write_bytes(originals[rows[(index + 1) % len(rows)]['id']])
    secrets.SystemRandom().shuffle(locations)
    (destination / 'catalog.json').write_text(json.dumps({'schema_version': 1, 'rows': locations}))


def _probe(name, root, record, source):
    try:
        Query(root).run(source, dict(record))
    except (FileNotFoundError, ValueError):
        return {'name': name, 'passed': True}
    except Exception as error:
        return {'name': name, 'passed': False, 'error': type(error).__name__}
    return {'name': name, 'passed': False, 'error': 'Returned a path for an unavailable/ambiguous record'}


def verify_submission(scene, loader, snapshot):
    result = {'schema_version': 1, 'mode': 'data-contract', 'score': False,
              'checked_records': 0, 'variants': [], 'probes': [], 'errors': [],
              'training_executed': False}
    try:
        scene, loader = Path(scene), Path(loader)
        info = verify_dataset(scene, snapshot)
        with regular(loader).open('rb') as stream:
            raw = stream.read(MAX_SOURCE + 1)
        if len(raw) > MAX_SOURCE:
            raise ValueError('Resolver source exceeds 64 KiB')
        source = raw.decode('utf-8')
        result['loader_sha256'] = hashlib.sha256(raw).hexdigest()
        result['snapshot_sha256'] = hashlib.sha256(regular(snapshot).read_bytes()).hexdigest()
        root, rows, catalog = scene / 'dataset', info['rows'], info['catalog']
        result['variants'].append(_check_records('original', root, rows, source))
        with tempfile.TemporaryDirectory(prefix='data-review-') as directory:
            scratch = Path(directory)
            relocated = scratch / 'relocated'
            _relocate(root, rows, catalog, relocated)
            result['variants'].append(_check_records('relocated', relocated, rows, source))
            easy = next(row for row in rows if row['difficulty'] == 'easy')
            hard = next(row for row in rows if row['difficulty'] == 'hard')
            wrong_hash = secrets.token_hex(32)
            while wrong_hash in {row['sha256'] for row in rows}:
                wrong_hash = secrets.token_hex(32)
            checks = [
                ('unknown', root, dict(hard, id='unknown-' + secrets.token_hex(8))),
                ('content', root, dict(hard, sha256=wrong_hash)),
                ('unknown_existing', root, dict(easy, id='unknown-' + secrets.token_hex(8))),
                # Another real record's hash catches "hash exists anywhere" checks.
                ('content_existing', root, dict(easy, sha256=hard['sha256'])),
            ]
            missing = scratch / 'missing'
            missing.mkdir()
            for name in ('catalog.json', 'manifest.json'):
                shutil.copyfile(root / name, missing / name)
            checks.append(('missing_current_file', missing, hard))
            ambiguous = scratch / 'ambiguous'
            shutil.copytree(root, ambiguous)
            duplicate = next(row for row in catalog if row['id'] == hard['id'])
            duplicate_path = 'duplicate/' + secrets.token_hex(8) + '.json'
            (ambiguous / 'duplicate').mkdir()
            shutil.copyfile(root / duplicate['path'], ambiguous / duplicate_path)
            (ambiguous / 'catalog.json').write_text(json.dumps({'schema_version': 1,
                'rows': catalog + [dict(duplicate, path=duplicate_path)]}))
            checks.append(('ambiguous_catalog', ambiguous, hard))
            result['probes'] = [_probe(name, directory, record, source)
                                for name, directory, record in checks]
        result['checked_records'] = sum(item['readable_count'] for item in result['variants'])
        result['score'] = (all(item['passed'] for item in result['variants'])
                           and all(item['passed'] for item in result['probes']))
        if not result['score']:
            result['errors'] = [item['name'] for item in result['variants'] + result['probes']
                                if not item['passed']]
        verify_dataset(scene, snapshot)
    except Exception as error:
        result['score'] = False
        result['errors'].append(type(error).__name__ + ': ' + str(error)[:300])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('scene', 'loader', 'snapshot'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    result = verify_submission(args.scene, args.loader, args.snapshot)
    print(json.dumps(result, allow_nan=False))
    return 0 if result['score'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
