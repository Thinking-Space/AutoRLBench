"""Assemble a labeled, already-relocated incident using only the standard library.

Checkpoint bytes are copied, never linked or deserialized. Inventories establish
byte identity, not DCP interpretability or successful model/optimizer restoration.
The original cursor (including its arm) is immutable. The resume driver owns arm
selection. No training, model conversion, downloads, or environment capture occur.
"""
from collections import Counter
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import stat
import sys
import tarfile


DEFAULT_SELECTION = Path(__file__).resolve().parents[1] / 'public_data/selection.json'
ITERATION = 3
SELECTED = 'iter_0000003'
TRACKER = 'latest_checkpointed_iteration.txt'
CURSOR = 'rollout/data_state_3.json'
EVIDENCE = ('INPUTS.json', 'run/effective_config.json', 'run/events.jsonl')


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _relative(value):
    _require(isinstance(value, str) and value and '\x00' not in value and '\\' not in value,
             'Expected a nonempty POSIX relative path')
    path = PurePosixPath(value)
    _require(not path.is_absolute() and '..' not in path.parts and path.parts
             and path.as_posix() == value, 'Path must be canonical and stay under its root')
    return path


def _path(root, name):
    relative = _relative(name)
    current = root
    _require(not root.is_symlink(), 'Root must not be a symlink')
    for part in relative.parts:
        current /= part
        _require(not current.is_symlink(), 'Symlinks are not permitted: ' + name)
    _require(current.resolve().is_relative_to(root.resolve()), 'Path escapes root: ' + name)
    return current


def _root(path):
    path = Path(path)
    _require(not path.is_symlink(), 'Root must not be a symlink')
    path = path.resolve(strict=True)
    _require(path.is_dir(), 'Expected a directory')
    return path


def _output(source, destination):
    destination = Path(destination)
    _require(not destination.is_symlink(), 'Destination must not be a symlink')
    destination = destination.resolve()
    _require(not destination.is_relative_to(source) and not source.is_relative_to(destination),
             'Source and destination must be separate directories')
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise FileExistsError('Destination must be absent or empty: ' + str(destination))
    return destination


def _sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _info(path):
    _require(stat.S_ISREG(path.lstat().st_mode), 'Expected a regular file: ' + path.name)
    return {'size': path.stat().st_size, 'sha256': _sha(path)}


def _inventory(root):
    result = {}

    def visit(directory):
        for path in sorted(directory.iterdir()):
            mode = path.lstat().st_mode
            if stat.S_ISDIR(mode):
                visit(path)
            else:
                name = path.relative_to(root).as_posix()
                _relative(name)
                result[name] = _info(path)

    visit(root)
    return dict(sorted(result.items()))


def _json(path):
    _require(path.is_file() and not path.is_symlink(), 'Missing regular JSON file: ' + path.name)
    value = json.loads(path.read_text(encoding='utf-8'))
    _require(isinstance(value, dict), 'Expected JSON object: ' + path.name)
    return value


def _lines(path):
    with path.open(encoding='utf-8') as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line)
                _require(isinstance(row, dict), 'Expected JSONL object: ' + path.name)
                yield row


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + '\n')


def _canonical_sha(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()


def _by_id(rows, label):
    _require(isinstance(rows, list) and rows, label + ' must contain rows')
    result = {}
    for row in rows:
        _require(isinstance(row, dict) and isinstance(row.get('id'), str) and row['id'],
                 label + ' has an invalid ID')
        _require(row['id'] not in result, label + ' has duplicate IDs')
        result[row['id']] = row
    return result


def _moved(name, move_subdir):
    path = _relative(name)
    return (PurePosixPath(move_subdir) / path.relative_to('live/hard')).as_posix() \
        if path.is_relative_to('live/hard') else name


def _dataset(root, selection_path, *, moved=None, require_stale=True):
    """Bind original and current files to selected IDs, source rows, and bytes."""
    selection = _json(selection_path)
    manifest = _json(_path(root, 'manifest.json'))
    case = _json(_path(root, 'CASE.json'))
    public = _json(_path(root, 'PUBLIC_PROVENANCE.json'))
    for label, value in [('selection', selection), ('manifest', manifest), ('CASE', case), ('PUBLIC', public)]:
        _require(type(value.get('schema_version')) is int and value['schema_version'] == 1,
                 label + ' schema_version must be 1')
    _require(case.get('synthetic_fault_injection') is True
             and public.get('synthetic_fault_injection') is True, 'Dataset must label its synthetic fault')
    _require(case.get('source') == public.get('source'), 'CASE and public source differ')
    for key, value in selection['source'].items():
        _require(case['source'].get(key) == value, 'Dataset public source differs from selection')
    _require(public['inputs']['selection_sha256'] == _sha(selection_path), 'Selection SHA256 mismatch')
    _require(public['case_sha256'] == _sha(root / 'CASE.json'), 'CASE SHA256 mismatch')
    selected = _by_id(selection['rows'], 'selection')
    case_rows = _by_id(case['rows'], 'CASE')
    rows = _by_id(manifest['rows'], 'manifest')
    train = {key: row for key, row in selected.items() if row['split'] == 'train'}
    evaluation = {key: row for key, row in selected.items() if row['split'] == 'eval'}
    _require(list(rows) == list(train) and set(case_rows) == set(selected), 'Dataset selected ID/order mismatch')
    _require(len(train) + len(evaluation) == len(selected), 'Unexpected selected split')
    counts = {split: dict(Counter(row['difficulty'] for row in selected.values() if row['split'] == split))
              for split in ('train', 'eval')}
    _require(counts == selection['expected_counts'] == public['verification']['counts'],
             'Dataset selected counts mismatch')
    _require(set(counts['train']) == {'easy', 'hard'}, 'Both easy and hard training groups are required')
    expected = selection['expected_runtime_sha256']
    _require(public['runtime_sha256'] == expected, 'Public runtime inventory differs from selection')
    _require(set(expected) == {row['path'] for row in rows.values()} | {'manifest.json', 'eval.jsonl'},
             'Selected runtime inventory is incomplete')
    for name, digest in expected.items():
        current = _moved(name, moved) if moved else name
        info = _info(_path(root, current))
        _require(info['sha256'] == digest and info['size'] == public['runtime_sizes'][name],
                 'Dataset SHA256/size mismatch: ' + name)
    eval_rows = _by_id(list(_lines(root / 'eval.jsonl')), 'eval') if evaluation else {}
    _require(list(eval_rows) == list(evaluation), 'Evaluation selected ID/order mismatch')
    catalog_rows = []
    for identifier, original in selected.items():
        provenance = case_rows[identifier]
        for key in ('split', 'difficulty', 'source_line', 'prompt_sha256', 'label_sha256', 'content_sha256',
                    'row_index_0based', 'public_record_sha256'):
            _require(provenance.get(key) == original.get(key), 'CASE selected metadata mismatch: ' + identifier)
        _require(provenance.get('source_row_id') == original['uuid']
                 and provenance.get('public_file') == original['file'], 'CASE selected source ID mismatch')
        if identifier in train:
            record = rows[identifier]
            for key in ('split', 'difficulty', 'source_line'):
                _require(record.get(key) == original[key], 'Manifest selected metadata mismatch')
            name = record['path']
            _require(_relative(name).is_relative_to('live/' + record['difficulty']),
                     'Initial manifest must use live difficulty paths')
            _require(record['sha256'] == expected[name], 'Manifest backing SHA256 mismatch')
            current = _moved(name, moved) if moved else name
            payload = _json(_path(root, current))
            catalog_rows.append(dict(record, path=current))
            if moved and record['difficulty'] == 'hard' and require_stale:
                _require(not _path(root, name).exists(), 'Hard manifest read is not stale')
        else:
            payload = eval_rows[identifier]
        _require(payload.get('id') == identifier, 'Backing payload ID mismatch')
        for key in ('difficulty', 'source_line', 'historical_accuracy'):
            _require(payload.get('metadata', {}).get(key) == original[key], 'Backing metadata mismatch')
        for key, value in [('prompt', payload['prompt']), ('label', payload['label']),
                           ('content', {k: payload[k] for k in ('prompt', 'label')})]:
            _require(_canonical_sha(value) == original[key + '_sha256'], 'Backing content mismatch')
    catalog = {'schema_version': 1, 'rows': catalog_rows}
    if moved:
        _require(_json(_path(root, 'catalog.json')) == catalog, 'Current catalog is incomplete or inconsistent')
    return catalog


def _checkpoint(root, recorded=None, *, files=None):
    _require(_path(root, TRACKER).read_text().strip() == str(ITERATION), 'Checkpoint tracker must select 3')
    files = _inventory(root) if files is None else files
    for name in (SELECTED + '/.metadata', SELECTED + '/common.pt'):
        _require(name in files and files[name]['size'] > 0, 'Missing checkpoint file: ' + name)
    _require(any(name.startswith(SELECTED + '/') and name.endswith('.distcp') and info['size'] > 0
                 for name, info in files.items()), 'Missing checkpoint raw shard files')
    if recorded is not None:
        _require(str(recorded.get('iteration')) == str(ITERATION) and recorded.get('selected') == SELECTED,
                 'Checkpoint evidence selects another iteration')
        references = recorded.get('referenced_files')
        _require(isinstance(references, dict) and references, 'Checkpoint reference inventory is empty')
        for name, info in references.items():
            _relative(name)
            _require(files.get(SELECTED + '/' + name) == info, 'Checkpoint referenced file mismatch: ' + name)
    return files


def _resume(root, checkpoint_dir):
    config = _json(_path(root, 'run/effective_config.json'))
    inputs = _json(_path(root, 'INPUTS.json'))
    state = _json(_path(root, checkpoint_dir + '/actor/' + CURSOR))
    for key in ('num_rollout', 'rollout_batch_size', 'n_samples_per_prompt', 'global_batch_size'):
        _require(type(config.get(key)) is int and config[key] > 0, 'Invalid training parameter: ' + key)
    _require(config['num_rollout'] == 8, 'Incident requires eight total updates')
    batch, samples = config['rollout_batch_size'], config['n_samples_per_prompt']
    _require(config['global_batch_size'] == batch * samples, 'Global batch differs from rollout parameters')
    expected = {'cursor': 4 * batch, 'group_index': 4 * batch, 'sample_index': 4 * batch * samples, 'step': 3}
    _require(all(type(state.get(key)) is int and state[key] == value for key, value in expected.items()),
             'Checkpoint datasource cursor is inconsistent with completed prefix')
    _require(state.get('arm') == inputs.get('arm') and state.get('arm') in ('healthy', 'faulty', 'repaired'),
             'Original datasource arm differs from prefix inputs')
    if 'steps' in inputs:
        _require(inputs['steps'] == 8, 'Prefix inputs require eight total updates')
    if 'relocate_at' in inputs:
        _require(inputs['relocate_at'] == 4, 'Prefix relocation must follow update 3')
    for name, key in [('manifest.json', 'manifest_sha256'), ('eval.jsonl', 'eval_sha256'), ('CASE.json', 'case_sha256')]:
        _require(inputs.get(key) == _sha(_path(root, 'dataset/' + name)), 'INPUTS dataset hash mismatch: ' + name)
    return {'checkpoint_iteration': 3, 'resume_start': 4, 'final_step': 7,
            'sample_cursor': state['cursor'], 'sample_index': state['sample_index'],
            'global_batch_size': config['global_batch_size']}, inputs, config


def _evidence(root):
    events = list(_lines(_path(root, 'run/events.jsonl')))
    steps = set(range(4))
    _require(events and all(type(row.get('step')) is int and row['step'] in steps for row in events),
             'Prefix events must cover only steps 0..3')
    for kind in ('rollout_begin', 'rollout_end'):
        _require([row['step'] for row in events if row.get('kind') == kind] == list(range(4)),
                 'Prefix events missing or duplicating completed steps: ' + kind)
    optimizer_paths = []
    directory = _path(root, 'run/optimizer_evidence')
    if directory.exists():
        optimizer_paths.extend('run/optimizer_evidence/' + name for name in _inventory(directory))
    combined = _path(root, 'run/optimizer.jsonl')
    if combined.exists():
        optimizer_paths.append('run/optimizer.jsonl')
    _require(optimizer_paths, 'Missing optimizer prefix evidence')
    observed = set()
    for name in optimizer_paths:
        for row in _lines(_path(root, name)):
            if row.get('event') != 'optimizer_step':
                continue
            role, step = row.get('role'), row.get('rollout_id')
            _require(role in ('actor', 'critic') and type(step) is int and step in steps
                     and row.get('update_successful') is True, 'Invalid optimizer prefix evidence')
            observed.add((role, step))
    _require(observed == {(role, step) for role in ('actor', 'critic') for step in steps},
             'Missing actor/critic optimizer evidence for prefix steps')
    return optimizer_paths


def _scene_name(source_name, move_subdir):
    if source_name.startswith('dataset/'):
        return 'dataset/' + _moved(source_name[len('dataset/'):], move_subdir)
    for role in ('actor', 'critic'):
        prefix = 'run/' + role + '/'
        if source_name.startswith(prefix):
            return 'checkpoints/' + role + '/' + source_name[len(prefix):]
    return source_name


def _subinventory(files, directory):
    return {name[len(directory) + 1:]: info for name, info in files.items()
            if name.startswith(directory + '/')}


def _exports(files, train_ids):
    return {'dataset': {'path': 'dataset', 'files': _subinventory(files, 'dataset'), 'train_ids': train_ids},
            'checkpoints': {role: {'path': 'checkpoints/' + role, 'iteration': ITERATION,
                                  'selected': SELECTED, 'files': _subinventory(files, 'checkpoints/' + role)}
                            for role in ('actor', 'critic')},
            'datasource_state': {'path': 'checkpoints/actor/' + CURSOR,
                                 **files['checkpoints/actor/' + CURSOR]}}


def create_scene(prefix: Path, destination: Path, move_subdir='archive/hard', *,
                 selection_path=DEFAULT_SELECTION) -> dict:
    """Copy a completed prefix and physically relocate only the copied hard tree.

    Default selection pins all 64 training/16 evaluation records. An explicit
    selection supports isolated synthetic schema fixtures. Output must be absent
    or empty. Validation finishes before copying; SCENE.json is written last.
    """
    prefix = _root(prefix)
    destination = _output(prefix, destination)
    move_subdir = _relative(move_subdir).as_posix()
    old = PurePosixPath('live/hard')
    moved = PurePosixPath(move_subdir)
    _require(not moved.is_relative_to(old) and not old.is_relative_to(moved), 'Relocation overlaps original tree')
    _require(not _path(prefix / 'dataset', move_subdir).exists(), 'Relocation destination already exists')
    selection_path = Path(selection_path)
    _require(not selection_path.is_symlink(), 'Selection must not be a symlink')
    selection_path = selection_path.resolve(strict=True)
    catalog = _dataset(_path(prefix, 'dataset'), selection_path)
    metadata, inputs, config = _resume(prefix, 'run')
    optimizer_paths = _evidence(prefix)
    recorded_path = _path(prefix, 'provenance/output_checkpoints.json')
    recorded = _json(recorded_path) if recorded_path.exists() else None
    source_files = {'dataset/' + name: info for name, info in _inventory(prefix / 'dataset').items()}
    for role in ('actor', 'critic'):
        files = _checkpoint(_path(prefix, 'run/' + role), recorded[role] if recorded else None)
        source_files.update({'run/' + role + '/' + name: info for name, info in files.items()})
    for name in (*EVIDENCE, *optimizer_paths, *(['provenance/output_checkpoints.json'] if recorded else [])):
        source_files[name] = _info(_path(prefix, name))
    source_files = dict(sorted(source_files.items()))
    destination.mkdir(parents=True, exist_ok=True)
    for name, info in source_files.items():
        # Copy the original dataset layout first; the rename is the only fault.
        target_name = _scene_name(name, 'live/hard')
        target = _path(destination, target_name)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(_path(prefix, name), target)
    target = _path(destination / 'dataset', move_subdir)
    target.parent.mkdir(parents=True, exist_ok=True)
    (destination / 'dataset/live/hard').rename(target)
    for row in catalog['rows']:
        row['path'] = _moved(row['path'], move_subdir)
    _write_json(destination / 'dataset/catalog.json', catalog)
    selection_target = destination / 'provenance/selection.json'
    selection_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(selection_path, selection_target)
    receipt = {'schema_version': 1, 'synthetic_fault_injection': True, **metadata,
               'relocation': {'from': 'live/hard', 'to': move_subdir},
               'restore_tested': False, 'checkpoint_validation': 'all_regular_files_sha256; DCP not deserialized',
               'source': {'inputs': inputs, 'parameters': config, 'files': source_files,
                          'selection_sha256': _sha(selection_target)},
               'files': _inventory(destination)}
    receipt.update(_exports(receipt['files'], [row['id'] for row in catalog['rows']]))
    # Verify the unpublished receipt first; failure never leaves a completion marker.
    _verify(destination, receipt, require_stale=True, actual=receipt['files'])
    _write_json(destination / 'SCENE.json', receipt)
    return receipt


def _verify(root, receipt, *, require_stale, actual=None):
    _require(type(receipt.get('schema_version')) is int and receipt['schema_version'] == 1
             and receipt.get('synthetic_fault_injection') is True, 'Invalid SCENE schema or synthetic marker')
    actual = _inventory(root) if actual is None else dict(actual)
    actual.pop('SCENE.json', None)
    _require(actual == receipt.get('files'), 'Scene file inventory SHA256/size/file-set mismatch')
    relocation = receipt['relocation']
    _require(relocation.get('from') == 'live/hard', 'Unexpected original hard directory')
    moved = _relative(relocation['to']).as_posix()
    source = receipt['source']
    for name, info in source['files'].items():
        _relative(name)
        _require(actual.get(_scene_name(name, moved)) == info, 'Original source file changed: ' + name)
    selection_path = _path(root, 'provenance/selection.json')
    _require(_sha(selection_path) == source['selection_sha256'], 'Scene selection SHA256 mismatch')
    catalog = _dataset(_path(root, 'dataset'), selection_path, moved=moved, require_stale=require_stale)
    for key, value in _exports(actual, [row['id'] for row in catalog['rows']]).items():
        _require(receipt.get(key) == value, 'Scene exported inventory mismatch: ' + key)
    metadata, inputs, config = _resume(root, 'checkpoints')
    _require(inputs == source['inputs'] and config == source['parameters'], 'Original source parameters changed')
    _require(all(receipt.get(key) == value for key, value in metadata.items()), 'Scene resume parameters mismatch')
    recorded_path = _path(root, 'provenance/output_checkpoints.json')
    recorded = _json(recorded_path) if recorded_path.exists() else None
    for role in ('actor', 'critic'):
        _checkpoint(_path(root, 'checkpoints/' + role), recorded[role] if recorded else None,
                    files=_subinventory(actual, 'checkpoints/' + role))
    _evidence(root)
    return receipt


def verify_scene(root: Path, *, require_stale=True) -> dict:
    """Verify all bytes, selected identities, stale reads, and resume continuity.

    require_stale=False skips only the missing-old-hard-path assertion. It does
    not permit changing any inventoried bytes, including the original manifest.
    """
    root = _root(root)
    return _verify(root, _json(_path(root, 'SCENE.json')), require_stale=require_stale)


def copy_scene(source: Path, destination: Path) -> dict:
    """Copy an entire verified scene into an absent/empty directory and reverify."""
    source = _root(source)
    destination = _output(source, destination)
    receipt = verify_scene(source)
    receipt_info = _info(source / 'SCENE.json')
    destination.mkdir(parents=True, exist_ok=True)
    for name in receipt['files']:
        target = _path(destination, name)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(_path(source, name), target)
    _verify(destination, receipt, require_stale=True)
    shutil.copyfile(_path(source, 'SCENE.json'), destination / 'SCENE.json')
    if _info(destination / 'SCENE.json') != receipt_info:
        (destination / 'SCENE.json').unlink()
        raise ValueError('Source scene receipt changed during copy')
    return receipt


def materialize_scene(scene: Path, workspace: Path) -> dict:
    """Populate workspace/incident; keep other workspace files and refuse reuse.

    Bytes are independent copies. Filesystem permissions are not an immutability
    guarantee; the frozen runner must compare checkpoint hashes to its trusted
    prepared receipt before and after execution.
    """
    workspace = Path(workspace)
    _require(not workspace.is_symlink(), 'Workspace must not be a symlink')
    return copy_scene(scene, workspace / 'incident')


class _HashReader:
    def __init__(self, stream):
        self.stream = stream
        self.digest = hashlib.sha256()

    def read(self, size):
        block = self.stream.read(size)
        self.digest.update(block)
        return block


def pack_scene(root: Path, archive_path: Path) -> dict:
    """Write a complete tar, optionally gzip/bzip2/xz compressed; never split it.

    Members are sorted regular files relative to the scene root, including
    SCENE.json. No absolute paths, directory owners, timestamps, or links enter
    tar members. Existing archives and archives inside the scene are rejected.
    """
    root = _root(root)
    archive_path = Path(archive_path)
    _require(not archive_path.is_symlink(), 'Archive must not be a symlink')
    archive_path = archive_path.resolve()
    _require(not archive_path.is_relative_to(root), 'Archive must be outside the scene')
    if archive_path.exists():
        raise FileExistsError('Archive already exists: ' + str(archive_path))
    modes = {'.tar': 'w|', '.tar.gz': 'w|gz', '.tgz': 'w|gz', '.tar.bz2': 'w|bz2', '.tar.xz': 'w|xz'}
    mode = next((mode for suffix, mode in modes.items() if archive_path.name.endswith(suffix)), None)
    _require(mode is not None, 'Archive suffix must be .tar, .tar.gz, .tgz, .tar.bz2, or .tar.xz')
    receipt = verify_scene(root)
    files = dict(receipt['files'], **{'SCENE.json': _info(root / 'SCENE.json')})
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with archive_path.open('xb') as output:
        try:
            with tarfile.open(fileobj=output, mode=mode, format=tarfile.PAX_FORMAT) as archive:
                for name, info in sorted(files.items()):
                    member = tarfile.TarInfo(name)
                    member.size, member.mode = info['size'], 0o644
                    with _path(root, name).open('rb') as stream:
                        reader = _HashReader(stream)
                        archive.addfile(member, reader)
                        _require(reader.digest.hexdigest() == info['sha256'] and not stream.read(1),
                                 'Scene changed while archiving: ' + name)
        except BaseException:
            archive_path.unlink()
            raise
    return {**_info(archive_path), 'file_count': len(files), 'scene_sha256': files['SCENE.json']['sha256']}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    create = commands.add_parser('create', help='Copy prefix checkpoint 3 and relocate its dataset')
    create.add_argument('--prefix', type=Path, required=True)
    create.add_argument('--destination', '--out', type=Path, required=True)
    create.add_argument('--move-subdir', default='archive/hard')
    create.add_argument('--selection', type=Path, default=DEFAULT_SELECTION)
    verify = commands.add_parser('verify', help='Verify inventory and resume metadata')
    verify.add_argument('root', type=Path)
    verify.add_argument('--no-require-stale', action='store_true')
    copy = commands.add_parser('copy', help='Copy a verified scene')
    copy.add_argument('source', type=Path)
    copy.add_argument('destination', type=Path)
    pack = commands.add_parser('pack', help='Pack a verified scene as a tar archive')
    pack.add_argument('root', type=Path)
    pack.add_argument('archive_path', type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == 'create':
            result = create_scene(args.prefix, args.destination, args.move_subdir, selection_path=args.selection)
        elif args.command == 'verify':
            result = verify_scene(args.root, require_stale=not args.no_require_stale)
        elif args.command == 'copy':
            result = copy_scene(args.source, args.destination)
        else:
            result = pack_scene(args.root, args.archive_path)
        # Do not print original configuration, environment, or filesystem paths.
        summary = ({key: result[key] for key in ('schema_version', 'checkpoint_iteration', 'resume_start',
                                               'final_step', 'sample_cursor', 'sample_index', 'restore_tested')}
                   if args.command != 'pack' else result)
        if args.command != 'pack':
            summary['file_count'] = len(result['files'])
        print(json.dumps(summary, sort_keys=True))
        return 0
    except (OSError, ValueError, KeyError, TypeError) as error:
        print('Scene operation failed: ' + str(error), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
