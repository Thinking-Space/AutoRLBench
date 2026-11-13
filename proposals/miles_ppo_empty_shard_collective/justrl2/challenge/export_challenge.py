#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Export a defective-only contestant mount and separate maintainer artifacts."""
import argparse
import ast
from collections import defaultdict
import difflib
import gzip
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile
from source_cleanup import clean_source, remove_diagnostic_tools
from window_reference import exported_expectations

HERE = Path(__file__).resolve().parent
EVIDENCE = HERE.parent
REVISION = '75814318f243af7e39af337c1173fc3927c9635b'
TARGET = 'miles/backends/training_utils/loss.py'
REPLAY_SHA256 = 'ecb7d044a6c515ab897a169e2bd352e5a0397d7b26f38beb1c6870dec4d8060c'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def inventory(root):
    return {str(p.relative_to(root)): sha(p.read_bytes())
            for p in sorted(root.rglob('*')) if p.is_file()}


def replace_once(text, before, after):
    if text.count(before) != 1:
        raise ValueError('Source anchor missing or ambiguous')
    return text.replace(before, after, 1)


def strip_diagnostics(text):
    """Remove all upstream adv-diag log calls, preserving every computation."""
    nodes = []
    for n in ast.walk(ast.parse(text)):
        if (isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
                and isinstance(n.value.func, ast.Attribute)
                and isinstance(n.value.func.value, ast.Name)
                and n.value.func.value.id == '_diag_logger'):
            nodes.append(n)
    lines = text.splitlines(keepends=True)
    for n in sorted(nodes, key=lambda n: n.lineno, reverse=True):
        lines[n.lineno - 1:n.end_lineno] = [' ' * n.col_offset + 'pass\n']
    result = ''.join(lines)
    ast.parse(result)
    return result


MOMENT_REDUCER = '# SPDX-License-Identifier: Apache-2.0\nfrom functools import reduce\nfrom operator import add\n\nimport torch\n\n\nclass MomentReducer:\n    def __init__(self, width=4096):\n        self.width = width\n        self._origins = {}\n\n    def __call__(self, values, mask):\n        key = (values.device, torch.float32)\n        origin = self._origins.get(key)\n        if origin is None:\n            origin = torch.zeros(3, device=values.device, dtype=torch.float32)\n            self._origins[key] = origin\n        blocks = (\n            torch.stack(((x * m).sum(), (x.square() * m).sum(), m.sum())).float()\n            for start in range(0, values.numel(), self.width)\n            for x, m in [(values[start:start + self.width], mask[start:start + self.width])]\n        )\n        return reduce(add, blocks, origin)\n'
MOMENT_IMPORTS = 'from miles.utils.moments import MomentReducer\n\n_moments = MomentReducer()\n\n'
MOMENT_CALL = '    stats_tensor = _moments(values, mask)\n'


def inject_cached_moment_reducer(text):
    tree = ast.parse(text)
    fn = next((n for n in tree.body if isinstance(n, ast.FunctionDef)
               and n.name == 'distributed_masked_whiten'), None)
    if fn is None:
        raise ValueError('Whitening function missing')
    nodes = [n for n in fn.body if isinstance(n, ast.Assign)
             and any(isinstance(t, ast.Name) and t.id in
                     {'local_sum', 'local_sum_sq', 'local_mask_sum', 'stats_tensor'} for t in n.targets)]
    if len(nodes) != 4:
        raise ValueError('Whitening moment construction differs from pinned source')
    lines = text.splitlines(keepends=True)
    lines[nodes[0].lineno - 1:nodes[-1].end_lineno] = [MOMENT_CALL]
    return MOMENT_IMPORTS + ''.join(lines)


def repair_cached_moment_reducer(text):
    if text.count(MOMENT_CALL) != 1:
        raise ValueError('Cached moment reducer call missing or ambiguous')
    return text.replace(MOMENT_CALL, '    stats_tensor = _moments(values, mask).clone()\n', 1)


def repair_moment_origin(text):
    if text.count('        return reduce(add, blocks, origin)\n') != 1:
        raise ValueError('Cached moment reduction missing or ambiguous')
    return text.replace('        return reduce(add, blocks, origin)\n',
                        '        if values.numel() == 0:\n            return origin.clone()\n        return reduce(add, blocks, origin)\n', 1)


def project_evidence(rows, trace, fixture):
    """Allow-list an observation view; never relabel or invent numerical data.

    Checkpoints 0/1 are projections of existing instrumentation, not new runs.
    Their callsites and the omitted fields are recorded only in the manifest.
    """
    ledgers = defaultdict(list)
    for row in rows:
        ledgers[(row['role'], row['rank'])].append(row)
    expected = {(role, rank) for role in ('actor', 'critic') for rank in range(4)}
    if set(ledgers) != expected or len(fixture['samples']) != 2:
        raise ValueError('Incomplete rank ledger or replay')
    workers, roles = [], {}
    for role, rank in sorted(expected):
        ledger = ledgers[role, rank]
        entries = [r for r in ledger if r['event'] == 'whitening_enter']
        ready = [r for r in ledger if r['event'] == 'advantages_ready']
        if len(entries) != 1 or len(ready) > 1:
            raise ValueError('Ambiguous checkpoint identity')
        entry = entries[0]
        cp_members, dp_members = [rank // 2 * 2, rank // 2 * 2 + 1], [rank % 2, rank % 2 + 2]
        for r in ledger:
            if (r['cp_members'] != cp_members or r['dp_members'] != dp_members
                    or r['cp_rank'] != rank % 2 or r['dp_rank'] != rank // 2
                    or r['cp_size'] != 2 or r['dp_size'] != 2):
                raise ValueError('Inconsistent process group ledger')
        sample = fixture['samples'][rank // 2]
        if (entry['response_lengths'] != [sample['response_length']]
                or entry['total_lengths'] != [len(sample['tokens'])]):
            raise ValueError('Replay and runtime geometry disagree')
        if ready and (ledger.index(ready[0]) <= ledger.index(entry)
                      or ready[0]['monotonic'] <= entry['monotonic']):
            raise ValueError('Out-of-order rank checkpoints')
        workers.append({'role': role, 'rank': rank, 'sample_indices': [rank // 2],
                        'checkpoints': [0, 1] if ready else [0]})
        roles[role] = {'cp_groups': [[0, 1], [2, 3]], 'dp_groups': [[0, 2], [1, 3]],
                       'rank_scope': 'role-local', 'world_size': 4}
    failures = [line for line in trace.splitlines()
                if line.startswith('torch.distributed.DistBackendError:')]
    if len(failures) != 1 or 'store->get' not in failures[0] or 'wait timeout after 60000ms' not in failures[0]:
        raise ValueError('Expected actual backend failure, not an outer deadline')
    # The replay is the exact original authored input, with all token IDs intact.
    return {'mesh.json': {'roles': roles}, 'replay.json': fixture,
            'progress.json': {'schema_version': 1, 'workers': workers,
                              'ordering': 'checkpoints are ordered within each worker only'},
            'failure.txt': '# Backend error excerpt; Python caller frames omitted.\n' + failures[0] + '\n'}


def read_source(source):
    if (source / '.git').exists():
        actual = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
        if actual != REVISION:
            raise ValueError('Source HEAD differs from pinned revision')
        data = subprocess.check_output(['git', '-C', str(source), 'archive', REVISION])
        with tarfile.open(fileobj=io.BytesIO(data)) as archive:
            files = {}
            for member in archive.getmembers():
                if member.isdir():
                    continue
                if not member.isfile() or member.name.startswith('/') or '..' in Path(member.name).parts:
                    raise ValueError('Source contains a link or invalid archive path')
                files[member.name] = archive.extractfile(member).read()
    else:
        if any(p.is_symlink() for p in source.rglob('*')):
            raise ValueError('Source tree must not contain symlinks')
        files = {str(p.relative_to(source)): p.read_bytes() for p in source.rglob('*') if p.is_file()}
    pins = json.loads((EVIDENCE / 'source_files.json').read_text())['files']
    if {name: sha(data) for name, data in files.items()} != pins:
        raise ValueError('Source file inventory differs from pinned public revision')
    return files


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + '\n')


def build_runtime_support():
    """Extract native-source loading helpers for the exported update runner."""
    tree = ast.parse((HERE / 'run_replay.py').read_text())
    constants = {'MAX_RESPONSE_LENGTH', 'PLAN', 'NATIVE_FUNCTIONS'}
    functions = {'_bound_names', '_source_functions', '_load_native'}
    nodes = [n for n in tree.body
             if isinstance(n, (ast.Import, ast.ImportFrom))
             or isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in constants for t in n.targets)
             or isinstance(n, ast.FunctionDef) and n.name in functions]
    module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), *nodes],
                        type_ignores=[])
    result = ast.unparse(ast.fix_missing_locations(module)) + '\n'
    ast.parse(result)
    return result


def archive_workspace(workspace, destination):
    with destination.open('wb') as raw, gzip.GzipFile(fileobj=raw, mode='wb', filename='', mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode='w') as archive:
            for p in sorted(workspace.rglob('*')):
                if not p.is_file():
                    continue
                data = p.read_bytes()
                info = tarfile.TarInfo(str(p.relative_to(workspace)))
                info.size, info.mode, info.mtime = len(data), 0o644, 0
                archive.addfile(info, io.BytesIO(data))


def read_evidence():
    published = json.loads((EVIDENCE / 'public_runtime_results.json').read_text())['published_evidence_sha256']
    for name in ('public_runtime_events.json', 'public_faulty_trace.txt'):
        if sha((EVIDENCE / name).read_bytes()) != published[name]:
            raise ValueError('Frozen evidence differs from the published receipt: ' + name)
    if sha((EVIDENCE / 'fixture.json').read_bytes()) != REPLAY_SHA256:
        raise ValueError('Replay differs from the recorded canonical JSON')
    return project_evidence(json.loads((EVIDENCE / 'public_runtime_events.json').read_text())['faulty'],
                            (EVIDENCE / 'public_faulty_trace.txt').read_text(),
                            json.loads((EVIDENCE / 'fixture.json').read_text()))


def export(source, output):
    source, output = source.resolve(), output.absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    files = read_source(source)
    evidence = read_evidence()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.challenge-', dir=output.parent) as tmp:
        stage = Path(tmp) / 'bundle'
        workspace, private = stage / 'workspace', stage / 'maintainer'
        workspace.mkdir(parents=True)
        private.mkdir()
        keep = lambda name: (name.startswith(('miles/', 'miles_plugins/', 'justrl2/model_args/'))
                             or name in ('train.py', 'LICENSE', 'setup.py', 'pyproject.toml', 'requirements.txt'))
        for name, data in files.items():
            if keep(name):
                target = workspace / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
        loss = (workspace / TARGET).read_text()
        begin = loss.index('        # Collective-communication invariant: distributed_masked_whiten')
        end = loss.index('        assert (', begin)
        control = strip_diagnostics(loss[:begin] + loss[end:])
        if 'if all_masks.numel() > 0:' in control:
            raise ValueError('Pinned caller unexpectedly contains the old local guard')
        (workspace / TARGET).write_text(control)
        whitening_path = workspace / 'miles/utils/distributed_utils.py'
        whitening_control = whitening_path.read_text()
        whitening_defective = inject_cached_moment_reducer(whitening_control)
        whitening_path.write_text(whitening_defective)
        (workspace / 'miles/utils/moments.py').write_text(MOMENT_REDUCER)
        # Retain the runtime compatibility change used in the recorded GPU run.
        actor = workspace / 'miles/backends/megatron_utils/actor.py'
        actor.write_text('from benchmark.process_group_compat import install as install_process_group_compat\n' +
                         replace_once(actor.read_text(), '        monkey_patch_torch_dist()\n',
                                      '        monkey_patch_torch_dist()\n        install_process_group_compat()\n'))
        helper = workspace / 'benchmark'
        helper.mkdir()
        (helper / '__init__.py').write_text('')
        for name in ('run_training.py', 'train.sh', 'replay_loader.py', 'sglang_compat.py', 'process_group_compat.py'):
            shutil.copyfile(EVIDENCE / name, helper / name)
        # Export the first-update launcher only; native restore instrumentation
        # belongs to the historical audit and is not installed in this mount.
        launch = (helper / 'train.sh').read_text().replace(
            '# Run inside the prepared JustRL2 workspace. Values are shared by both arms.',
            '# Run inside this JustRL2 workspace with externally staged assets.')
        begin = launch.index('if [[ -n "${BENCH_RESTORE_ROOT:-}" ]]; then\n')
        end = launch.index('\nfi\n', begin) + len('\nfi\n')
        (helper / 'train.sh').write_text(launch[:begin] + launch[end:])
        runner = (helper / 'run_training.py').read_text()
        begin = runner.index('        if os.environ.get("BENCH_RESTORE_ROOT"):\n')
        end = runner.index('        (output / "effective_config.json")', begin)
        (helper / 'run_training.py').write_text((runner[:begin] + runner[end:]).replace(
            '"BENCH_RESTORE_ROOT", "BENCH_RESTORE_ITERATION", ', ''))
        # Preserve the historical full-GPU launcher only for reviewers. Its
        # built-in NCCL dumps and runtime metadata are not contestant helpers.
        gpu_launch = private / 'gpu_launch'
        gpu_launch.mkdir()
        for name in ('run_training.py', 'train.sh', 'replay_loader.py', 'sglang_compat.py'):
            (helper / name).rename(gpu_launch / name)
        observations = workspace / 'observed'
        observations.mkdir()
        for name, data in evidence.items():
            if name != 'replay.json':
                continue
            if isinstance(data, str):
                (observations / name).write_text(data)
            else:
                write_json(observations / name, data)
        shutil.copyfile(HERE / 'instruction.md', workspace / 'instruction.md')
        # Only the actual inputs are contestant-visible. Recorded topology,
        # per-rank progress and failure stacks stay in the reviewer boundary.
        observations.rename(workspace / 'inputs')
        for name, data in evidence.items():
            if name == 'replay.json':
                continue
            if isinstance(data, str):
                (private / name).write_text(data)
            else:
                write_json(private / name, data)
        debug = workspace / 'debug'
        debug.mkdir()
        shutil.copyfile(HERE / 'run_window.py', debug / 'run_window.py')
        write_json(workspace / 'inputs/expected.json', exported_expectations(evidence['replay.json']))
        (debug / 'runtime_support.py').write_text(build_runtime_support())
        evaluator = private / 'window_evaluator'
        evaluator.mkdir()
        shutil.copyfile(HERE / 'run_window.py', evaluator / 'run_window.py')
        (evaluator / 'runtime_support.py').write_text(build_runtime_support())
        (evaluator / 'inputs').mkdir()
        shutil.copyfile(EVIDENCE / 'fixture.json', evaluator / 'inputs/replay.json')
        write_json(evaluator / 'inputs/expected.json', exported_expectations(evidence['replay.json']))
        shutil.copyfile(HERE / 'window_probe.py', private / 'window_probe.py')
        shutil.copyfile(HERE / 'window_reference.py', private / 'window_reference.py')
        shutil.copyfile(HERE / 'sequence_audit.py', private / 'sequence_audit.py')
        for folder in ('miles', 'miles_plugins'):
            for path in (workspace / folder).rglob('*.py'):
                path.write_text(clean_source(path.read_text()))
        remove_diagnostic_tools(workspace)
        for path in workspace.rglob('*.py'):
            ast.parse(path.read_text(), filename=str(path.relative_to(workspace)))
        repair_parts = []
        helper_defective = (workspace / 'miles/utils/moments.py').read_text()
        helper_control = repair_moment_origin(helper_defective)
        for relative, defective_text, control_text in (
                (TARGET, (workspace / TARGET).read_text(), control),
                ('miles/utils/distributed_utils.py', whitening_defective,
                 repair_cached_moment_reducer(whitening_defective)),
                ('miles/utils/moments.py', helper_defective, helper_control)):
            defective_text, control_text = clean_source(defective_text), clean_source(control_text)
            if defective_text != control_text:
                repair_parts.extend(difflib.unified_diff(
                    defective_text.splitlines(True), control_text.splitlines(True),
                    fromfile='a/' + relative, tofile='b/' + relative))
        repair = ''.join(repair_parts)
        (private / 'repair.patch').write_text(repair)
        shutil.copyfile(HERE / 'oracle.md', private / 'oracle.md')
        shutil.copyfile(HERE / 'topology_verifier.py', private / 'topology_verifier.py')
        staged = inventory(workspace)
        manifest = {'schema_version': 1, 'source_revision': REVISION,
                    'source_files': {name: sha(data) for name, data in files.items()},
                    'workspace_files': staged, 'omitted_source_files': sorted(set(files) - set(staged)),
                    'maintainer_files': inventory(private),
                    'changed_source_files': {name: {'before': sha(files[name]), 'after': staged[name]}
                                             for name in files if name in staged and sha(files[name]) != staged[name]},
                    'frozen_evidence': {name: sha((EVIDENCE / name).read_bytes()) for name in
                                        ('fixture.json', 'public_runtime_events.json', 'public_faulty_trace.txt')},
                    'projection': {'source_arm': 'faulty',
                                   'reviewer_only': ['CP/DP membership ledger', 'all rank progress', 'backend exception', 'full original artifacts'],
                                   'contestant_inputs': ['replay token IDs, lengths, rewards and masks',
                                                         'independent aggregate loss references',
                                                         'authored synthetic critic and log-ratio formulas in runner'],
                                   'provided_runtime_diagnostics': False},
                    'code_transformations': ['allow-list native framework modules and model args',
                                             'remove answer-revealing repair comment', 'replace all adv-diag logger calls with pass',
                                             'add a reusable window-statistics reducer with an in-place collective boundary', 'retain common process-group compatibility helper',
                                             'keep historical GPU launcher and its diagnostic setup reviewer-only'],
                    'observation_policy': 'No supplied log/print calls or non-license comments in framework copies; no prebuilt trace; investigation method is unrestricted.',
                    'gpu_revalidated': False, 'difficulty_calibrated': False,
                    'boundary': 'mount ONLY workspace/; keep repository and maintainer/ inaccessible; network disabled'}
        write_json(private / 'manifest.json', manifest)
        archive_workspace(workspace, stage / 'workspace.tar.gz')
        manifest['workspace_archive_sha256'] = sha((stage / 'workspace.tar.gz').read_bytes())
        write_json(private / 'manifest.json', manifest)
        stage.rename(output)
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True, help='pinned source checkout or exact extracted archive')
    parser.add_argument('--output', type=Path, required=True, help='fresh destination outside contestant access')
    args = parser.parse_args()
    receipt = export(args.source, args.output)
    print(json.dumps({key: receipt[key] for key in ('source_revision', 'workspace_archive_sha256',
                                                  'gpu_revalidated', 'difficulty_calibrated')}, indent=2))
