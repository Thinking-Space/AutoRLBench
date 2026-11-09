#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Export a fresh runtime-debugging variant, without historical failure logs."""
import argparse
import ast
import difflib
import json
from pathlib import Path
import shutil
import sys
import tempfile

HERE = Path(__file__).resolve().parent
import source_bundle as base
from source_cleanup import clean_source, remove_diagnostic_tools

IMPORT = 'from miles.utils.collective_plan import reduce_sum\n'
PLAN = 'miles/utils/collective_plan.py'


def route_reductions(text, expected):
    # Source pins are checked first. Restrict the transform to SUM all_reduce
    # calls in the two native functions, not other distributed operations.
    tree = ast.parse(text)
    calls = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == 'dist' and node.func.attr == 'all_reduce'):
            calls.append(node.func)
    if len(calls) != expected:
        raise ValueError('Pinned SUM call count differs')
    lines = text.splitlines(True)
    for node in sorted(calls, key=lambda n: (n.lineno, n.col_offset), reverse=True):
        if node.lineno != node.end_lineno:
            raise ValueError('Unexpected multiline callable')
        line = lines[node.lineno - 1]
        lines[node.lineno - 1] = line[:node.col_offset] + 'reduce_sum' + line[node.end_col_offset:]
    return IMPORT + ''.join(lines)


def route_function(text, function_name, expected):
    function = next(n for n in ast.parse(text).body
                    if isinstance(n, ast.FunctionDef) and n.name == function_name)
    lines = text.splitlines(True)
    block = ''.join(lines[function.lineno - 1:function.end_lineno])
    routed = route_reductions(block, expected).removeprefix(IMPORT)
    lines[function.lineno - 1:function.end_lineno] = [routed]
    return IMPORT + ''.join(lines)


def repair_plan(text):
    return base.replace_once(text, 'key = _signature(tensor, group)',
                             'key = (group, *_signature(tensor, group))')


def export(source, output):
    output = Path(output).absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    files = base.read_source(Path(source).resolve())
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.runtime-', dir=output.parent) as tmp:
        staging = Path(tmp) / 'export'
        workspace, maintainer = staging / 'workspace', staging / 'maintainer'
        workspace.mkdir(parents=True)
        maintainer.mkdir()
        for name, data in files.items():
            if not (name.startswith(('miles/', 'miles_plugins/', 'justrl2/model_args/'))
                    or name in ('train.py', 'LICENSE', 'setup.py', 'pyproject.toml', 'requirements.txt')):
                continue
            path = workspace / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        for folder in ('miles', 'miles_plugins'):
            for path in (workspace / folder).rglob('*.py'):
                path.write_text(clean_source(path.read_text()))
        remove_diagnostic_tools(workspace)
        for relative, function, count in (
            ('miles/utils/ppo_utils.py', 'get_advantages_and_returns_batch', 2),
            ('miles/utils/distributed_utils.py', 'distributed_masked_whiten', 1),
        ):
            path = workspace / relative
            path.write_text(route_function(path.read_text(), function, count))
        shutil.copyfile(HERE / 'collective_plan.py', workspace / PLAN)
        debug = workspace / 'debug'
        debug.mkdir()
        shutil.copyfile(HERE / 'run_replay.py', debug / 'run_replay.py')
        shutil.copyfile(HERE / 'instruction.md', workspace / 'instruction.md')
        for path in workspace.rglob('*.py'):
            ast.parse(path.read_text(), filename=str(path.relative_to(workspace)))
        defective = (workspace / PLAN).read_text()
        repaired = repair_plan(defective)
        patch = ''.join(difflib.unified_diff(defective.splitlines(True), repaired.splitlines(True),
                                            fromfile='a/' + PLAN, tofile='b/' + PLAN))
        (maintainer / 'repair.patch').write_text(patch)
        shutil.copyfile(HERE / 'oracle.md', maintainer / 'oracle.md')
        raw = {name: base.sha(data) for name, data in files.items()}
        final = base.inventory(workspace)
        manifest = {
            'schema_version': 1, 'case_kind': 'constructed_reduction_plan_cache_regression',
            'source_revision': base.REVISION, 'source_files': raw, 'workspace_files': final,
            'maintainer_files': base.inventory(maintainer),
            'changed_source_files': {name: {'before': raw[name], 'after': digest}
                                     for name, digest in final.items() if name in raw and raw[name] != digest},
            'added_source_files': sorted(set(final) - set(raw)),
            'old_local_guard_present': False,
            'provided_runtime_diagnostics': False,
            'instrumentation_policy': 'No prebuilt trace switch, tensor hooks, route records or intermediate snapshots; investigation method is unrestricted.', 'historical_gpu_evidence_applies': False,
            'gpu_training_validated': False, 'difficulty_calibrated': False,
            'scope': 'Native PPO/CP functions plus constructed reduction-plan transport, real CPU/Gloo; synthetic critic inputs',
            'boundary': 'Only workspace/ is contestant-visible; no parent directory, git, network or canonical source mount.',
        }
        base.archive_workspace(workspace, staging / 'workspace.tar.gz')
        manifest['workspace_archive_sha256'] = base.sha((staging / 'workspace.tar.gz').read_bytes())
        base.write_json(maintainer / 'manifest.json', manifest)
        staging.rename(output)
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    receipt = export(args.source, args.output)
    print(json.dumps({k: receipt[k] for k in ('case_kind', 'workspace_archive_sha256',
                                            'historical_gpu_evidence_applies', 'gpu_training_validated')}, indent=2))
