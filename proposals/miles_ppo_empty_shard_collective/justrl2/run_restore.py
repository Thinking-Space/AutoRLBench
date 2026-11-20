#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Restore both original healthy checkpoints, validate all state, run one update.

Run inside the existing dedicated 8-GPU image; no image build is required::

    BENCH_DEDICATED_CONTAINER=1 python3 run_restore.py \
      --prepared ATTEMPT/prepared --native-model ATTEMPT/native \
      --checkpoint-root ATTEMPT/pair-run/healthy --output NEW_DIRECTORY

Original prepared inputs and checkpoints are read only. A copy of the healthy
workspace is instrumented under NEW_DIRECTORY/workspace; only NEW_DIRECTORY/run
receives training checkpoints. RESTORE_RESULT.json is written only on success.
"""
import argparse
import json
import os
from pathlib import Path
import shutil

from checkpoint_evidence import file_sha, inspect_checkpoint, verify_conversion
from restore_evidence import _disjoint, validate_progress
from run_pair import run_arm

HERE = Path(__file__).resolve().parent


def instrument_model(text):
    if 'from benchmark.restore_evidence import restore_load_checkpoint, restore_after_save' in text:
        return text
    from build_case import replace_once
    text = 'from benchmark.restore_evidence import restore_load_checkpoint, restore_after_save\n' + text
    text = replace_once(text, '    iteration, _ = load_checkpoint(\n',
                        '    iteration, _ = restore_load_checkpoint(args, load_checkpoint,\n')
    text = replace_once(text, '    if hashes is not None:\n',
                        '    restore_after_save(args, iteration)\n\n    if hashes is not None:\n')
    return text


def copy_workspace(prepared, destination):
    source = Path(prepared) / 'pair/healthy'
    shutil.copytree(source, destination, symlinks=True,
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    for name in ('Megatron-LM', 'sglang'):
        target = destination / name
        if target.is_symlink():
            resolved = (source / name).resolve(strict=True)
            target.unlink()
            target.symlink_to(resolved, target_is_directory=True)
    for name in ('restore_evidence.py', 'run_training.py', 'train.sh', 'runtime_evidence.py'):
        target = destination / 'benchmark' / name
        if target.is_symlink():
            raise ValueError('Refuse helper symlink: ' + str(target))
        shutil.copyfile(HERE / name, target)
    model = destination / 'miles/backends/megatron_utils/model.py'
    if model.is_symlink():
        raise ValueError('Refuse source symlink')
    model.write_text(instrument_model(model.read_text()))
    return destination


def verify_events(rows, original, output, iteration):
    expected_workers = {(role, rank) for role in ('actor', 'critic') for rank in range(4)}
    relevant = [r for r in rows if r['event'] in ('restore_loaded','optimizer_step','restore_saved')]
    if {(r['role'], r['rank']) for r in relevant} != expected_workers:
        raise ValueError('Missing or unexpected restore worker')
    results = {}
    for role, rank in sorted(expected_workers):
        events = [r for r in relevant if (r['role'],r['rank']) == (role,rank)]
        if [e['event'] for e in events] != ['restore_loaded','optimizer_step','restore_saved']:
            raise ValueError('Expected one load, one update, one save, in order on every rank')
        loaded, step, saved = events
        if len({e['pid'] for e in events}) != 1:
            raise ValueError('Worker restarted during restore validation')
        for e in events:
            if (e['cp_size'],e['dp_size'],e['cp_rank'],e['dp_rank'],e['cp_members'],e['dp_members']) != (
                    2,2,rank%2,rank//2,[rank//2*2,rank//2*2+1],[rank%2,rank%2+2]):
                raise ValueError('Wrong CP2 DP2 topology')
        if loaded['payload_exact'] is not True or saved['payload_exact'] is not True:
            raise ValueError('Missing exact checkpoint payload comparison')
        if (loaded['loaded_iteration'],loaded['next_rollout_id'],saved['saved_iteration']) != (
                iteration,iteration+1,iteration+1):
            raise ValueError('Wrong restore/save iteration')
        if Path(loaded['load_root']).resolve() != (original/role).resolve() or Path(saved['save_root']).resolve() != (output/role).resolve():
            raise ValueError('Unexpected load/save path')
        if (step['rollout_id'],step['step_id']) != (iteration+1,0) or any(
                step[k] is not True for k in ('update_successful','grad_norm_finite','sampled_finite')):
            raise ValueError('Missing successful finite next optimizer update')
        before, after = loaded['snapshot'], saved['snapshot']
        for fp in (before, after):
            if not fp['optimizer_steps'] or min(fp['optimizer_steps']) < 1 or not all(fp['moment_elements'].values()):
                raise ValueError('Missing stepped Adam state')
        validate_progress(before, after)
        results[f'{role}:{rank}'] = {'loaded':before, 'saved':after}
    for role in ('actor', 'critic'):
        if not any(r.get('sampled_changed') is True for r in relevant if r['role']==role and r['event']=='optimizer_step'):
            raise ValueError('No demonstrated FP32 update for ' + role)
    return {'ranks_verified':8, 'snapshots':results}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--native-model', type=Path, required=True)
    parser.add_argument('--checkpoint-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--iteration', type=int, default=0)
    parser.add_argument('--timeout', type=int, default=2400)
    args = parser.parse_args()
    if os.environ.get('BENCH_DEDICATED_CONTAINER') != '1':
        raise ValueError('Requires a dedicated 8-GPU container: BENCH_DEDICATED_CONTAINER=1')
    prepared, native, original, output = (p.resolve() for p in (
        args.prepared,args.native_model,args.checkpoint_root,args.output))
    if args.iteration != 0:
        raise ValueError('This validator targets the preserved iteration-0 pair')
    if not all(_disjoint(output,p) for p in (prepared,native,original)):
        raise ValueError('Fresh output must be disjoint from all original inputs')
    output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((prepared/'pair/pair_manifest.json').read_text())
    for name, sha in manifest['healthy_files'].items():
        if file_sha(prepared/'pair/healthy'/name) != sha:
            raise ValueError('Prepared healthy source changed: '+name)
    verify_conversion(prepared, native)
    replay = prepared/'replay/replay.pt'
    replay_manifest = json.loads((prepared/'replay/manifest.json').read_text())
    if file_sha(replay) != replay_manifest['files']['replay.pt']['sha256']:
        raise ValueError('Replay differs from manifest')
    before = {role:inspect_checkpoint(original/role, expected_iteration=args.iteration, require_optimizer=True)
              for role in ('actor','critic')}
    (output/'INPUT_CHECKPOINTS.json').write_text(json.dumps(before,indent=2)+'\n')
    workspace = copy_workspace(prepared, output/'workspace')
    env = {'BENCH_MODEL':str(prepared/'model'), 'BENCH_NATIVE_MODEL':str(native),
           'BENCH_REPLAY':str(replay), 'BENCH_RESTORE_ROOT':str(original),
           'BENCH_RESTORE_ITERATION':str(args.iteration)}
    record = run_arm(workspace, output/'run', env, args.timeout)
    (output/'ATTEMPT.json').write_text(json.dumps(record,indent=2)+'\n')
    # Rehash originals even if training failed; never silently accept source mutation.
    after = {role:inspect_checkpoint(original/role, expected_iteration=args.iteration, require_optimizer=True)
             for role in ('actor','critic')}
    if before != after:
        raise ValueError('Original checkpoint contents changed')
    if record['returncode'] != 0 or record['outer_deadline_reached']:
        raise ValueError('Restore training failed; see run/stdout.log and ATTEMPT.json')
    returned = json.loads((output/'run/TRAINING_RETURNED.json').read_text())
    if returned.get('train_coroutine_returned') is not True:
        raise ValueError('Native training did not return normally')
    rows = [json.loads(line) for path in sorted((output/'run/evidence').glob('*.jsonl'))
            for line in path.read_text().splitlines() if line.strip()]
    result = verify_events(rows, original, output/'run', args.iteration)
    result['checkpoints'] = {role:inspect_checkpoint(output/'run'/role, expected_iteration=args.iteration+1,
                                                   require_optimizer=True) for role in ('actor','critic')}
    result.update(checkpoint_restore_tested=True, original_checkpoints_unchanged=True,
                  native_model_optimizer_payloads_exact=True, resumed_rollout_id=args.iteration+1,
                  additional_updates=1, source_revision=manifest['revision'])
    (output/'RESTORE_RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('snapshots','checkpoints')},indent=2))


if __name__ == '__main__':
    main()
