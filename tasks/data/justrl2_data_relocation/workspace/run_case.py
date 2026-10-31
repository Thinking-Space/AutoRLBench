#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run one public data-relocation arm in a new home using three visible GPUs.

Fresh assets are converted with one GPU before native three-GPU training. A
previously produced --reuse-native directory must carry a valid conversion
receipt and is copied and rehashed. No cluster API or job submission is used.
"""
import argparse
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import uuid

from prepare_assets import (ARCHIVE, HERE, file_record, file_sha, inventory, read_json,
                            safe_path, verify_converter_input, verify_file,
                            verify_inventory, verify_prepared, write_json)

PLUGINS = ('storage.py', 'prepare_data.py', 'relocation_rollout.py', 'run_native.py', 'stage_runtime.py', 'train.sh')
CONVERSION_SCRIPT = '''set -euo pipefail
source justrl2/model_args/minicpm5-2b.sh
python3 tools/convert_hf_to_torch_dist.py "${MODEL_ARGS[@]}" --attention-backend fused --no-gradient-accumulation-fusion --hf-checkpoint "$HF_SOURCE" --save "$NATIVE_OUTPUT"
'''


def clean_env(environ):
    excluded = {'WORLD_SIZE', 'RANK', 'LOCAL_RANK', 'LOCAL_WORLD_SIZE', 'GROUP_RANK',
                'ROLE_RANK', 'ROLE_WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT', 'PYTHONPATH',
                'BENCH_EVIDENCE_DIR', 'TENSORBOARD_DIR', 'HF_SOURCE', 'NATIVE_OUTPUT'}
    return {key: value for key, value in environ.items()
            if key not in excluded and not key.startswith(('RELOCATION_', 'MODEL_ARGS_', 'TORCHELASTIC_'))}


def conversion_env(code, model, native, environ, port):
    env = clean_env(environ)
    visible = environ.get('CUDA_VISIBLE_DEVICES', '0').split(',')
    if not visible[0].strip() or visible[0].strip() == '-1':
        raise ValueError('Native conversion requires a visible GPU')
    env.update(CUDA_VISIBLE_DEVICES=visible[0].strip(), WORLD_SIZE='1', RANK='0', LOCAL_RANK='0',
               MASTER_ADDR='127.0.0.1', MASTER_PORT=str(port),
               PYTHONPATH=f'{code}:{code}/Megatron-LM:{code}/sglang/python',
               HF_SOURCE=str(model), NATIVE_OUTPUT=str(native), PYTHONUNBUFFERED='1',
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', OMP_NUM_THREADS='1',
               CUDA_DEVICE_MAX_CONNECTIONS='1', NVTE_FUSED_ATTN='1', NVTE_FLASH_ATTN='0', NVTE_UNFUSED_ATTN='0')
    return env


def run_logged(command, *, cwd, env, log):
    """Keep raw merged stdout/stderr bytes even when a subprocess fails."""
    with Path(log).open('xb') as stream:
        process = subprocess.Popen(command, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        try:
            while True:
                block = process.stdout.read1(64 * 1024)
                if not block:
                    break
                stream.write(block)
                stream.flush()
                if hasattr(sys.stdout, 'buffer'):
                    sys.stdout.buffer.write(block)
                else:
                    sys.stdout.write(block.decode('utf-8', errors='replace'))
                sys.stdout.flush()
            return process.wait()
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise
        finally:
            process.stdout.close()


def verify_native_inputs(native, receipt):
    if receipt['iteration'] != 'release' or receipt['selected'] != 'release':
        raise ValueError('Expected the public release checkpoint')
    tracker = native / 'latest_checkpointed_iteration.txt'
    if tracker.is_symlink() or not tracker.is_file() or tracker.read_text().strip() != 'release':
        raise ValueError('Native checkpoint tracker changed')
    files = receipt['referenced_files']
    if not files or '.metadata' not in files or not any(name.endswith('.distcp') for name in files):
        raise ValueError('Native checkpoint has no metadata/referenced payload')
    for name, expected in files.items():
        verify_file(safe_path(native / 'release', name), expected)
    return {name: file_record(native / 'release' / name) for name in files}


def source_inventory_hash(source_receipt):
    import json
    return hashlib.sha256(json.dumps(source_receipt['original_files'], sort_keys=True).encode()).hexdigest()


def verify_conversion(native, model, converter, assets, source_receipt):
    receipt = read_json(native / 'conversion_receipt.json')
    if receipt['model_revision'] != assets['model_revision'] or receipt['source_revision'] != assets['justrl2_revision']:
        raise ValueError('Native conversion revisions differ from public pins')
    if receipt['source_inventory_sha256'] != source_inventory_hash(source_receipt):
        raise ValueError('Native conversion source inventory differs from public pins')
    if receipt['hf_files'] != {a['path']: a['sha256'] for a in assets['assets']}:
        raise ValueError('Native conversion HF assets differ from public pins')
    if receipt['converter_input'] != converter:
        raise ValueError('Native converter input differs from prepared exact safetensors')
    verify_converter_input(model, receipt['converter_input'], assets)
    if receipt['conversion_log_sha256'] != file_sha(native / 'conversion.log'):
        raise ValueError('Native conversion log hash mismatch')
    verify_native_inputs(native, receipt['native_checkpoint'])
    from runtime_support.checkpoint_evidence import inspect_checkpoint
    if inspect_checkpoint(native, expected_iteration='release') != receipt['native_checkpoint']:
        raise ValueError('Native checkpoint metadata differs from conversion receipt')
    return receipt


def convert_native(home, code, model, converter, assets, source_receipt, port):
    native = home / 'native'
    native.mkdir()
    env = conversion_env(code, model, native, os.environ, port)
    write_json(home / 'conversion_invocation.json', {
        'command': ['bash', '-c', CONVERSION_SCRIPT], 'cwd': str(code),
        'environment': {k: env[k] for k in ('CUDA_VISIBLE_DEVICES', 'WORLD_SIZE', 'RANK', 'LOCAL_RANK',
                                           'MASTER_ADDR', 'MASTER_PORT', 'PYTHONPATH', 'HF_SOURCE', 'NATIVE_OUTPUT')},
    })
    returncode = run_logged(['bash', '-c', CONVERSION_SCRIPT], cwd=code, env=env, log=native / 'conversion.log')
    if returncode:
        raise subprocess.CalledProcessError(returncode, 'native HF conversion')
    from runtime_support.checkpoint_evidence import inspect_checkpoint
    receipt = {
        'source_revision': assets['justrl2_revision'], 'model_revision': assets['model_revision'],
        'source_inventory_sha256': source_inventory_hash(source_receipt),
        'converter_input': converter, 'hf_files': {a['path']: a['sha256'] for a in assets['assets']},
        'native_checkpoint': inspect_checkpoint(native, expected_iteration='release'),
        'conversion_log_sha256': file_sha(native / 'conversion.log'),
    }
    write_json(native / 'conversion_receipt.json', receipt)
    return verify_conversion(native, model, converter, assets, source_receipt)


def stage_source(home, prepared, assets):
    command = [sys.executable, str(home / 'plugin/stage_runtime.py'),
               '--source-archive', str(prepared / ARCHIVE), '--helpers', str(HERE / 'runtime_support'),
               '--output', str(home / 'source')]
    returncode = run_logged(command, cwd=HERE, env=clean_env(os.environ), log=home / 'staging.log')
    if returncode:
        raise subprocess.CalledProcessError(returncode, command)
    code = home / 'source' / ('JustRL-II-' + assets['justrl2_revision'])
    receipt = read_json(home / 'source/source_receipt.json')
    if (receipt['revision'] != assets['justrl2_revision'] or
            receipt['source_archive_sha256'] != assets['public_source_archive']['sha256'] or
            receipt['original_files'] != read_json(HERE / 'runtime_support/source_files.json')['files']):
        raise ValueError('Staged source receipt differs from public pins')
    if receipt['instrumented_files'].keys() != receipt['original_files'].keys():
        raise ValueError('Staged source inventory lost files')
    for name, digest in receipt['instrumented_files'].items():
        if file_sha(safe_path(code, name)) != digest:
            raise ValueError('Staged source file changed: ' + name)
    for name, digest in receipt['helpers'].items():
        if file_sha(safe_path(code / 'benchmark', name)) != digest:
            raise ValueError('Staged runtime helper changed: ' + name)
    return code, receipt


def training_env(args, home, code, model):
    env = clean_env(os.environ)
    env.update(RELOCATION_CODE=str(code), RELOCATION_PLUGIN=str(home / 'plugin'),
               RELOCATION_DATA=str(home / 'dataset'), RELOCATION_OUTPUT=str(home / 'run'),
               RELOCATION_MODEL=str(model), RELOCATION_NATIVE=str(home / 'native'),
               RELOCATION_ARM=args.arm, RELOCATION_STEPS=str(args.steps), RELOCATION_AT=str(args.relocate_at),
               RELOCATION_BATCH=str(args.batch), RELOCATION_SAMPLES=str(args.samples),
               RELOCATION_MAX_RESPONSE=str(args.max_response), RELOCATION_MAX_TOTAL_TOKENS=str(args.max_total_tokens),
               PYTHONUNBUFFERED='1')
    return env


def check_training_gpus():
    import torch
    if torch.cuda.device_count() != 3:
        raise ValueError('Expose exactly three GPUs for actor1 + critic1 + rollout1; conversion temporarily uses the first')


def run_case(args):
    prepared, home = args.prepared.resolve(), args.output.resolve()
    if home.exists():
        raise FileExistsError('Use a new run home: ' + str(home))
    if home.is_relative_to(prepared) or prepared.is_relative_to(home):
        raise ValueError('Prepared assets and output home must be separate directories')
    assets, converter = verify_prepared(prepared)
    check_training_gpus()
    home.mkdir(parents=True)
    print('RELOCATION_HOME=' + str(home), flush=True)
    result = {'home': str(home), 'arm': args.arm, 'returncode': 1, 'stage': 'staging'}
    provenance = home / 'provenance'
    provenance.mkdir()
    native_before = None
    reused_before = None
    native = home / 'native'
    model = prepared / 'hf-safetensors'
    try:
        (home / 'plugin').mkdir()
        for name in PLUGINS:
            shutil.copyfile(HERE / 'runtime' / name, home / 'plugin' / name)
        shutil.copytree(prepared / 'dataset', home / 'dataset')
        verify_inventory(home / 'dataset', inventory(prepared / 'dataset'))
        for filename, target in (('CASE.json', 'CASE.json'), ('manifest.json', 'initial_manifest.json'),
                                 ('eval.jsonl', 'eval.jsonl'), ('PUBLIC_PROVENANCE.json', 'PUBLIC_PROVENANCE.json')):
            shutil.copyfile(home / 'dataset' / filename, provenance / target)
        code, source_receipt = stage_source(home, prepared, assets)
        shutil.copyfile(home / 'source/source_receipt.json', provenance / 'source_receipt.json')
        result['stage'] = 'conversion'
        if args.reuse_native:
            reused = args.reuse_native.resolve()
            verify_conversion(reused, model, converter, assets, source_receipt)
            reused_before = inventory(reused)
            shutil.copytree(reused, native)
            verify_inventory(native, reused_before)
            conversion = verify_conversion(native, model, converter, assets, source_receipt)
        else:
            conversion = convert_native(home, code, model, converter, assets, source_receipt, args.conversion_port)
        native_inputs = verify_native_inputs(native, conversion['native_checkpoint'])
        native_before = inventory(native)
        write_json(provenance / 'native_all_files_before.json', native_before)
        shutil.copyfile(native / 'conversion_receipt.json', provenance / 'conversion_receipt.json')
        inputs = {
            'synthetic_fault_injection': True, 'job_id': 'public-' + uuid.uuid4().hex,
            'arm': args.arm, 'steps': args.steps, 'relocate_at': args.relocate_at,
            'model_files': {name: row['sha256'] for name, row in converter['files'].items()},
            'model_revision': assets['model_revision'], 'native_checkpoint_files': native_inputs,
            'source_revision': assets['justrl2_revision'],
            'case_sha256': file_sha(home / 'dataset/CASE.json'),
            'manifest_sha256': file_sha(home / 'dataset/manifest.json'), 'eval_sha256': file_sha(home / 'dataset/eval.jsonl'),
            'plugin_files': {name: file_sha(home / 'plugin' / name) for name in PLUGINS},
        }
        write_json(home / 'INPUTS.json', inputs)
        result.update(stage='training', job_id=inputs['job_id'])
        returncode = run_logged(['bash', str(home / 'plugin/train.sh')], cwd=code,
                                env=training_env(args, home, code, model), log=home / 'stdout.log')
        result['returncode'] = returncode
        if returncode == 0:
            result.update(stage='checkpoint_validation', returncode=1)
            from runtime_support.checkpoint_evidence import inspect_checkpoint
            outputs = {role: inspect_checkpoint(home / 'run' / role, expected_iteration=args.steps - 1,
                                                require_optimizer=True) for role in ('actor', 'critic')}
            write_json(provenance / 'output_checkpoints.json', outputs)
            returned = read_json(home / 'run/TRAINING_RETURNED.json')
            if returned.get('training_returned') is not True or returned.get('arm') != args.arm:
                raise ValueError('Missing native training completion marker')
            result.update(stage='complete', returncode=0)
    except BaseException as exc:
        result.update(returncode=1, error=str(exc))
        raise
    finally:
        try:
            for name in ('manifest.json', 'relocation.json'):
                path = home / 'dataset' / name
                if path.exists():
                    shutil.copyfile(path, provenance / ('final_' + name))
            if native_before is not None:
                after = inventory(native)
                write_json(provenance / 'native_all_files_after.json', after)
                if native_before != after:
                    raise ValueError('Original native checkpoint directory changed during training')
                unchanged = verify_native_inputs(native, conversion['native_checkpoint'])
                write_json(provenance / 'original_native_unchanged.json', unchanged)
                verify_converter_input(model, converter, assets)
            if reused_before is not None:
                verify_inventory(args.reuse_native.resolve(), reused_before)
        except BaseException as exc:
            result.update(returncode=1, error=str(exc), stage='input_validation')
            raise
        finally:
            write_json(home / 'RESULT.json', result)
    return result['returncode']


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='New complete run home')
    parser.add_argument('--arm', choices=('faulty',), default='faulty')
    parser.add_argument('--steps', type=int, default=8)
    parser.add_argument('--relocate-at', '--at', type=int, default=4)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--samples', type=int, default=2)
    parser.add_argument('--max-response', type=int, default=32768)
    parser.add_argument('--max-total-tokens', type=int, default=262144)
    parser.add_argument('--conversion-port', type=int, default=29365)
    parser.add_argument('--reuse-native', type=Path, help='Receipt-validated native directory, copied into the new home')
    args = parser.parse_args(argv)
    if args.relocate_at < 2 or args.steps != 2 * args.relocate_at:
        parser.error('--steps must equal 2 * --relocate-at, with --relocate-at >= 2')
    if any(value <= 0 for value in (args.batch, args.samples, args.max_response, args.max_total_tokens)):
        parser.error('batch, samples and token budgets must be positive')
    if not 1 <= args.conversion_port <= 65535:
        parser.error('--conversion-port must be in 1..65535')
    return args


def main(argv=None):
    args = parse_args(argv)
    try:
        return run_case(args)
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
