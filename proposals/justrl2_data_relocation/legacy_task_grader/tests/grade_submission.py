#!/usr/bin/env python3
"""Stdlib-only structural grader for a fresh data-relocation training run.

No submitted Python, pickle, tensor, or training code is executed. ``--home`` is
explicit offline calibration: archives may omit large checkpoint payloads, but
all checkpoint receipts and every payload present must still validate.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys


RESOURCES = Path(__file__).resolve().parent.parent / 'resources'
_module_spec = importlib.util.spec_from_file_location(
    '_hidden_relocation_evidence_audit', RESOURCES / 'evidence_audit.py')
audit = importlib.util.module_from_spec(_module_spec)
_module_spec.loader.exec_module(audit)
EVIDENCE_ERRORS = (OSError, ValueError, TypeError, KeyError, IndexError,
                   StopIteration, UnicodeError, RecursionError, OverflowError)


def _evidence_tree(home):
    audit._require(home.is_dir() and not home.is_symlink(), 'Missing run home or linked run home')
    # Only these evidence trees are consumed. Never follow paths from config to
    # the model cache, prepared assets, student workspace, or historical logs.
    for name in ('INPUTS.json', 'RESULT.json', 'export', 'dataset', 'provenance', 'run', 'native'):
        root = home / name
        audit._require(not root.is_symlink(), f'Linked evidence is unsupported: {name}')
        if root.is_dir():
            for path in root.rglob('*'):
                audit._require(not path.is_symlink(), f'Linked evidence is unsupported: {path.relative_to(home)}')


def _completion(home, inputs, spec):
    receipts = [home / name for name in ('RESULT.json', 'export/RESULT.json')
                if (home / name).is_file()]
    audit._require(receipts, 'Missing root/export RESULT.json completion receipt')
    for path in receipts:
        result = audit._load(path)
        audit._require(type(result.get('returncode')) is int and result['returncode'] == 0
                       and result.get('job_id') == inputs['job_id'], 'Unsuccessful or mismatched completion receipt')
        if 'arm' in result:
            audit._require(result['arm'] == inputs['arm'], 'Completion arm label mismatch')
        if 'stage' in result:
            audit._require(result['stage'] == 'complete', 'Completion stage is not complete')
    returned = audit._load(home / 'run/TRAINING_RETURNED.json')
    audit._require(returned.get('training_returned') is True and returned.get('arm') == inputs['arm'],
                   'Invalid TRAINING_RETURNED completion/arm')
    audit._require(type(returned.get('physical_gpus')) is int
                   and returned['physical_gpus'] == spec['physical_gpus'], 'Expected three physical GPUs')
    audit._require(audit._number(returned.get('elapsed_seconds'), 'elapsed_seconds') > 0,
                   'Invalid completion elapsed time')
    packages = audit._object(returned.get('packages'), 'TRAINING_RETURNED packages')
    for name in ('torch', 'ray', 'sglang', 'transformers'):
        audit._text(packages.get(name), f'Native package {name}')


def _pins(home, inputs, config, spec, reference):
    for key in ('steps', 'relocate_at', 'model_revision', 'source_revision', 'model_files'):
        audit._require(inputs.get(key) == spec[key], f'Frozen INPUTS {key} mismatch')
    runtime = reference['expected_runtime_sha256']
    for key, filename in [('manifest_sha256', 'manifest.json'), ('eval_sha256', 'eval.jsonl')]:
        audit._require(inputs[key] == runtime[filename], f'Frozen {filename} hash mismatch')
    for key, expected in spec['config'].items():
        actual = config.get(key)
        audit._require(type(actual) is type(expected) and actual == expected, f'Frozen config {key} mismatch')
    source = audit._load(home / 'provenance/source_receipt.json')
    audit._require(source.get('source_archive_sha256') == spec['source_archive_sha256'],
                   'Frozen source archive mismatch')
    conversion = audit._load(home / 'provenance/conversion_receipt.json')
    audit._require(conversion.get('hf_files') == spec['public_model_files'], 'Frozen public model hashes mismatch')


def _frozen_data(home, inputs, reference, spec):
    initial, payloads, evaluation, after_hash = audit._audit_dataset(home, inputs)
    train = [row for row in reference['rows'] if row['split'] == 'train']
    fixed_eval = [row for row in reference['rows'] if row['split'] == 'eval']
    audit._require(len(initial) == spec['train_count'] and len(evaluation) == spec['eval_count'],
                   'Expected 64 training and 16 fixed evaluation identities')
    audit._require([row['id'] for row in initial] == [row['id'] for row in train], 'Frozen training order/IDs changed')
    audit._require([row['id'] for row in evaluation] == [row['id'] for row in fixed_eval], 'Frozen evaluation order/IDs changed')
    case = audit._load(home / 'provenance/CASE.json')
    actual = case['rows']
    expected = {row['id']: row for row in reference['rows']}
    audit._require(len(actual) == len(expected) and {row['id'] for row in actual} == set(expected),
                   'CASE must preserve every frozen source identity')
    for row in actual:
        frozen = expected[row['id']]
        for key in ('id', 'split', 'difficulty', 'source_line', 'prompt_sha256', 'label_sha256', 'content_sha256'):
            audit._require(row.get(key) == frozen[key], f'Frozen source {row["id"]}/{key} mismatch')
        # Old calibration CASE records predate public provenance. Both formats
        # must match identical frozen runtime bytes and content hashes.
        if 'public_file' in row:
            for key, selected in [('public_file', 'file'), ('source_row_id', 'uuid'),
                                  ('row_index_0based', 'row_index_0based'), ('public_record_sha256', 'public_record_sha256')]:
                audit._require(row.get(key) == frozen[selected], f'Public source {row["id"]}/{key} mismatch')
    if 'repo' in case['source']:
        audit._require(all(case['source'].get(key) == value for key, value in reference['source'].items()),
                       'Public dataset revision/source mismatch')
    runtime = reference['expected_runtime_sha256']
    for row in initial:
        path = f'archive/hard/{row["id"]}.json' if row['difficulty'] == 'hard' else row['path']
        payload = audit._load(home / 'dataset' / path)
        audit._require(row['sha256'] == runtime[row['path']], 'Backing hash differs from frozen selection')
        content = {'prompt': payload['prompt'], 'label': payload['label']}
        audit._require(audit._content_hash(content) == expected[row['id']]['content_sha256'],
                       'Backing content differs from frozen public source')
    for row in evaluation:
        content = {'prompt': row['prompt'], 'label': row['label']}
        audit._require(audit._content_hash(content) == expected[row['id']]['content_sha256'],
                       'Evaluation content differs from frozen public source')
    archive = home / 'dataset/archive/hard'
    audit._require({path.name for path in archive.iterdir()} ==
                   {row['id'] + '.json' for row in train if row['difficulty'] == 'hard'},
                   'Relocation archive does not contain exactly the original hard files')
    journal = audit._load(home / 'dataset/relocation.json')
    audit._require(journal == audit._load(home / 'provenance/final_relocation.json'),
                   'Live relocation journal differs from provenance')
    return initial, payloads, evaluation, after_hash


def _event_coverage(events, inputs, config, initial, payloads, after_hash, spec):
    steps = audit._audit_events(events, inputs, config, initial, payloads, after_hash)
    for step in steps:
        audit._require(step['terminal_accounting'], 'Missing terminal accounting')
        endpoint = next(row for row in events if row['kind'] == 'rollout_end' and row['step'] == step['step'])
        audit._require(set(endpoint['metrics']) == set(step['metrics']), 'Suppressed required accounting metrics')
        audit._require(step['abort_rate'] == 0 and step['hard_fraction'] == spec['post_hard_fraction'],
                       'Every batch must have zero file aborts and one-half hard samples')
    return steps


def _evaluation(run, config, evaluation, spec):
    audit._require({path.name for path in run.glob('eval_*.json')} == set(spec['eval_phases']),
                   'Missing/extra structured fixed evaluation phases')
    summaries, identities, output_format = audit._audit_eval(run, config, evaluation, spec['steps'])
    audit._require(output_format == 'structured', 'Structured evaluation is required')
    dataset = audit._repr_fields(config['eval_datasets'][0], 'EvalDatasetConfig')
    for key, expected in {'max_response_len': config['eval_max_response_len'],
                          'temperature': config['eval_temperature'], 'top_p': 1.0, 'top_k': -1,
                          'rm_type': None, 'custom_generate_function_path': None,
                          'input_key': 'prompt', 'label_key': 'label', 'metadata_key': 'metadata',
                          'metadata_overrides': {}, 'stop': None, 'stop_token_ids': None,
                          'min_new_tokens': None}.items():
        value = audit._literal(dataset, key)
        audit._require(type(value) is type(expected) and value == expected, f'Frozen eval dataset {key} mismatch')
    # The pinned tokenizer renders these single-user prompts with this template.
    # Bind rendered prompts to source bytes, not merely to the other snapshots.
    for identity, source in zip(identities, evaluation):
        rendered = ''.join(f'<|im_start|>{message["role"]}\n{message["content"]}<|im_end|>\n'
                           for message in source['prompt']) + '<|im_start|>assistant\n'
        audit._require(identity['prompt'] == rendered, 'Evaluation prompt differs from frozen source/template')
    return summaries


def _check_inventory(root, inventory, context):
    inventory = audit._inventory(inventory, context)
    for name, expected in inventory.items():
        path = root / name
        audit._require(path.is_file() and path.stat().st_size == expected['size']
                       and audit._hash(path) == expected['sha256'], f'{context}: payload hash/size mismatch: {name}')


def _checkpoint_payloads(home, inputs, provenance, fresh):
    # The base audit rehashes every exported final payload that is present.
    # Fresh harness runs must contain all payloads, whereas offline archives
    # explicitly retain only receipts for the multi-GB model/optimizer files.
    if fresh:
        audit._require(all(provenance['checkpoint_payloads_rehashed'].values()),
                       'Fresh run is missing final checkpoint model/optimizer payloads')
    native = home / 'native'
    if fresh or native.exists():
        before = audit._load(home / 'provenance/native_all_files_before.json')
        after = audit._load(home / 'provenance/native_all_files_after.json')
        audit._require(before == after, 'Original native directory changed during training')
        actual_names = {str(path.relative_to(native)) for path in native.rglob('*') if path.is_file()}
        audit._require(actual_names == set(before), 'Original native file inventory changed')
        _check_inventory(native, after, 'Original native checkpoint')
        conversion = audit._load(home / 'provenance/conversion_receipt.json')
        selected = conversion['native_checkpoint']['selected']
        for name, expected in inputs['native_checkpoint_files'].items():
            audit._require(after.get(str(Path(selected) / name)) == expected,
                           'Original native payloads disagree with conversion/INPUTS')


def inspect_solution(home, *, fresh=False):
    """Validate structural coverage, raising on incomplete or inconsistent evidence."""
    home = Path(home)
    spec = audit._load(RESOURCES / 'grader_spec.json')
    reference = audit._load(RESOURCES / 'reference_selection.json')
    _evidence_tree(home)
    inputs = audit._audit_inputs(home)
    _completion(home, inputs, spec)
    config = audit._load(home / 'run/effective_config.json')
    audit._audit_config(config, inputs)
    _pins(home, inputs, config, spec, reference)
    initial, payloads, evaluation, after_hash = _frozen_data(home, inputs, reference, spec)
    events = audit._lines(home / 'run/events.jsonl')
    steps = _event_coverage(events, inputs, config, initial, payloads, after_hash, spec)
    optimizer = audit._audit_optimizer(home / 'run', spec['steps'])
    eval_summaries = _evaluation(home / 'run', config, evaluation, spec)
    provenance = audit._audit_provenance(home, inputs, config, events, steps)
    audit._require(provenance['complete'], 'Missing full provenance')
    _checkpoint_payloads(home, inputs, provenance, fresh)
    return {'arm': inputs['arm'], 'validated_steps': len(steps), 'final_step': spec['final_step'],
            'structural_coverage': 1, 'pre': audit._phase(steps[:spec['relocate_at']]),
            'post': audit._phase(steps[spec['relocate_at']:]), 'optimizer': optimizer,
            'evaluation': eval_summaries, 'provenance': provenance, 'offline_calibration': not fresh}


def grade_solution(home) -> int:
    """Return 0/1 from supplied run-home evidence, including recorded archives.

    The default CLI adds mandatory payload presence for a fresh harness run.
    No arm label is interpreted as a success or failure outcome.
    """
    try:
        inspect_solution(home)
    except EVIDENCE_ERRORS:
        return 0
    return 1


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    audit._require(not path.is_symlink(), f'Linked output is unsupported: {path.name}')
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + '\n')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=Path(os.environ.get('AUTORLBENCH_RESULTS', '/results')))
    parser.add_argument('--home', type=Path, help='Explicit offline calibration run home (never a default fallback)')
    args = parser.parse_args(argv)
    results = args.results.expanduser().resolve()
    home = args.home.expanduser() if args.home is not None else results / 'experiment_logs/run-home'
    errors, details = [], {}
    try:
        if args.home is None:
            audit._require(not (results / 'experiment_logs').is_symlink(), 'Linked experiment_logs is unsupported')
        details = inspect_solution(home, fresh=args.home is None)
    except EVIDENCE_ERRORS as exc:
        errors.append(f'{type(exc).__name__}: {exc}')
    score = int(not errors)
    metrics = {'task_id': 'justrl2_data_relocation', 'metric_key': 'structural_coverage',
               'structural_coverage': score, 'required_final_score': 1, 'final_step': 7,
               'score': score, 'case_score': score, 'passed': bool(score), 'errors': errors,
               'deterministic': True, 'judge_used': False, 'heldout_accuracy_threshold': None,
               'offline_calibration': args.home is not None, 'evidence': details}
    result = {'passed': bool(score), 'score': score, 'metrics': metrics}
    experiment = {'task_id': metrics['task_id'],
                  'logs': {'source_logs': [str(home) if args.home else 'experiment_logs/run-home']},
                  'metrics': {'metric_key': metrics['metric_key'], 'metric_status': 'available' if score else 'invalid',
                              'structural_coverage': score, 'final_step': 7,
                              'eval_points': details.get('evaluation', []), 'errors': errors}}
    try:
        for name, value in [('experiment_metrics.json', experiment), ('metrics.json', metrics), ('result.json', result)]:
            _write_json(results / name, value)
    except EVIDENCE_ERRORS as exc:
        print(f'Cannot write grader outputs: {exc}', file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    if errors:
        print('; '.join(errors), file=sys.stderr)
    return 0 if score else 1


if __name__ == '__main__':
    raise SystemExit(main())
