#!/usr/bin/env python3
"""Independently validate fresh parquet and native CPU input-consumption output."""
import argparse
import json
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', type=Path, default=Path(os.environ.get('AUTORLBENCH_WORKSPACE', '/workspace')))
    parser.add_argument('--results', type=Path, default=Path(os.environ.get('AUTORLBENCH_RESULTS', '/results')))
    parser.add_argument('--prepared', type=Path, default=Path(os.environ.get('AUTORLBENCH_PREPARED', '/autorlbench/prepared')))
    parser.add_argument('--environment', type=Path, default=Path(os.environ.get('AUTORLBENCH_ENVIRONMENT', '/autorlbench/environment')))
    args = parser.parse_args()
    sys.path.insert(0, str(args.environment))
    from input_contract import check_rows, digest, read_rows, regular, verify_pins, write_json
    results, workspace = args.results.resolve(), args.workspace.resolve()
    checks = {}
    failure = None
    try:
        grading = json.loads((Path(__file__).resolve().parents[1] / 'resources/grader_spec.json').read_text())
        report = regular(workspace / 'report.md', workspace)
        if not report.read_text().strip():
            raise ValueError('Missing or empty report.md')
        specification = json.loads((args.environment / 'contract_spec.json').read_text())
        verify_pins(workspace, specification)
        checks['non_target_code_and_inputs'] = True
        assets = args.prepared / 'input_assets'
        for relative, expected in specification['trusted_assets'].items():
            if digest(regular(assets / relative, assets)) != expected:
                raise ValueError('Trusted prepared asset changed')
        output = results / 'experiment_logs'
        receipt = json.loads(regular(output / 'receipt.json', results).read_text())
        if receipt.get('completed') is not True or receipt.get('training_steps') != 0:
            raise ValueError('Fresh CPU rerun did not complete')
        sources = {split: read_rows(assets / f'gsm8k/main/{split}-00000-of-00001.parquet') for split in ['train', 'test']}
        for split in ['train', 'test']:
            actual_path = regular(output / f'main/{split}.parquet', results)
            live_path = regular(workspace / f'data/preprocessed/{split}.parquet', workspace)
            if digest(actual_path) != digest(live_path) or digest(actual_path) != receipt['fresh_main_sha256'][split + '.parquet']:
                raise ValueError('Regenerated data and actual workspace data differ')
            check_rows(sources[split], read_rows(actual_path), split)
        checks['complete_regenerated_data'] = True
        checks['policy_prompts_and_reward_labels'] = True
        from omegaconf import OmegaConf
        config = OmegaConf.load(regular(output / 'resolved-config.yaml', results))
        paths = config.data.train_files
        paths = [paths] if isinstance(paths, str) else list(paths)
        expected_train = workspace / 'data/preprocessed/train.parquet'
        if [str(Path(p).resolve()) for p in paths] != [str(expected_train.resolve())]:
            raise ValueError('Launch input points elsewhere')
        if config.reward.custom_reward_function.path is not None or config.data.prompt_key != 'prompt':
            raise ValueError('Reward/input contract changed')
        if config.data.max_prompt_length != 512 or config.data.max_response_length != 512:
            raise ValueError('Input limits changed')
        if str(config.actor_rollout_ref.model.path) != str(workspace / 'model/Qwen2.5-1.5B-Instruct'):
            raise ValueError('Model path changed')
        consumed_path = regular(output / 'consumed.jsonl', results)
        if consumed_path.stat().st_size > 64 * 1024**2:
            raise ValueError('Consumption capture exceeds size bound')
        consumed = [json.loads(line) for line in consumed_path.read_text().split('\n') if line.strip()]
        if len(consumed) != grading['train_rows'] or sorted(row['index'] for row in consumed) != list(range(grading['train_rows'])):
            raise ValueError('Native dataset consumption is incomplete')
        actual = read_rows(expected_train)
        for row in consumed:
            expected = actual[row['index']]
            if row['prompt'] != expected['prompt'] or row['reward_model'] != expected['reward_model']:
                raise ValueError('Consumed prompt/label differs from regenerated data')
        if receipt['consumed_train_sha256'] != digest(expected_train):
            raise ValueError('Consumed data fingerprint differs')
        checks['launcher_and_native_dataset_binding'] = True
        if set(checks) != set(grading['checks']):
            raise ValueError('Grader specification and implemented checks differ')
    except Exception as error:
        failure = type(error).__name__ + ': ' + str(error)
    passed = failure is None
    result = {'passed': passed, 'score': 1.0 if passed else 0.0,
              'metrics': {'status': 'graded', 'checks': checks, 'failure': failure, 'training_steps': 0}}
    write_json(results / 'result.json', result)
    write_json(results / 'metrics.json', result['metrics'])
    print(json.dumps(result, ensure_ascii=False))
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
