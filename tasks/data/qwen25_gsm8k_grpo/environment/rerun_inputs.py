"""Run the submitted production preprocessing and dataset path on CPU only."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid

from input_contract import digest, regular, verify_pins, write_json


def run(command, env, cwd, log):
    with log.open('w') as stream:
        result = subprocess.run(command, cwd=cwd, env=env, stdout=stream, stderr=subprocess.STDOUT, timeout=240)
    if result.returncode:
        raise RuntimeError(f'Production command failed with exit {result.returncode};see {log.name}')


def main():
    workspace = Path(os.environ['AUTORLBENCH_WORKSPACE']).resolve()
    results = Path(os.environ['AUTORLBENCH_RESULTS']).resolve()
    environment = Path(os.environ['AUTORLBENCH_ENVIRONMENT']).resolve()
    prepared = Path(os.environ['AUTORLBENCH_PREPARED']).resolve()
    specification = json.loads((environment / 'contract_spec.json').read_text())
    output = results / 'experiment_logs'
    output.mkdir(parents=True, exist_ok=True)
    receipt = {'schema_version': 1, 'started_at': time.time(), 'training_steps': 0, 'gpu_required': False}
    try:
        verify_pins(workspace, specification)
        assets = prepared / 'input_assets'
        for relative, expected in specification['trusted_assets'].items():
            if digest(regular(assets / relative, assets)) != expected:
                raise ValueError('Prepared source input changed: ' + relative)
        # Remove only the staged run's rebuildable artifacts. Existing files,
        # source hashes and completion markers cannot stand in for regeneration.
        data = workspace / 'data/preprocessed'
        if data.is_symlink():
            raise ValueError('Processed data directory is a symlink')
        if data.exists():
            shutil.rmtree(data)
        env = {**os.environ, 'PYTHONPATH': str(workspace / 'scripts') + ':' + str(workspace / 'verl'),
               'PYTHONDONTWRITEBYTECODE': '1', 'HF_HUB_OFFLINE': '1', 'HF_DATASETS_OFFLINE': '1',
               'TRANSFORMERS_OFFLINE': '1', 'TOKENIZERS_PARALLELISM': 'false'}
        # This is the same data generation entry used in prepare.
        run([sys.executable, str(environment / 'build_data.py')], env, workspace, output / 'regenerate.log')
        main_data = output / 'main'
        main_data.mkdir()
        for name in ['train.parquet', 'test.parquet']:
            source = regular(data / name, workspace)
            shutil.copyfile(source, main_data / name)
        receipt['fresh_main_sha256'] = {name: digest(main_data / name) for name in ['train.parquet', 'test.parquet']}

        # Resolve the real launch configuration without creating Ray or workers.
        name = 'inputs_' + uuid.uuid4().hex[:12]
        run(['bash', str(workspace / 'run_gsm8k_grpo.sh'), '--input-config', '--run', name],
            env, workspace, output / 'launch.log')
        config_path = workspace / 'outputs' / name / 'resolved-config.yaml'
        shutil.copyfile(regular(config_path, workspace), output / 'resolved-config.yaml')
        from omegaconf import OmegaConf
        config = OmegaConf.load(config_path)
        actual = config.data.train_files
        paths = [actual] if isinstance(actual, str) else list(actual)
        if [str(Path(p).resolve()) for p in paths] != [str((data / 'train.parquet').resolve())]:
            raise ValueError('Launcher is not bound to the regenerated training data')
        if config.reward.custom_reward_function.path is not None or config.data.prompt_key != 'prompt':
            raise ValueError('Reward or prompt contract changed')
        from transformers import AutoTokenizer
        from verl.utils.dataset.rl_dataset import RLHFDataset
        tokenizer = AutoTokenizer.from_pretrained(assets / 'tokenizer', local_files_only=True)
        dataset = RLHFDataset(paths, tokenizer, config.data, processor=None)
        with (output / 'consumed.jsonl').open('w') as stream:
            for row in dataset:
                stream.write(json.dumps({'index': row['index'], 'prompt': row['raw_prompt'],
                                         'reward_model': row['reward_model']}, ensure_ascii=False) + '\n')
        receipt['consumed_rows'] = len(dataset)
        receipt['consumed_train_sha256'] = digest(data / 'train.parquet')
        verify_pins(workspace, specification)
        receipt['completed'] = True
    except Exception as error:
        receipt.update(completed=False, error=type(error).__name__ + ': ' + str(error))
    finally:
        receipt['elapsed_s'] = time.time() - receipt['started_at']
        write_json(output / 'receipt.json', receipt)
    print(json.dumps({'completed': receipt['completed'], 'elapsed_s': receipt['elapsed_s'], 'training_steps': 0}), flush=True)
    # A valid rerun may produce defective data;the independent verifier assigns
    # the binary score from the actual rows,not a candidate-reported metric.
    return 0 if receipt['completed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
