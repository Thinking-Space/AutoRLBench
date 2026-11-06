#!/usr/bin/env python3
"""Run the historical experiment in the foreground with benchmark output paths."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    results = Path(os.environ.get('AUTORLBENCH_RESULTS', ROOT / 'results/new_run')).resolve()
    logs = results / 'experiment_logs'
    logs.mkdir(parents=True, exist_ok=True)
    output_root = results / 'runtime_outputs'
    for name in ['logs', 'wandb', 'outputs', 'checkpoints', 'verl_step_records', 'ray', 'ray-spill']:
        (output_root / name).mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(PYTHON=env.get('PYTHON', sys.executable),
               VLLM_PYTHON=env.get('VLLM_PYTHON', env.get('PYTHON', sys.executable)),
               GLM_PYTHON=env.get('GLM_PYTHON', '/opt/maestro-glm/bin/python'),
               LOG_DIR=str(logs), WANDB_DIR=str(logs), WANDB_MODE='offline',
               PYTHONDONTWRITEBYTECODE='1')
    env['TOOL_SERVER_URL'] = 'http://127.0.0.1:30381/get_observation'
    env['MAESTRO_LOG_ROOT'] = str(output_root / 'logs')
    env['MAESTRO_RUNTIME_TMP'] = str(output_root / 'tmp')
    env['RAY_SESSION_ROOT'] = str(output_root / 'ray')
    env['RAY_object_spilling_directory'] = str(output_root / 'ray-spill')
    env['HF_HOME'] = str(output_root / 'huggingface')
    env['HUGGINGFACE_HUB_CACHE'] = str(output_root / 'huggingface/hub')
    env['TRANSFORMERS_CACHE'] = str(output_root / 'huggingface/hub')
    env['VLLM_CACHE_ROOT'] = str(output_root / 'vllm-cache')
    env['TORCHINDUCTOR_CACHE_DIR'] = str(output_root / 'torchinductor-cache')
    env['TRITON_CACHE_DIR'] = str(output_root / 'triton-cache')
    env['WANDB_DATA_DIR'] = str(output_root / 'wandb/data')
    env['WANDB_CACHE_DIR'] = str(output_root / 'wandb/cache')
    env['WANDB_CONFIG_DIR'] = str(output_root / 'wandb/config')
    command = ['bash', 'scripts/train_4_gpus.sh', 'trainer.resume_mode=disable',
               f'actor_rollout_ref.agent.tool_server_url={env["TOOL_SERVER_URL"]}',
               f'hydra.run.dir={logs}/hydra_runs/current',
               'trainer.experiment_name=maestro-grpo',
               f'trainer.default_local_dir={output_root}/checkpoints',
               f'trainer.rollout_data_dir={output_root}/verl_step_records']
    if args.dry_run:
        env['DRY_RUN'] = '1'
    elif not Path(env['GLM_PYTHON']).is_file():
        raise SystemExit('Missing GLM interpreter; use the maestro-vllm011 benchmark image.')
    # Generate a runtime-only output-routing overlay; keep the historical entry point intact.
    historical = (ROOT / 'scripts/train_4_gpus.sh').read_text()
    replacements = [
        ('cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."', f'cd -- "{ROOT}"'),
        ('mkdir -p .cache/ray .cache/ray-spill checkpoints verl_step_records',
         'mkdir -p .cache/ray .cache/ray-spill checkpoints verl_step_records "${MAESTRO_RUNTIME_TMP}"'),
        ('ray_storage_dir="$(mktemp -d "$PWD/.cache/ray/run.XXXXXX")"',
         'ray_storage_dir="$(mktemp -d "${RAY_SESSION_ROOT}/run.XXXXXX")"'),
        ('export RAY_object_spilling_directory="$PWD/.cache/ray-spill"',
         'export RAY_object_spilling_directory="${RAY_object_spilling_directory}"'),
        ('export HF_HOME="${HF_HOME:-.cache/huggingface}"', 'export HF_HOME="${HF_HOME}"'),
        ('export VLLM_CACHE_ROOT="$PWD/.cache/vllm"', 'export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT}"'),
        ('export TORCHINDUCTOR_CACHE_DIR="$PWD/.cache/torchinductor"', 'export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR}"'),
        ('export TRITON_CACHE_DIR="$PWD/.cache/triton"', 'export TRITON_CACHE_DIR="${TRITON_CACHE_DIR}"'),
        ('action_stop_tokens_file="$(mktemp .cache/action_stop_tokens.XXXXXX)"',
         'action_stop_tokens_file="$(mktemp "${MAESTRO_RUNTIME_TMP}/action_stop_tokens.XXXXXX")"'),
        ('port=$(shuf -i 30000-31000 -n 1)', 'port=30381'),
        ('tool_server_url=http://$host:$port/get_observation', 'tool_server_url="${TOOL_SERVER_URL}"'),
    ]
    for old, new in replacements:
        if historical.count(old) != 1:
            raise SystemExit(f'Historical launcher overlay target changed: {old!r}')
        historical = historical.replace(old, new, 1)
    overlay = output_root / 'train_4_gpus.sh'
    overlay.write_text(historical)
    overlay.chmod(0o755)
    command[1] = str(overlay)
    # The launcher monitors its service children and remains attached until training exits.
    return subprocess.run(command, cwd=ROOT, env=env, check=False).returncode


if __name__ == '__main__':
    raise SystemExit(main())
