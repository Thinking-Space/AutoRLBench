"""Portable paths and cache handoff for a GSM8K GRPO run."""
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / 'model/Qwen2.5-1.5B-Instruct'


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.partial')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def configure(run_name, workspace):
    assert run_name.replace('_', '').replace('-', '').isalnum()
    run = ROOT / 'outputs' / run_name
    run.mkdir(parents=True, exist_ok=True)
    temporary = ROOT / '.tmp' / run_name
    temporary.mkdir(parents=True, exist_ok=True)
    # Process managers use AF_UNIX sockets beneath TMPDIR;keep that spelling
    # short even when a workspace lives under a deep host directory.
    short_tmp = Path('/tmp') / ('gsm_' + hashlib.sha256(str(temporary).encode()).hexdigest()[:16])
    if short_tmp.is_symlink():
        assert short_tmp.resolve() == temporary.resolve()
    elif short_tmp.exists():
        raise RuntimeError('Temporary path already belongs to another process')
    else:
        short_tmp.symlink_to(temporary, target_is_directory=True)
    mapping = {
        'GSM_ROOT': str(ROOT), 'RUN_DIR': str(run),
        'PYTHONPATH': str(ROOT / 'scripts') + ':' + str(ROOT / 'verl'),
        'PYTHONDONTWRITEBYTECODE': '1', 'TOKENIZERS_PARALLELISM': 'false', 'OMP_NUM_THREADS': '4',
        'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'HF_DATASETS_OFFLINE': '1',
        'HF_HOME': str(ROOT / '.cache/huggingface'), 'XDG_CACHE_HOME': str(ROOT / '.cache'),
        'HF_DATASETS_CACHE': str(ROOT / '.cache/datasets'),
        'TRITON_CACHE_DIR': str(ROOT / '.cache' / run_name / 'triton'),
        'TORCHINDUCTOR_CACHE_DIR': str(ROOT / '.cache' / run_name / 'inductor'),
        'TORCH_EXTENSIONS_DIR': str(ROOT / '.cache/extensions'),
        'VLLM_CACHE_ROOT': str(ROOT / '.cache' / run_name / 'vllm'),
        'VLLM_CONFIG_ROOT': str(ROOT / '.cache/vllm-config'),
        'FLASHINFER_WORKSPACE_BASE': str(ROOT / '.cache/flashinfer'),
        'CUDA_CACHE_PATH': str(ROOT / '.cache' / run_name / 'cuda'),
        'TMPDIR': str(short_tmp), 'WANDB_MODE': 'offline',
        'WANDB_API_KEY': 'offline', 'WANDB_DIR': str(run),
        'WANDB_CACHE_DIR': str(ROOT / '.cache/wandb'),
        'VERL_FILE_LOGGER_PATH': str(run / 'metrics.jsonl'),
        'RAY_USAGE_STATS_ENABLED': '0', 'VLLM_NO_USAGE_STATS': '1', 'DO_NOT_TRACK': '1',
    }
    for key, value in mapping.items():
        if value.startswith(str(ROOT)) and key not in ['GSM_ROOT', 'RUN_DIR', 'PYTHONPATH', 'VERL_FILE_LOGGER_PATH']:
            Path(value).mkdir(parents=True, exist_ok=True)
    os.environ.update(mapping)
    return run
