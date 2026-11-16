"""Own a local Ray cluster, then execute the bundled Miles async driver."""
import os
import runpy
from pathlib import Path

FORWARDED = ["PYTHONPATH", "PYTHONDONTWRITEBYTECODE", "NO_VCS_VERSION", "TOKENIZERS_PARALLELISM",
             "CUDA_DEVICE_MAX_CONNECTIONS", "WANDB_MODE", "PYTORCH_CUDA_ALLOC_CONF", "RUN_EVIDENCE_DIR",
             "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"]


def main():
    import ray
    import torch

    workspace = Path(__file__).resolve().parents[1]
    extra = [n for n in os.environ.get("FORWARD_ENV", "").split(",") if n]
    env_vars = {name: os.environ[name] for name in FORWARDED + extra if name in os.environ}
    gpus = torch.cuda.device_count()
    ray.init(address="local", num_gpus=gpus, num_cpus=max(16, gpus * 8), include_dashboard=False,
             log_to_driver=True, runtime_env={"env_vars": env_vars})
    try:
        runpy.run_path(str(workspace / "runtime/source/train_async_entrypoint.py"), run_name="__main__")
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
