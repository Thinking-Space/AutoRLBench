"""Own a local Ray cluster, then execute the unmodified Miles driver."""
import os
from pathlib import Path
import runpy


def main():
    import ray

    workspace = Path(__file__).resolve().parents[1]
    names = ["NO_VCS_VERSION", "PYTHONPATH", "PYTHONDONTWRITEBYTECODE", "TOKENIZERS_PARALLELISM", "RETRY_GENERATION_BACKEND",
             "RETRY_SMOKE_LOG_DIR", "HARBOR_ASYNC_MAX_INFLIGHT_GROUPS", "HARBOR_ASYNC_QUEUE_MAXSIZE",
             "HARBOR_RESAMPLE_SAME_PROMPT_MAX_TIMES", "MILES_EXPERIMENTAL_ROLLOUT_REFACTOR",
             "MILES_EXPERIMENTAL_FT_TRAINER", "CUDA_DEVICE_MAX_CONNECTIONS", "WANDB_MODE", "SWANLAB_API_KEY"]
    required = 1 if os.environ["RETRY_GENERATION_BACKEND"] == "hf-cpu" else 2
    ray.init(address="local", num_gpus=required, num_cpus=max(8, required * 4),
             include_dashboard=False, log_to_driver=True,
             runtime_env={"env_vars": {name: os.environ[name] for name in names}})
    try:
        runpy.run_path(str(workspace / "runtime/source/train_async_entrypoint.py"), run_name="__main__")
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
