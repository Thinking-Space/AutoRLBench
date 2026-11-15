"""Launch the archived native Miles train_async entrypoint, never a local trainer."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

WORKSPACE = Path(__file__).resolve().parents[1]
# Dependency checks must inspect the same bundled source as the child driver.
sys.path[:0] = [str(WORKSPACE / "vendor"), str(WORKSPACE)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation-backend", choices=["sglang", "hf-cpu"], default="hf-cpu")
    parser.add_argument("--print-command", action="store_true", help="Build launch arguments without importing GPU backends")
    options, extra = parser.parse_known_args()
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(WORKSPACE / "vendor"), str(WORKSPACE), env.get("PYTHONPATH", "")])
    env.update(PYTHONDONTWRITEBYTECODE="1", TOKENIZERS_PARALLELISM="false", NO_VCS_VERSION="1",
               MILES_EXPERIMENTAL_ROLLOUT_REFACTOR="0", MILES_EXPERIMENTAL_FT_TRAINER="0",
               RETRY_GENERATION_BACKEND=options.generation_backend,
               HARBOR_ASYNC_MAX_INFLIGHT_GROUPS="32", HARBOR_ASYNC_QUEUE_MAXSIZE="1",
               HARBOR_RESAMPLE_SAME_PROMPT_MAX_TIMES="1", CUDA_DEVICE_MAX_CONNECTIONS="1",
               WANDB_MODE="disabled", SWANLAB_API_KEY="")
    logs = WORKSPACE / "experiment_logs" / (time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + f"-{os.getpid()}")
    env["RETRY_SMOKE_LOG_DIR"] = str(logs)
    gpu_count = 1 if options.generation_backend == "hf-cpu" else 2
    assets = WORKSPACE / "assets/public-qwen2"
    args = [sys.executable, "scripts/native_entrypoint.py",
            "--train-backend", "megatron", "--actor-num-nodes", "1", "--actor-num-gpus-per-node", "1",
            "--rollout-num-gpus", "1", "--rollout-num-gpus-per-engine", "1", "--num-gpus-per-node", str(gpu_count),
            "--hf-checkpoint", str(assets), "--load", str(assets), "--megatron-to-hf-mode", "bridge",
            "--no-load-optim", "--no-load-rng", "--finetune", "--bf16", "--num-layers", "2", "--hidden-size", "128",
            "--ffn-hidden-size", "256", "--num-attention-heads", "4", "--group-query-attention",
            "--num-query-groups", "2", "--kv-channels", "32", "--vocab-size", "256",
            "--make-vocab-size-divisible-by", "128", "--swiglu", "--disable-bias-linear", "--add-qkv-bias",
            "--normalization", "RMSNorm", "--norm-epsilon", "1e-6", "--rotary-base", "10000",
            "--use-rotary-position-embeddings", "--untie-embeddings-and-output-weights",
            "--seq-length", "256", "--tensor-model-parallel-size", "1", "--pipeline-model-parallel-size", "1",
            "--context-parallel-size", "1", "--expert-model-parallel-size", "1", "--expert-tensor-parallel-size", "1",
            "--rollout-function-path", "runtime.public_smoke.rollout",
            "--data-source-path", "runtime.public_smoke.PublicDataSource",
            "--custom-generate-function-path", "runtime.public_generate.generate",
            "--dynamic-sampling-filter-path", "runtime.public_smoke.trigger_filter",
            "--custom-megatron-before-train-step-hook-path", "runtime.public_smoke.observe_train_step",
            "--start-rollout-id", "0", "--num-rollout", "3", "--rollout-batch-size", "1", "--n-samples-per-prompt", "4",
            "--global-batch-size", "4", "--micro-batch-size", "1", "--rollout-max-response-len", "8",
            "--rollout-temperature", "1.0", "--rollout-top-k", "-1", "--rollout-top-p", "1.0",
            "--advantage-estimator", "grpo", "--kl-coef", "0", "--entropy-coef", "0",
            "--eps-clip", "0.2", "--eps-clip-high", "0.28", "--optimizer", "adam", "--lr", "0.001",
            "--lr-decay-style", "constant", "--weight-decay", "0", "--attention-dropout", "0", "--hidden-dropout", "0",
            "--attention-backend", "flash", "--sglang-mem-fraction-static", "0.3",
            "--sglang-disable-cuda-graph", "--no-use-prometheus-scrape", "--update-weights-interval", "100"]
    if options.generation_backend == "hf-cpu":
        # Native framework option disables SGLang servers/weight sync ONLY.
        # Rollout callback, Ray actor, Megatron fwd/bwd/GRPO and optimizer stay real.
        args += ["--debug-train-only"]
    args += extra
    if options.print_command:
        import shlex
        print(shlex.join(args))
        return 0
    import torch
    required = 1 if options.generation_backend == "hf-cpu" else 2
    if torch.cuda.device_count() < required:
        raise RuntimeError(f"Native Miles smoke requires {required} visible CUDA GPU(s); no CPU training fallback")
    missing = []
    for name in ("ray", "megatron", "megatron.bridge", "transformer_engine", "sglang", "sglang_router"):
        try:
            found = importlib.util.find_spec(name)
        except ModuleNotFoundError:
            found = None
        if found is None:
            missing.append(name)
    if missing:
        raise RuntimeError("Native Miles runtime dependencies missing: " + ", ".join(missing))
    if not (assets / "public_asset.json").exists():
        subprocess.run([sys.executable, str(WORKSPACE / "scripts/prepare_public_assets.py")], check=True, env=env)
    logs.mkdir(parents=True)
    (logs / "launch.json").write_text(json.dumps({"argv": args, "generation_backend": options.generation_backend,
          "trainer": "archived Miles train_async + Ray + Megatron", "gpu_count": required,
          "historical_replay": False}, indent=2) + "\n")
    print(f"Smoke evidence: {logs}", flush=True)
    return subprocess.call(args, cwd=WORKSPACE, env=env)


if __name__ == "__main__":
    raise SystemExit(main())
