"""Launch the bundled Miles fully-async GRPO trainer on the public workload.

Topology, model and rollout settings live in ``scripts/run_config.json``; this
file only turns them into the Miles command line, starts any local services the
rollout needs, and records the exact launch in ``experiment_logs/<run>/``.

    python3 scripts/train_reproduction.py                 # full run
    python3 scripts/train_reproduction.py --print-command # show the trainer argv
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[1]


def model_args(cfg: dict) -> list[str]:
    """Megatron shape arguments, derived from the HF config of the prepared model."""
    hf = json.loads((WORKSPACE / cfg["model"]["path"] / "config.json").read_text())
    text = hf.get("text_config", hf)
    args = [
        "--spec", "miles_plugins.models.qwen3_5", "get_qwen3_5_spec",
        "--disable-bias-linear", "--qk-layernorm", "--group-query-attention",
        "--num-attention-heads", str(text["num_attention_heads"]),
        "--num-query-groups", str(text["num_key_value_heads"]),
        "--kv-channels", str(text["head_dim"]),
        "--num-layers", str(text["num_hidden_layers"]),
        "--hidden-size", str(text["hidden_size"]),
        "--normalization", "RMSNorm", "--apply-layernorm-1p", "--position-embedding-type", "rope",
        "--norm-epsilon", str(text.get("rms_norm_eps", 1e-6)),
        "--rotary-percent", str(text.get("rope_parameters", text).get("partial_rotary_factor", 0.25)),
        "--rotary-base", str(int(text.get("rope_parameters", text).get("rope_theta", 10_000_000))),
        "--swiglu", "--untie-embeddings-and-output-weights", "--vocab-size", str(text["vocab_size"]),
        "--attention-output-gate",
    ]
    if text.get("num_experts"):
        layers = text["num_hidden_layers"]
        args += [
            "--ffn-hidden-size", str(text["moe_intermediate_size"]),
            "--moe-ffn-hidden-size", str(text["moe_intermediate_size"]),
            "--moe-shared-expert-intermediate-size", str(text["shared_expert_intermediate_size"]),
            "--moe-router-score-function", "softmax", "--moe-router-topk", str(text["num_experts_per_tok"]),
            "--moe-layer-freq", f"[1]*{layers}", "--num-experts", str(text["num_experts"]),
            "--moe-grouped-gemm", "--moe-token-drop-policy", "probs", "--moe-router-dtype", "fp32",
            "--moe-permute-fusion", "--moe-aux-loss-coeff", "0", "--moe-shared-expert-gate",
            "--moe-token-dispatcher-type", "alltoall",
        ]
    else:
        args += ["--ffn-hidden-size", str(text["intermediate_size"])]
    return args


def torch_dist_dir(cfg: dict) -> Path:
    return WORKSPACE / (cfg["model"]["path"] + "_torch_dist")


def convert_checkpoint(cfg: dict, env: dict, log_dir: Path) -> None:
    """One-time HF -> Megatron torch_dist conversion (offline, on the task GPUs)."""
    target = torch_dist_dir(cfg)
    if (target / "latest_checkpointed_iteration.txt").is_file():
        return
    conv = cfg["conversion"]
    argv = [sys.executable, "-m", "torch.distributed.run", "--nproc-per-node", str(conv["gpus"]),
            "scripts/convert_hf_to_torch_dist.py", *model_args(cfg),
            # Shard the conversion by pipeline stage, never by expert: mbridge's loader for the
            # fused 3-D expert tensors of the public checkpoint ignores the expert-parallel offset,
            # so with EP>1 every rank would load experts 0..E/EP-1. Training reshards on load.
            "--tensor-model-parallel-size", "1", "--pipeline-model-parallel-size", str(conv["pp"]),
            "--expert-model-parallel-size", "1", "--expert-tensor-parallel-size", "1",
            "--seq-length", "4096", "--max-position-embeddings", "4096", "--bf16",
            "--rollout-batch-size", "1",  # required by the Miles argument set; unused here
            "--hf-checkpoint", str(WORKSPACE / cfg["model"]["path"]), "--save", str(target)]
    (log_dir / "convert_command.json").write_text(json.dumps(argv, indent=2) + "\n")
    with open(log_dir / "convert.log", "ab") as log:
        rc = subprocess.call(argv, cwd=WORKSPACE, env=env, stdout=log, stderr=subprocess.STDOUT)
    if rc != 0 or not (target / "latest_checkpointed_iteration.txt").is_file():
        raise RuntimeError(f"checkpoint conversion failed (exit {rc}); see {log_dir / 'convert.log'}")


def build_command(cfg: dict) -> list[str]:
    topo, roll, train, router = cfg["topology"], cfg["rollout"], cfg["training"], cfg["router"]
    model = str(WORKSPACE / cfg["model"]["path"])
    argv = [
        sys.executable, "scripts/native_entrypoint.py",
        "--train-backend", "megatron",
        "--actor-num-nodes", "1", "--actor-num-gpus-per-node", str(topo["actor_gpus"]),
        "--rollout-num-gpus", str(topo["rollout_gpus"]),
        "--rollout-num-gpus-per-engine", str(topo["rollout_gpus_per_engine"]),
        "--num-gpus-per-node", str(topo["actor_gpus"] + topo["rollout_gpus"]),
        "--hf-checkpoint", model, "--ref-load", str(torch_dist_dir(cfg)), "--bf16",
        *model_args(cfg),
        "--seq-length", str(roll["max_context_len"]), "--max-position-embeddings", str(roll["max_context_len"]),
        "--tensor-model-parallel-size", str(train["tp"]), "--pipeline-model-parallel-size", "1",
        "--context-parallel-size", "1", "--expert-model-parallel-size", str(train["ep"]),
        "--expert-tensor-parallel-size", "1",
        "--recompute-granularity", "full", "--recompute-method", "uniform", "--recompute-num-layers", "1",
        "--use-dynamic-batch-size", "--max-tokens-per-gpu", str(train["max_tokens_per_gpu"]),
        # 248k-entry vocabulary: never materialize [tokens, vocab] logits
        "--fused-logprob", "--log-probs-max-tokens-per-gpu", str(train["log_probs_max_tokens_per_gpu"]),
        "--advantage-estimator", "grpo", "--entropy-coef", "0", "--eps-clip", "0.2", "--eps-clip-high", "0.28",
        "--optimizer", "adam", "--lr", str(train["lr"]), "--lr-decay-style", "constant",
        "--weight-decay", "0.1", "--adam-beta1", "0.9", "--adam-beta2", "0.98",
        "--attention-dropout", "0.0", "--hidden-dropout", "0.0",
        "--attention-softmax-in-fp32", "--attention-backend", "flash",
        "--rollout-function-path", "runtime.rollout.generate_rollout",
        "--custom-generate-function-path", "runtime.agent_generate.generate",
        # Generate-function options reach the legacy rollout path through Miles'
        # --custom-config-path YAML (written next to the run logs by main()).
        "--custom-config-path", "{custom_config}",
        "--prompt-data", str(WORKSPACE / roll["data"]), "--input-key", "messages", "--label-key", "label",
        "--metadata-key", "metadata", "--rollout-shuffle", "--rollout-seed", "20261116",
        "--num-rollout", str(roll["num_rollout"]), "--rollout-batch-size", str(roll["batch_groups"]),
        "--n-samples-per-prompt", str(roll["samples_per_group"]),
        "--global-batch-size", str(roll["batch_groups"] * roll["samples_per_group"]),
        "--rollout-max-response-len", str(roll["max_new_tokens_per_turn"]),
        "--rollout-max-context-len", str(roll["max_context_len"]),
        "--rollout-temperature", "1.0", "--rollout-top-p", "1.0",
        "--sglang-mem-fraction-static", str(roll["sglang_mem_fraction"]),
        "--sglang-server-concurrency", str(roll["server_concurrency"]),
        "--update-weights-interval", "1", "--pause-generation-mode", "abort",
        "--custom-megatron-before-train-step-hook-path", "runtime.training_evidence.observe_train_step",
        "--no-use-prometheus-scrape",
    ]
    if train.get("fused_logprob_seq_chunk_size"):
        argv += ["--fused-logprob-seq-chunk-size", str(train["fused_logprob_seq_chunk_size"])]
    if train.get("grads_in_fp32", True):
        argv += ["--accumulate-allreduce-grads-in-fp32"]
    if train.get("optimizer_cpu_offload"):
        argv += ["--optimizer-cpu-offload", "--use-precision-aware-optimizer"]
        if train.get("overlap_cpu_optimizer_transfers", False):
            argv += ["--overlap-cpu-optimizer-d2h-h2d"]
    if train["tp"] > 1:
        argv += ["--sequence-parallel"]
    if router.get("use_miles_router"):
        argv += ["--use-miles-router"]
    if router.get("policy"):
        argv += ["--sglang-router-policy", router["policy"]]
    return argv + cfg.get("extra_args", [])


def start_services(cfg: dict, env: dict, log_dir: Path) -> list[subprocess.Popen]:
    procs = []
    for svc in cfg.get("services", []):
        cmd = [sys.executable, *[a.replace("{log_dir}", str(log_dir)) for a in svc["argv"]]]
        out = open(log_dir / f"{svc['name']}.log", "ab")
        procs.append(subprocess.Popen(cmd, cwd=WORKSPACE, env=env, stdout=out, stderr=subprocess.STDOUT))
        url = svc.get("health_url")
        for _ in range(120):
            if url is None:
                break
            try:
                import urllib.request

                # local service: never route the probe through a host proxy
                urllib.request.build_opener(urllib.request.ProxyHandler({})).open(url, timeout=2)
                break
            except Exception:
                time.sleep(0.5)
        else:
            raise RuntimeError(f"service {svc['name']} did not become healthy")
    return procs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--print-command", action="store_true")
    parser.add_argument("--config", type=Path, default=WORKSPACE / "scripts/run_config.json")
    options, extra = parser.parse_known_args()
    cfg = json.loads(options.config.read_text())
    argv = build_command(cfg) + extra
    if options.print_command:
        print(shlex.join(argv))
        return 0
    run_name = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + f"-{os.getpid()}"
    # The harness collects $AUTORLBENCH_RESULTS/experiment_logs; a manual launch logs into the workspace.
    log_root = Path(os.environ["AUTORLBENCH_RESULTS"]) if os.environ.get("AUTORLBENCH_RESULTS") else WORKSPACE
    log_dir = log_root / "experiment_logs" / run_name
    log_dir.mkdir(parents=True, exist_ok=True)
    custom_config = log_dir / "generate_args.yaml"
    custom_config.write_text(json.dumps({
        "generate_max_turns": cfg["rollout"]["max_turns"],
        "generate_tool_call_parser": cfg["rollout"]["tool_call_parser"],
    }, indent=2) + "\n")
    argv = [str(custom_config) if a == "{custom_config}" else a for a in argv]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(WORKSPACE / "vendor"), str(WORKSPACE), env.get("PYTHONPATH", "")])
    env.update({k: str(v) for k, v in cfg.get("env", {}).items()})
    env["FORWARD_ENV"] = ",".join(sorted(cfg.get("env", {})))
    env.update(RUN_EVIDENCE_DIR=str(log_dir), PYTHONDONTWRITEBYTECODE="1", NO_VCS_VERSION="1",
               TOKENIZERS_PARALLELISM="false", CUDA_DEVICE_MAX_CONNECTIONS="1", WANDB_MODE="disabled",
               PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    (log_dir / "launch.json").write_text(json.dumps(
        {"argv": argv, "config": cfg, "env": {k: env[k] for k in sorted(cfg.get("env", {}))}}, indent=2) + "\n")
    convert_checkpoint(cfg, env, log_dir)
    procs = start_services(cfg, env, log_dir)
    try:
        with open(log_dir / "train.log", "ab") as log:
            trainer = subprocess.Popen(argv, cwd=WORKSPACE, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            for line in trainer.stdout:
                sys.stdout.buffer.write(line)
                sys.stdout.flush()
                log.write(line)
            return trainer.wait()
    finally:
        for proc in procs:
            proc.send_signal(signal.SIGTERM)
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                proc.kill()


if __name__ == "__main__":
    raise SystemExit(main())
