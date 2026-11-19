#!/usr/bin/env python3
"""Launch the Miles + Harbor + mini-swe-agent GRPO observation run."""

from __future__ import annotations

import os
import shlex
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import typer

from launch_common import agentic_pythonpath_dirs, harbor_env_vars
from miles.utils.external_utils import command_utils


@dataclass
class ScriptArgs(command_utils.ExecuteTrainConfig):
    mode: Literal["normal", "debug_rollout_only"] = "normal"
    run_id: str = command_utils.create_run_id()
    megatron_model_type: str = "qwen3-30B-A3B"
    num_gpus_per_node: int = 8
    megatron_path: str = os.environ.get("MEGATRON_PATH", "/root/Megatron-LM")

    skip_prepare: bool = False
    base_dir: str = os.environ.get("MILES_MODEL_BASE", "/tmp/models")
    model_name: str = os.environ.get("MODEL_NAME", "Qwen3-30B-A3B-autorlbench")
    hf_checkpoint: str = os.environ.get("MODEL_PATH", "")
    ref_load: str = os.environ.get("MILES_REF_LOAD", "")
    prompt_data: str = os.environ.get(
        "MILES_PROMPT_DATA",
        str(Path(__file__).with_name("data") / "miles_prompts.jsonl"),
    )
    save_traces_dir: str = os.environ.get("MILES_DUMP_DIR", "")

    max_seq_len: int = 32768
    num_rollout: int = 1
    rollout_batch_size: int = int(os.environ.get("ROLLOUT_BATCH_SIZE", "16"))
    n_samples_per_prompt: int = int(os.environ.get("N_SAMPLES_PER_PROMPT", "4"))
    global_batch_size: int = int(os.environ.get("GLOBAL_BATCH_SIZE", "64"))

    harbor_env_type: str = os.environ.get("HARBOR_ENV_TYPE", "")
    harbor_env_kwargs: str = os.environ.get("HARBOR_ENV_KWARGS", "{}")
    sandbox_backend: str = os.environ.get("SANDBOX_BACKEND", "tencent_ags")
    harbor_tasks_dir: str = os.environ.get("HARBOR_TASKS_DIR", "")
    harbor_trials_dir: str = os.environ.get("HARBOR_TRIALS_DIR", "/tmp/harbor_trials")
    agent_model_name: str = os.environ.get("AGENT_MODEL_NAME", "model")
    agent_timeout: int = int(os.environ.get("AGENT_TIMEOUT", "5400"))
    daytona_api_key_file: str = os.environ.get("DAYTONA_API_KEY_FILE", "")
    e2b_api_key_file: str = os.environ.get("E2B_API_KEY_FILE", "")
    modal_config_file: str = os.environ.get("MODAL_CONFIG_PATH", "")
    novita_api_key_file: str = os.environ.get("NOVITA_API_KEY_FILE", "")

    session_server_external_host: str = os.environ.get("SESSION_SERVER_EXTERNAL_HOST", "")
    session_server_workers: int = int(os.environ.get("SESSION_SERVER_WORKERS", "1"))
    miles_host_ip: str = os.environ.get("MILES_HOST_IP", "")

    def __post_init__(self) -> None:
        if not self.hf_checkpoint:
            raise ValueError("set MODEL_PATH or --hf-checkpoint")
        if not self.harbor_tasks_dir:
            self.harbor_tasks_dir = str(Path(__file__).with_name("harbor_tasks"))
        if not self.ref_load:
            self.ref_load = f"{self.base_dir.rstrip('/')}/{self.model_name}_torch_dist"
        if not self.save_traces_dir:
            self.save_traces_dir = os.environ.get(
                "MILES_DUMP_DIR", "/tmp/autorlbench-results/dump_details"
            )


def cleanup() -> None:
    my_pid = os.getpid()
    parent_pid = os.getppid()
    exclude = f"grep -v '^{my_pid}$' | grep -v '^{parent_pid}$'"
    for process_name in ("sglang", "train.py", "MegatronTrain"):
        subprocess.run(
            f"pgrep -f '{process_name}' | {exclude} | xargs -r kill 2>/dev/null || true",
            shell=True,
            check=False,
        )
    time.sleep(5)


def prepare(args: ScriptArgs) -> None:
    backend = args.create_backend()
    backend.convert_checkpoint(
        model_name=args.model_name,
        megatron_model_type=args.megatron_model_type,
        num_gpus_per_node=args.num_gpus_per_node,
        dir_dst=args.base_dir,
        hf_checkpoint=args.hf_checkpoint,
        megatron_path=args.megatron_path,
        extra_args="--no-gradient-accumulation-fusion",
    )


def agent_train_args(args: ScriptArgs) -> str:
    external_host = ""
    if args.session_server_external_host:
        external_host = (
            f"--session-server-external-host {shlex.quote(args.session_server_external_host)} "
        )
    if args.sandbox_backend == "tencent_ags":
        agent_function = "tencent_harbor_agent.run"
    elif args.sandbox_backend == "e2b":
        agent_function = "harbor_agent_function.run"
    else:
        raise ValueError(
            f"unsupported SANDBOX_BACKEND={args.sandbox_backend!r}; use tencent_ags or e2b"
        )
    return (
        "--custom-generate-function-path miles.rollout.generate_hub.agentic_tool_call.generate "
        f"--custom-agent-function-path {agent_function} "
        "--custom-rm-path generate.reward_func "
        "--rollout-function-path generate.RolloutFn "
        "--dynamic-sampling-filter-path "
        "miles.rollout.filter_hub.dynamic_sampling_filters.check_no_aborted "
        "--tito-model qwen3 "
        "--use-session-server v2 "
        "--session-server-port 30000 "
        f"--session-server-workers {args.session_server_workers} "
        "--session-message-matcher strict "
        f"{external_host}"
    )


def execute(args: ScriptArgs) -> None:
    backend = args.create_backend()
    checkpoint_args = (
        f"--hf-checkpoint {shlex.quote(args.hf_checkpoint)} "
        f"--ref-load {shlex.quote(args.ref_load)} "
    )
    rollout_args = (
        f"--prompt-data {shlex.quote(args.prompt_data)} "
        "--input-key prompt "
        "--metadata-key metadata "
        "--rollout-shuffle "
        f"--num-rollout {args.num_rollout} "
        f"--rollout-batch-size {args.rollout_batch_size} "
        f"--n-samples-per-prompt {args.n_samples_per_prompt} "
        "--rollout-temperature 1.0 "
        "--rollout-max-response-len 4096 "
        f"--max-seq-len {args.max_seq_len} "
        f"--global-batch-size {args.global_batch_size} "
        "--balance-data "
    )
    performance_args = (
        "--tensor-model-parallel-size 4 "
        "--sequence-parallel "
        "--pipeline-model-parallel-size 1 "
        "--context-parallel-size 1 "
        "--expert-model-parallel-size 8 "
        "--expert-tensor-parallel-size 1 "
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
        "--no-gradient-accumulation-fusion "
        "--use-dynamic-batch-size "
        "--max-tokens-per-gpu 16384 "
        "--optimizer-cpu-offload "
        "--overlap-cpu-optimizer-d2h-h2d "
        "--use-precision-aware-optimizer "
    )
    grpo_args = (
        "--advantage-estimator grpo "
        "--use-kl-loss "
        "--kl-loss-coef 0.01 "
        "--kl-loss-type low_var_kl "
        "--entropy-coef 0.0 "
        "--eps-clip 0.2 "
        "--eps-clip-high 0.28 "
    )
    optimizer_args = (
        "--optimizer adam "
        "--lr 1e-6 "
        "--lr-decay-style constant "
        "--weight-decay 0.1 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.98 "
    )
    sglang_args = (
        "--rollout-num-gpus-per-engine 8 "
        "--sglang-mem-fraction-static 0.7 "
        "--sglang-cuda-graph-max-bs-decode 512 "
        "--sglang-tool-call-parser qwen3_coder "
        "--sglang-reasoning-parser qwen3 "
        "--sglang-router-port 31000 "
    )
    parallel_args = "--eval-global-batch-size 2 "
    misc_args = (
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        "--attention-backend flash "
        "--colocate "
        f"--actor-num-nodes {args.num_nodes} "
        f"--actor-num-gpus-per-node {args.num_gpus_per_node} "
        f"--rollout-num-gpus {args.num_gpus_per_node} "
    )
    debug_args = "--debug-rollout-only " if args.mode == "debug_rollout_only" else ""
    trace_args = f"--dump-details {shlex.quote(args.save_traces_dir)} "

    train_args = "".join(
        (
            checkpoint_args,
            rollout_args,
            optimizer_args,
            grpo_args,
            trace_args,
            performance_args,
            parallel_args,
            sglang_args,
            agent_train_args(args),
            misc_args,
            debug_args,
        )
    )

    extra_env_vars = {
        "PYTHONPATH": ":".join(
            [
                str(Path(__file__).resolve().parent),
                args.megatron_path,
                *agentic_pythonpath_dirs(),
                str(command_utils.repo_base_dir),
            ]
        ),
        **harbor_env_vars(args),
        "no_proxy": "127.0.0.1,localhost,0.0.0.0",
        "NO_PROXY": "127.0.0.1,localhost,0.0.0.0",
    }
    if args.miles_host_ip:
        extra_env_vars["MILES_HOST_IP"] = args.miles_host_ip
    for endpoint_var in ("E2B_API_URL", "E2B_SANDBOX_URL"):
        if os.environ.get(endpoint_var, "").strip():
            extra_env_vars[endpoint_var] = os.environ[endpoint_var]
    for runtime_var in (
        "SESSION_SERVER_PUBLIC_URL",
        "E2B_DOMAIN",
        "E2B_PROXY",
        "TENCENT_AGS_TEMPLATE",
        "TENCENT_AGS_IMAGE_REWRITE",
        "TENCENT_AGS_IMAGE_REGISTRY_TYPE",
        "TENCENT_AGS_ENVD_PORT",
        "TENCENT_AGS_FORCE_HTTP1",
        "TENCENT_AGS_REQUEST_TIMEOUT",
        "TENCENT_AGS_CREATE_RETRIES",
        "TENCENT_AGS_AGENT_WHEELHOUSE_TAR",
        "TENCENT_AGS_AGENT_INSTALL_TIMEOUT",
        "TENCENT_AGS_UPLOAD_CHUNK_BYTES",
    ):
        if value := os.environ.get(runtime_var, "").strip():
            extra_env_vars[runtime_var] = value

    backend.execute_train(
        train_args=train_args,
        config=args,
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type=args.megatron_model_type,
        megatron_path=args.megatron_path,
        extra_env_vars=extra_env_vars,
    )


@command_utils.dataclass_cli
def main(args: ScriptArgs) -> None:
    cleanup()
    if not args.skip_prepare:
        prepare(args)
    execute(args)


if __name__ == "__main__":
    typer.run(main)
