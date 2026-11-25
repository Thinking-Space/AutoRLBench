"""Training recipe for this task's experiment.

Same stack as the stock Qwen3-0.6B FSDP recipe (GRPO on dapo-math, colocated
single node), but every rollout request is driven through the white-box
gateway by a custom generate function, and the router runs a routing-key
policy. All paths are resolved relative to this file, so the recipe runs
from any staging location.
"""

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

import typer

from miles.utils.external_utils import command_utils

TASK_ROOT = Path(__file__).resolve().parent.parent
MODEL_NAME = "Qwen3-0.6B"


@dataclass
class ScriptArgs(command_utils.ExecuteTrainConfig):
    num_gpus_per_node: int = 2
    num_rollout: int = 10
    rollout_batch_size: int = 8
    n_samples_per_prompt: int = 4
    rollout_max_response_len: int = 2048
    extra_args: str = ""


def execute(args: ScriptArgs):
    U = args.create_backend()
    model_path = str(TASK_ROOT / "models" / MODEL_NAME)
    data_path = str(TASK_ROOT / "data" / "dapo_math_17k_dedup.jsonl")

    ckpt_args = f"--hf-checkpoint {model_path} " f"--ref-load {model_path} "

    rollout_args = (
        f"--prompt-data {data_path} "
        "--input-key prompt "
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--balance-data "
        "--rm-type deepscaler "
        f"--num-rollout {args.num_rollout} "
        f"--rollout-batch-size {args.rollout_batch_size} "
        f"--n-samples-per-prompt {args.n_samples_per_prompt} "
        f"--rollout-max-response-len {args.rollout_max_response_len} "
        "--rollout-temperature 1 "
        f"--global-batch-size {args.rollout_batch_size * args.n_samples_per_prompt} "
        "--sglang-router-policy consistent_hashing "
        "--custom-generate-function-path rollout.gateway_generate.generate "
    )

    grpo_args = (
        "--use-kl-loss "
        "--advantage-estimator grpo "
        "--kl-loss-coef 0.00 "
        "--kl-loss-type low_var_kl "
        "--kl-coef 0.00 "
        "--entropy-coef 0.00 "
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
        "--rollout-num-gpus-per-engine 1 "
        "--sglang-decode-log-interval 1000 "
        "--sglang-mem-fraction-static 0.75 "
        "--sglang-attention-backend fa3 "
        "--sglang-chunked-prefill-size 4096 "
    )

    # The pod's egress goes through a whitelist proxy that refuses pod IPs,
    # and the training addresses engines and gates by pod IP, so the ray
    # workers need them exempted from the proxy (appended to any inherited
    # exemptions, never replacing them).
    pod_ip = subprocess.check_output(["hostname", "-I"], text=True).split()[0]
    inherited = os.environ.get("no_proxy") or os.environ.get("NO_PROXY") or ""
    local_exempt = f"127.0.0.1,localhost,{pod_ip}"
    proxy_exempt = f"{local_exempt},{inherited}" if inherited else local_exempt
    env_vars_json = (
        "{"
        '"PYTORCH_CUDA_ALLOC_CONF":"expandable_segments:True",'
        f'"no_proxy":"{proxy_exempt}",'
        f'"NO_PROXY":"{proxy_exempt}"'
        "}"
    )

    train_backend_args = (
        "--train-backend fsdp "
        "--update-weight-buffer-size 536870912 "
        "--gradient-checkpointing "
        "--attn-implementation sdpa "
        f"--train-env-vars '{env_vars_json}' "
    )

    perf_args = "--use-dynamic-batch-size --max-tokens-per-gpu 9216 "

    misc_args = (
        f"--actor-num-nodes {args.num_nodes} "
        f"--actor-num-gpus-per-node {args.num_gpus_per_node} "
        "--colocate "
    )

    U.execute_train(
        train_args=(
            f"{ckpt_args} "
            f"{rollout_args} "
            f"{grpo_args} "
            f"{optimizer_args} "
            f"{sglang_args} "
            f"{train_backend_args} "
            f"{perf_args} "
            f"{misc_args} "
            f"{args.extra_args} "
        ),
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type=None,
        # env for the whole ray job (driver and actors): PYTHONPATH must carry
        # the workspace's miles tree ahead of the image's editable install,
        # plus the proxy exemptions.
        extra_env_vars={
            "no_proxy": proxy_exempt,
            "NO_PROXY": proxy_exempt,
            "PYTHONPATH": os.environ.get("PYTHONPATH", ""),
        },
    )


@command_utils.dataclass_cli
def main(args: ScriptArgs):
    execute(args)


if __name__ == "__main__":
    typer.run(main)
