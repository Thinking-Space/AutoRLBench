"""Compose the original QAT CLI in the official image and check the active entry point."""
import argparse
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import subprocess
import sys

import yaml


ENTRY = "verl.trainer.main_ppo"
RUNS = {
    26018: dict(gpus=2, steps=3, student_score_source="hf_or_rollout", block_align_coef=0.0, lr=0),
    9073: dict(gpus=4, steps=150, student_score_source="vllm_native_turboquant", block_align_coef=10.0, lr=1.5e-6),
    7277: dict(gpus=4, steps=2, student_score_source="vllm_native_turboquant", block_align_coef=0.5,
               lr=1e-6, prompt_limit=2048, top_k=64, candidate_set="only_stu", actor_microbatch_per_gpu=4),
}
HOST_ENV = {"PATH", "CPATH", "C_INCLUDE_PATH", "CPLUS_INCLUDE_PATH", "PYTHONPATH"}
HOST_PREFIXES = ("/historical/shared/", "/historical/home/")


def pinned_manifest(path, checksum):
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != checksum:
        raise ValueError("The recorded Hydra command does not match its source hash")
    return json.loads(raw)


def portable_command(manifest, source, assets, output, model=None):
    original = manifest["command"]
    index = original.index(ENTRY)
    if original[0] != "env" or original[index - 1] != "-m":
        raise ValueError("Expected the recorded env/Python/VeRL launch sequence")
    captured = dict(value.split("=", 1) for value in original[1:index - 2])
    model = model if model is not None else assets / "nemotron-brorl"
    dataset = assets / "skywork_or1_7b_math_difficulty1-15_16k.parquet"
    verification = output / "rollout_kv_verification.json"
    new_args = {
        "+ray_kwargs.ray_init._temp_dir": str(output / "ray"),
        "actor_rollout_ref.model.path": str(model),
        "reward_model.model.path": str(model),
        "data.train_files": str(dataset),
        "data.val_files": str(dataset),
        "trainer.default_local_dir": str(output / "checkpoints"),
        "trainer.rollout_data_dir": str(output / "rollouts"),
        "trainer.validation_data_dir": str(output / "validation"),
        "+mopd.rollout_kv_verification_record": str(verification),
    }
    overrides = []
    replacements = []
    for argument in original[index + 1:]:
        key, value = argument.split("=", 1)
        if key.startswith("+ray_kwargs.ray_init.runtime_env.env_vars.") and key.rsplit(".", 1)[-1] in HOST_ENV:
            replacements.append(key)
            continue
        if key in new_args:
            value = new_args[key]
            replacements.append(key)
        if any(prefix in value for prefix in HOST_PREFIXES):
            raise ValueError(f"Historical host path in Hydra argument {key}")
        overrides.append(f"{key}={value}")
    environment = os.environ.copy()
    for key in HOST_ENV | {"MOPD_ROLLOUT_KV_VERIFICATION_RECORD", "SWANLAB_LOG_DIR"}:
        captured.pop(key, None)
    for key, value in captured.items():
        if any(prefix in value for prefix in HOST_PREFIXES):
            raise ValueError(f"Historical host path in environment setting {key}")
    environment.update(captured)
    environment.update(
        PYTHONPATH=os.pathsep.join((str(source / "vendor/verl"), str(source))),
        CUDA_VISIBLE_DEVICES="",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        PYTHONDONTWRITEBYTECODE="1",
        SWANLAB_MODE="disabled",
        SWANLAB_LOG_DIR=str(output / "swanlog"),
        MOPD_ROLLOUT_KV_VERIFICATION_RECORD=str(verification),
    )
    return [sys.executable, "-B", "-m", ENTRY, "--cfg", "job", *overrides], environment, replacements


def scientific_config(config):
    actor = config["actor_rollout_ref"]["actor"]
    rollout = config["actor_rollout_ref"]["rollout"]
    return {
        "prompts": config["data"]["train_batch_size"],
        "prompt_limit": config["data"]["max_prompt_length"],
        "response_limit": config["data"]["max_response_length"],
        "responses_per_prompt": rollout["n"],
        "rollout_response_limit": rollout.get("response_length"),
        "rollout_context_limit": rollout.get("max_model_len"),
        "advantage": config["algorithm"]["adv_estimator"],
        "top_k": rollout["log_prob_top_k"],
        "candidate_set": rollout["top_k_strategy"],
        "teacher": config["reward_model"]["enable"],
        "student_score_source": config["mopd"]["rollout_student_logprob_source"],
        "block_align_coef": actor["block_align_coef"],
        "actor_microbatch_per_gpu": actor["ppo_micro_batch_size_per_gpu"],
        "actor_minibatch": actor["ppo_mini_batch_size"],
        "actor_warmup_steps": actor["optim"].get("lr_warmup_steps"),
        "actor_lr_scheduler": actor["optim"].get("lr_scheduler_type"),
        "lr": actor["optim"]["lr"],
        "weight_decay": actor["optim"]["weight_decay"],
        "kv_cache_dtype": rollout["engine_kwargs"].get("vllm", {}).get("kv_cache_dtype"),
        "free_cache_engine": rollout["free_cache_engine"],
        "steps": config["trainer"]["total_training_steps"],
        "gpus": config["trainer"]["n_gpus_per_node"],
    }


def reward_two_gpu_command(command):
    # codex: Preserve 16 responses per rank and one optimizer minibatch per rollout.
    updates = {
        "data.max_response_length": 8192,
        "data.train_batch_size": 32,
        "actor_rollout_ref.actor.ppo_mini_batch_size": 32,
        "actor_rollout_ref.rollout.response_length": 8192,
        "actor_rollout_ref.rollout.max_model_len": 8704,
        "trainer.n_gpus_per_node": 2,
        "+mopd.global_batch": 32,
        "+mopd.steps_per_epoch": 500,
        "+mopd.planned_dataset_passes": "0.300000",
    }
    seen = set()
    result = []
    for item in command:
        key = item.split("=", 1)[0]
        if key in updates:
            seen.add(key)
            item = f"{key}={updates[key]}"
        result.append(item)
    if seen != updates.keys():
        raise ValueError(f"Missing original Reward settings: {updates.keys() - seen}")
    return result


def native_launch(command, environment, dependencies, nvfp4, k2v3_overlay, qualification_steps):
    # codex: Preserve the composed training CLI and propagate only worker-local runtime paths.
    command = command[:4] + command[6:]
    worker_environment = {"VLLM_ALLOW_INSECURE_SERIALIZATION": "1"}
    if dependencies:
        environment["PYTHONPATH"] += os.pathsep + str(dependencies)
    if k2v3_overlay:
        environment["PYTHONPATH"] = os.pathsep.join((str(k2v3_overlay), environment["PYTHONPATH"]))
    worker_environment["PYTHONPATH"] = environment["PYTHONPATH"]
    if nvfp4:
        # codex: The archived re-sync code writes FlashInfer CUTLASS packed tensors.
        command.append("+actor_rollout_ref.rollout.engine_kwargs.vllm.linear_backend=flashinfer_cutlass")
        worker_environment.update(
            QAT_NVFP4_ROLLOUT="1",
            QAT_NVFP4_ROLLOUT_MODEL=str(nvfp4),
            QAT_WA_SCHEME="nvfp4",
            QAT_WA_STATE=str(nvfp4 / "modelopt_state.pt"),
        )
    for name, value in worker_environment.items():
        command.append(f"+ray_kwargs.ray_init.runtime_env.env_vars.{name}={json.dumps(value)}")
    environment.update(worker_environment)
    environment.pop("CUDA_VISIBLE_DEVICES")
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        environment["CUDA_VISIBLE_DEVICES"] = os.environ["CUDA_VISIBLE_DEVICES"]

    # The old cluster's SwanLab service is outside the offline reproduction.
    command = [value for value in command if not value.startswith("trainer.logger=")]
    command.append("trainer.logger=[console]")
    if qualification_steps:
        # codex: End the native trial after this many updates, leaving the original checkpoint interval intact.
        stop = "trainer.total_training_steps="
        command = [
            f"{stop}{qualification_steps}" if value.startswith(stop) else value for value in command
        ]
    return command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job", type=int, choices=RUNS)
    for argument in ("source", "assets", "recorded", "reference", "output"):
        parser.add_argument(f"--{argument}", required=True, type=Path)
    parser.add_argument("--recorded-sha256", required=True)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--dependencies", type=Path)
    parser.add_argument("--nvfp4", type=Path)
    parser.add_argument("--k2v3-overlay", type=Path)
    parser.add_argument("--reward-qualification-steps", type=int, choices=range(1, 31), metavar="1..30")
    parser.add_argument("--reward-two-gpu", action="store_true")
    parser.add_argument("--reward-resume", type=Path)
    args = parser.parse_args()
    if metadata.version("vllm") != "0.24.0":
        raise RuntimeError("Expected the official vLLM 0.24.0 runtime")
    reference = scientific_config(yaml.safe_load(args.reference.read_bytes()))
    required = dict(prompts=64, prompt_limit=512, response_limit=16384, responses_per_prompt=1,
                    advantage="token_reward_direct", top_k=512, candidate_set="union", teacher=True,
                    actor_microbatch_per_gpu=2, free_cache_engine=True)
    required.update(RUNS[args.job])
    if any(reference[key] != value for key, value in required.items()):
        raise ValueError("The original resolved reference does not match the verified QAT run")
    reward_trial = args.job == 9073 and args.run
    if args.reward_two_gpu and not reward_trial:
        raise ValueError("The reduced training workload applies only to the native Reward trial")
    if any(bool(value) != reward_trial for value in (args.k2v3_overlay, args.reward_qualification_steps)):
        raise ValueError("Reward training requires the pinned K2V3 overlay and an explicit qualification horizon")
    if reward_trial and not (args.k2v3_overlay / "vllm" / "__init__.py").is_file():
        raise ValueError("The K2V3 overlay has not been staged")
    if args.run and (args.job == 26018) != bool(args.nvfp4):
        raise ValueError("Only the Cache training recipe takes the reconstructed NVFP4 assets")
    if args.run and args.nvfp4 and not all(
        (args.nvfp4 / name).is_file() for name in ("model.safetensors", "modelopt_state.pt")
    ):
        raise ValueError("The historical checkpoint reconstruction is incomplete")
    resume = args.reward_resume
    if resume:
        # codex: Verify the native model, optimizer, RNG and dataloader state before resuming.
        prior = args.reward_qualification_steps - 8
        if not (reward_trial and args.reward_two_gpu and prior in (8, 16)):
            raise ValueError("Expected the two-GPU continuation at CP8 or CP16")
        resume = resume.resolve(strict=True)
        marker = resume.parent / "latest_checkpointed_iteration.txt"
        if resume.name != f"global_step_{prior}" or marker.read_text().strip() != str(prior):
            raise ValueError("Unexpected checkpoint or latest iteration")
        files = [resume / "data.pt"] + [
            resume / "actor" / f"{part}_world_size_2_rank_{rank}.pt"
            for part in ("model", "optim", "extra_state") for rank in (0, 1)
        ]
        if any(not path.is_file() or path.stat().st_size == 0 for path in files):
            raise ValueError("Checkpoint is missing native training state")
    args.output.mkdir(parents=True, exist_ok=False)
    command, env, remaps = portable_command(
        pinned_manifest(args.recorded, args.recorded_sha256), args.source, args.assets, args.output, args.model
    )
    result = subprocess.run(command, cwd=args.source, env=env, capture_output=True, check=False)
    (args.output / "compose.stdout").write_bytes(result.stdout)
    (args.output / "compose.stderr").write_bytes(result.stderr)
    result.check_returncode()
    parsed = yaml.safe_load(result.stdout)
    portable = scientific_config(parsed)
    if portable != reference:
        raise ValueError("The portable CLI changed the original scientific configuration")
    receipt = {
        "historical_job": args.job, "scientific_config": portable, "host_path_overrides": remaps,
        "package_versions": {name: metadata.version(name) for name in ("vllm", "torch", "ray", "transformers")},
        "command_source": "captured original effective-settings argv", "cuda_visible_devices": "",
        "resolved_config_sha256": hashlib.sha256(result.stdout).hexdigest(),
    }
    (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt))
    if args.run:
        command = native_launch(
            command, env, args.dependencies, args.nvfp4, args.k2v3_overlay, args.reward_qualification_steps
        )
        if args.reward_two_gpu:
            command = reward_two_gpu_command(command)
        if resume:
            overrides = {"trainer.resume_mode": "resume_path", "trainer.resume_from_path": str(resume),
                         "trainer.del_local_ckpt_after_load": "False"}
            command = [item for item in command if item.partition("=")[0] not in overrides]
            command.extend(f"{key}={value}" for key, value in overrides.items())
        # Ray opens Unix-domain sockets whose names cannot contain the full result path.
        command = [
            "+ray_kwargs.ray_init._temp_dir=/tmp/qat-ray"
            if value.startswith("+ray_kwargs.ray_init._temp_dir=") else value
            for value in command
        ]
        native_config = subprocess.run(
            [*command[:4], "--cfg", "job", *command[4:]],
            cwd=args.source, env=dict(env, CUDA_VISIBLE_DEVICES=""), capture_output=True, check=True,
        )
        expected = dict(reference)
        if reward_trial:
            expected["steps"] = args.reward_qualification_steps
        if args.reward_two_gpu:
            expected.update(gpus=2, prompts=32, actor_minibatch=32, response_limit=8192,
                            rollout_response_limit=8192, rollout_context_limit=8704)
        settings = yaml.safe_load(native_config.stdout)
        effective = scientific_config(settings)
        if resume:
            trainer = settings["trainer"]
            if (trainer["resume_mode"], trainer["resume_from_path"], trainer["del_local_ckpt_after_load"]) != (
                    "resume_path", str(resume), False):
                raise ValueError("The native trainer did not receive the resume configuration")
        if effective != expected:
            raise ValueError("The training launch changed the historical scientific configuration")
        (args.output / "native-config.yaml").write_bytes(native_config.stdout)
        (args.output / "native-receipt.json").write_text(json.dumps({
            "historical_job": args.job,
            "scientific_config": effective,
            "resume_checkpoint": str(resume) if resume else None,
            "resolved_config_sha256": hashlib.sha256(native_config.stdout).hexdigest(),
        }, indent=2) + "\n", encoding="utf-8")
        expected_gpus = expected["gpus"]
        cuda_check = f"import torch; assert torch.cuda.device_count() == {expected_gpus}"
        if args.nvfp4:
            cuda_check += (
                "; from vllm.engine.arg_utils import EngineArgs"
                "; assert 'linear_backend' in EngineArgs.__dataclass_fields__"
                "; from vllm.config import KernelConfig"
                "; assert KernelConfig(linear_backend='flashinfer_cutlass').linear_backend == 'flashinfer_cutlass'"
            )
        subprocess.run([sys.executable, "-B", "-c", cuda_check], cwd=args.source, env=env, check=True)
        with (args.output / "train.stdout").open("xb") as out, (args.output / "train.stderr").open("xb") as err:
            subprocess.run(command, cwd=args.source, env=env, stdout=out, stderr=err, check=True)


if __name__ == "__main__":
    main()
