"""Launch the recorded native QAT math RL run."""

import argparse
import importlib.util
import json
import subprocess
from pathlib import Path

import yaml


DATASET = "skywork_or1_7b_math_difficulty1-15_16k.parquet"


def load_historical_launcher(workspace):
    spec = importlib.util.spec_from_file_location(
        "qat_launch", workspace / "qat_historical_reward_two_gpu_launch.py"
    )
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    return launcher


def make_rl_command(command, reward_path, response_limit=8192, historical=False):
    changes = {
        "algorithm.adv_estimator": "reinforce_plus_plus",
        "reward_model.enable": "false",
        "reward_model.reward_kwargs.compute_true_reward": "true",
        "actor_rollout_ref.actor.block_align_coef": "0.0",
        "actor_rollout_ref.rollout.engine_kwargs.vllm.max_logprobs": "0",
        "actor_rollout_ref.rollout.logprobs": "0",
        "actor_rollout_ref.rollout.log_prob_top_k": "0",
        "ray_kwargs.ray_init._temp_dir": "/tmp/autorl-math-rl",
        "trainer.experiment_name": "nemotron_skywork-turboquant_k2v3-ba10.0-math-rl",
        "data.max_response_length": str(response_limit),
        "actor_rollout_ref.rollout.response_length": str(response_limit),
        "actor_rollout_ref.rollout.max_model_len": str(response_limit + 512),
    }
    if historical:
        del changes["actor_rollout_ref.actor.block_align_coef"]
    source = {arg.partition("=")[0].lstrip("+"): arg for arg in command[4:]}
    if not changes.keys() <= source.keys():
        raise ValueError(f"Missing historical overrides: {sorted(changes.keys() - source.keys())}")
    overrides = []
    for original in command[4:]:
        key = original.partition("=")[0]
        if key.startswith("+mopd.") or key == "+actor_rollout_ref.rollout.reward_mode":
            continue
        name = key.lstrip("+")
        overrides.append(f"{key}={changes[name]}" if name in changes else original)
    rollout_dir = source["trainer.rollout_data_dir"].partition("=")[2]
    return [
        *command[:4], *overrides, "+actor_rollout_ref.actor.use_rollout_log_probs=false",
        f"custom_reward_function.path={json.dumps(str(reward_path))}",
        "custom_reward_function.name=compute_score",
        "trainer.rollout_dump.enable=true",
        f"trainer.rollout_dump.dir={rollout_dir}",
        "trainer.rollout_dump.fail_on_error=true",
    ]


def check_config(config, steps, response_limit=8192, historical=False):
    actor = config["actor_rollout_ref"]["actor"]
    rollout = config["actor_rollout_ref"]["rollout"]
    algorithm = config["algorithm"]
    trainer = config["trainer"]
    data = config["data"]
    prompts, gpus, block_align = (64, 4, 10.0) if historical else (32, 2, 0.0)
    conditions = {
        "answer-verifiable RL": (
            algorithm["adv_estimator"] == "reinforce_plus_plus" and not config["reward_model"]["enable"]
            and config["reward_model"]["reward_kwargs"]["compute_true_reward"]
            and actor["block_align_coef"] == block_align and not actor["use_kl_loss"] and not algorithm["use_kl_in_reward"]
        ),
        "policy loss configuration": (
            algorithm["rollout_correction"]["bypass_old_logprob_for_rollout"]
            and not algorithm["rollout_correction"].get("use_pure_rollout_correction", False)
            and actor["use_rollout_log_probs"] is False
            and actor["policy_loss"]["loss_mode"] == "vanilla"
        ),
        "optimization batch and epochs": (
            data["train_batch_size"] == actor["ppo_mini_batch_size"] == prompts
            and rollout["n"] == actor["ppo_epochs"] == 1
        ),
        "native sampled logprobs": (
            rollout["name"] == "vllm" and rollout["mode"] == "sync" and rollout["calculate_log_probs"]
            and rollout["logprobs"] == rollout["log_prob_top_k"] == 0
            and rollout["engine_kwargs"]["vllm"]["kv_cache_dtype"] == "turboquant_k2v3_nc"
        ),
        "same science budget": (
            data["max_response_length"] == rollout["response_length"] == response_limit
            and data["max_prompt_length"] == rollout["prompt_length"] == 512
            and rollout["max_model_len"] == response_limit + 512 and trainer["n_gpus_per_node"] == gpus
            and actor["ppo_micro_batch_size_per_gpu"] == 2
            and trainer["total_training_steps"] == steps and trainer["save_freq"] == 30
        ),
    }
    failed = [name for name, passed in conditions.items() if not passed]
    if failed:
        raise ValueError(f"Composed configuration failed: {', '.join(failed)}")


def resume_command(command, checkpoint, steps):
    if checkpoint.name != "global_step_8" or steps <= 8:
        raise ValueError("Expected a continuation beyond the original-workload CP8")
    lineage = checkpoint.parent
    if (lineage / "latest_checkpointed_iteration.txt").read_text().strip() != "8":
        raise ValueError("The original-workload CP8 was not saved successfully")
    run = lineage.parent
    if (run.parent / "exit_code").read_text().strip() != "0":
        raise ValueError("The parent training did not exit successfully")
    parent = yaml.safe_load((run / "native-config.yaml").read_bytes())
    check_config(parent, 8, 16384, historical=True)
    files = [checkpoint / "data.pt"] + [
        checkpoint / "actor" / f"{part}_world_size_4_rank_{rank}.pt"
        for rank in range(4) for part in ("model", "optim", "extra_state")
    ]
    if any(not path.is_file() or path.stat().st_size == 0 for path in files):
        raise ValueError("Incomplete native model, optimizer, scheduler, RNG or cursor state")
    settings = {
        "trainer.resume_mode": "resume_path",
        "trainer.resume_from_path": json.dumps(str(checkpoint)),
        "trainer.del_local_ckpt_after_load": "false",
    }
    return [
        *(item for item in command if item.partition("=")[0] not in settings),
        *(f"{key}={value}" for key, value in settings.items()),
    ]


def check_resume_fidelity(parent, continuation):
    def flattened(config):
        scratch = str(Path(config["actor_rollout_ref"]["model"]["path"]).parent.parent)
        output = str(Path(config["trainer"]["default_local_dir"]).parent)
        reward = config["custom_reward_function"]["path"]

        def visit(value, key):
            if isinstance(value, dict):
                return {field: entry for name, child in value.items()
                        for field, entry in visit(child, f"{key}{name}.").items()}
            if isinstance(value, str):
                value = value.replace(scratch + "/", "<scratch>/").replace(output + "/", "<output>/")
                if key == "custom_reward_function.path." and value == reward:
                    value = Path(value).name
            return {key.removesuffix("."): value}

        return visit(config, "")

    previous, current = flattened(parent), flattened(continuation)
    restore_fields = {
        "trainer.total_training_steps", "trainer.resume_mode", "trainer.resume_from_path",
        "trainer.del_local_ckpt_after_load",
    }
    changed = sorted(key for key in previous.keys() | current.keys()
                     if key not in restore_fields
                     and (key not in previous or key not in current or previous[key] != current[key]))
    if changed:
        raise ValueError(f"CP8 continuation changed effective configuration: {', '.join(changed)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("workspace", "prepared", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--steps", type=int, choices=range(1, 61), required=True)
    parser.add_argument("--response-limit", type=int, choices=(8192, 16384), default=8192)
    parser.add_argument("--historical", action="store_true", help="Use the original four-GPU, 64-prompt QAT workload")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--compose-only", action="store_true")
    args = parser.parse_args()
    if args.historical and args.response_limit != 16384:
        parser.error("--historical requires the original 16K response limit")
    if args.resume and not args.historical:
        parser.error("--resume requires the original QAT workload")
    workspace, prepared, output = (path.resolve() for path in (args.workspace, args.prepared, args.output))
    source = workspace / "qat"
    assets = workspace / "assets"
    overlay = prepared / "k2v3"
    reward_path = Path(__file__).with_name("math_reward.py").resolve()
    trainer = source / "vendor/verl/verl/trainer/ppo/ray_trainer.py"
    for path in (assets / DATASET, assets / "nemotron-brorl" / "config.json", overlay / "vllm" / "__init__.py",
                 reward_path, trainer):
        if not path.is_file():
            raise FileNotFoundError(f"Missing prepared asset: {path}")
    if "QAT math RL: initialize metadata for the rollout bypass" not in trainer.read_text(encoding="utf-8"):
        raise ValueError("Apply the runtime compatibility patch to the isolated experiment workspace")
    launcher = load_historical_launcher(workspace)
    digest = (workspace / "recorded.sha256").read_text().split()[0]
    manifest = launcher.pinned_manifest(workspace / "j9073-effective.json", digest)
    command, environment, _ = launcher.portable_command(manifest, source, assets, output)
    dependencies = f"{prepared / 'accelerate-overlay'}:{prepared / 'dependencies'}"
    command = launcher.native_launch(command, environment, dependencies, None, overlay, args.steps)
    if not args.historical:
        command = launcher.reward_two_gpu_command(command)
    command = make_rl_command(command, reward_path, args.response_limit, args.historical)
    if args.resume:
        args.resume = args.resume.resolve(strict=True)
        command = resume_command(command, args.resume, args.steps)
    output.mkdir(parents=True, exist_ok=False)
    composed = subprocess.run(
        [*command[:4], "--cfg", "job", *command[4:]], cwd=source,
        env=dict(environment, CUDA_VISIBLE_DEVICES=""), capture_output=True, check=False,
    )
    (output / "compose.stderr").write_bytes(composed.stderr)
    composed.check_returncode()
    config = yaml.safe_load(composed.stdout)
    check_config(config, args.steps, args.response_limit, args.historical)
    if args.resume:
        trainer = config["trainer"]
        if (trainer["resume_mode"], trainer["resume_from_path"], trainer["del_local_ckpt_after_load"]) != (
            "resume_path", str(args.resume), False
        ):
            raise ValueError("The composed configuration did not preserve the restore settings")
        parent_config = args.resume.parent.parent / "native-config.yaml"
        check_resume_fidelity(yaml.safe_load(parent_config.read_bytes()), config)
    (output / "native-config.yaml").write_bytes(composed.stdout)
    (output / "native-command.json").write_text(json.dumps(command, indent=2) + "\n")
    if args.compose_only:
        return
    with (output / "train.stdout").open("xb") as stdout, (output / "train.stderr").open("xb") as stderr:
        subprocess.run(command, cwd=source, env=environment, stdout=stdout, stderr=stderr, check=True)


if __name__ == "__main__":
    main()
