#!/usr/bin/env python3
"""Prepare a relocated recorded config, check inputs, or launch the issue run."""

import argparse
import importlib.util
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import runpy
import re
import shutil
import tempfile
from datetime import datetime, timezone

from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "idea_source"))


def runtime_defaults():
    """Optional machine-local paths, never credentials or historical evidence."""
    path = ROOT.parent / "runtime.local.json"
    if not path.is_file():
        return {}
    values = json.loads(path.read_text())
    unknown = set(values) - {"awm_python", "model_path", "model_revision", "dataset_revision"}
    if unknown:
        raise ValueError("Unsupported runtime.local.json keys: " + ", ".join(sorted(unknown)))
    return values


def hub_token():
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"]
    path = os.environ.get("HF_TOKEN_PATH")
    return Path(path).read_text().strip() if path else None


def immutable_revision(value):
    if not value or not re.fullmatch(r"[0-9a-fA-F]{40}", value):
        raise ValueError("Downloads require a 40-character immutable HF commit SHA")
    return value


def download_data(repo_id, revision):
    from huggingface_hub import snapshot_download
    immutable_revision(revision)
    snapshot = Path(snapshot_download(repo_id, repo_type="dataset", revision=revision, token=hub_token(),
                                     allow_patterns=["rl/*", "runtime/*", "manifest.json"]))
    root = ROOT / "data/AgentWorldModel-1K"
    mapping = [(snapshot / "rl/train.parquet", root / "verl/train.parquet"),
               (snapshot / "rl/validation.parquet", root / "verl/test.parquet")]
    names = ["gen_db", "gen_envs", "gen_sample", "gen_tasks", "gen_verifier.pure_code"]
    mapping += [(snapshot / "runtime" / (name + ".jsonl"), root / (name + ".jsonl")) for name in names]
    for source, target in mapping:
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copy2(source, target)
    manifest = json.loads((ROOT.parent / "environment/data_manifest.json").read_text())
    expected = {item["path"]: item for item in manifest["files"]}
    for _, target in mapping:
        item = expected[target.relative_to(ROOT).as_posix()]
        with target.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if target.stat().st_size != item["bytes"] or digest != item["sha256"]:
            raise ValueError("Restored data differs from historical inputs: " + target.name)


def prepare(args):
    original = OmegaConf.load(ROOT / "config/.hydra/config.yaml")
    config = OmegaConf.create(OmegaConf.to_container(original, resolve=False))
    data = ROOT / "data/AgentWorldModel-1K"
    output = Path(args.output).absolute()
    python = Path(args.awm_python).absolute()
    model = Path(args.model_path).expanduser()
    if not model.is_dir():
        if model.is_absolute() or args.model_path.startswith("."):
            raise FileNotFoundError(model)
        immutable_revision(args.model_revision)
        from huggingface_hub import snapshot_download
        model = Path(snapshot_download(args.model_path, revision=args.model_revision, token=hub_token(),
                                       local_dir=ROOT / "models/IDEA_sft"))
    model = model.absolute()
    config.data.train_files = str(data / "verl/train.parquet")
    config.data.val_files = str(data / "verl/test.parquet")
    config.actor_rollout_ref.model.path = str(model)
    config.env.awm.root = str(ROOT / "awm_source")
    config.env.awm.python = str(python)
    config.env.awm.data_root = str(data)
    config.env.awm.work_root = str(output / "awm_runtime")
    # Code rewards do not use a hosted judge; never carry a recorded credential
    # into a new config or console log.
    config.env.awm.judge_api_key = (
        os.environ.get("AWM_JUDGE_API_KEY", "EMPTY")
        if config.env.awm.reward_mode == "sql" else "EMPTY"
    )
    if config.env.awm.reward_mode == "sql" and config.env.awm.judge_api_key == "EMPTY":
        raise RuntimeError("SQL reward mode requires AWM_JUDGE_API_KEY")
    for field, folder in (("analysis_dump_dir", "idea_analysis"),
                          ("augmented_observation_dump_dir", "idea_augmented_observations"),
                          ("state_group_dump_dir", "idea_state_group")):
        config.algorithm.seed[field] = str(output / folder)
    config.trainer.rollout_data_dir = str(output / "rollouts")
    config.trainer.default_local_dir = str(output / "checkpoints")
    config.trainer.resume_mode = "disable"
    config.trainer.resume_from_path = None
    output.mkdir(parents=True, exist_ok=True)
    OmegaConf.resolve(config)
    OmegaConf.save(config, output / "config.yaml")
    return config, output


def check(config):
    import pyarrow.parquet as pq
    data = Path(config.env.awm.data_root)
    required = ["gen_tasks", "gen_envs", "gen_db", "gen_sample", "gen_verifier.pure_code"]
    records = {}
    for name in required:
        with (data / f"{name}.jsonl").open() as stream:
            records[name] = [json.loads(line) for line in stream if line.strip()]
    scenarios = {name: {row["scenario"] for row in rows} for name, rows in records.items()}
    tasks = {row["scenario"]: row["tasks"] for row in records["gen_tasks"]}
    splits = {}
    keys = {}
    for split in ("train", "test"):
        rows = pq.read_table(data / f"verl/{split}.parquet").to_pylist()
        keys[split] = set()
        for row in rows:
            item = row["env_kwargs"]
            scenario, index = item["scenario"], item["task_idx"]
            assert item["task"] == tasks[scenario][index], (scenario, index)
            assert all(scenario in values for values in scenarios.values()), scenario
            keys[split].add((scenario, index))
        assert len(keys[split]) == len(rows), f"duplicate {split} task"
        splits[split] = rows
    assert not keys["train"] & keys["test"], "train/test overlap"
    assert len(keys["train"] | keys["test"]) == sum(len(t) for t in tasks.values())
    model = Path(config.actor_rollout_ref.model.path)
    for filename in ("config.json", "tokenizer.json", "tokenizer_config.json"):
        if not (model / filename).is_file():
            raise FileNotFoundError(model / filename)
    if not list(model.glob("*.safetensors")) and not list(model.glob("pytorch_model*.bin")):
        raise FileNotFoundError(f"Model weights not found in {model}")
    if not Path(config.env.awm.python).is_file():
        raise FileNotFoundError(config.env.awm.python)
    return splits, {"split_rows": {k: len(v) for k, v in splits.items()},
                    "scenario_counts": {k: len(v) for k, v in scenarios.items()},
                    "task_alignment": "passed", "train_test_overlap": 0,
                    "model_config_and_weights_present": True}


def smoke(config, output, splits):
    # Load the packaged environment directly, without importing unrelated envs.
    path = ROOT / "idea_source/agent_system/environments/env_package/awm/envs.py"
    spec = importlib.util.spec_from_file_location("packaged_awm_env", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    selected = []
    seen = set()
    for rows in (splits["train"], splits["test"], splits["train"][1:]):
        for row in rows:
            item = row["env_kwargs"]
            if item["scenario"] not in seen:
                selected.append(item)
                seen.add(item["scenario"])
                break
    env = module.AWMMCPVectorEnv(len(selected), 1, config.env)
    try:
        tasks, _ = env.reset(selected)
        observations, _, dones, infos = env.step([{"name": "list_tools", "arguments": None}] * len(selected))
        for item, observation, done, info in zip(selected, observations, dones, infos):
            if done or info.get("reward_type") == "environment_error":
                raise RuntimeError(f"Environment startup failed: {item['scenario']}: {info}")
            assert observation, "empty tool listing"
        _, rewards, dones, infos = env.step([{"name": "final", "content": "Smoke test, no task actions performed."}] * len(selected))
        assert all(dones), "final action did not terminate"
        for item, info in zip(selected, infos):
            if info.get("reward_type") in {"environment_error", "judge_error", "server_error"}:
                raise RuntimeError(f"Verifier failed: {item['scenario']}: {info}")
        return [{"scenario": item["scenario"], "task_idx": item["task_idx"],
                 "reset": bool(task), "list_tools": "passed", "final": "passed",
                 "reward": reward, "reward_type": info.get("reward_type")}
                for item, task, reward, info in zip(selected, tasks, rewards, infos)]
    finally:
        env.close()


def main():
    defaults = runtime_defaults()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=os.environ.get("MODEL_PATH", defaults.get("model_path", "locaszzz/IDEA_sft")))
    parser.add_argument("--model-revision", default=os.environ.get("MODEL_REVISION", defaults.get("model_revision")))
    parser.add_argument("--dataset-revision", default=os.environ.get("DATASET_REVISION", defaults.get("dataset_revision")))
    parser.add_argument("--dataset-repo", default="locaszzz/IDEA_training_data")
    parser.add_argument("--download-data", action="store_true", help="Fetch RL/runtime inputs from the Hub dataset if absent; authenticate only if required")
    parser.add_argument("--awm-python", default=os.environ.get("AWM_PYTHON", defaults.get("awm_python", "/opt/awm/bin/python")))
    parser.add_argument("--output", default=str(ROOT / "reproduction" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")))
    parser.add_argument("--smoke", action="store_true", help="Test MCP startup, list_tools, and code verifier; no model inference")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--train", action="store_true", help="Launch the recorded 200-step issue configuration; requires 8 free 80GB GPUs")
    mode.add_argument("--config-only", action="store_true", help="Also check Hydra composition without starting Ray or training")
    args, overrides = parser.parse_known_args()
    if overrides and not (args.train or args.config_only):
        parser.error("Hydra overrides require --train or --config-only")
    if args.download_data:
        download_data(args.dataset_repo, args.dataset_revision)
    if not Path(args.awm_python).is_file():
        parser.error("Set AWM_PYTHON or configure runtime.local.json with the separate AWM interpreter")
    config, output = prepare(args)
    splits, report = check(config)
    report["python"] = sys.executable
    report["scope"] = "input checks; NOT a completed training run"
    if args.smoke:
        report["environment_smoke"] = smoke(config, output, splits)
        import verl.workers.fsdp_workers as workers
        tests = runpy.run_path(str(ROOT / "idea_source/tests/test_awm_projection.py"))
        names = [name for name in tests if name.startswith("test_")]
        for name in names:
            tests[name]()
        report["fsdp_workers_import"] = str(Path(workers.__file__).relative_to(ROOT))
        report["projection_tests_passed"] = len(names)
    (output / "checks.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    if args.train or args.config_only:
        env = os.environ.copy()
        env["PYTHONPATH"] = str(ROOT / "idea_source")
        env["PATH"] = f"{Path(sys.executable).parent}:{env.get('PATH', '')}"
        env.setdefault("WANDB_MODE", "offline")
        env["WANDB_DIR"] = str(output)
        env.setdefault("TOKENIZERS_PARALLELISM", "false")
        env.setdefault("VLLM_ATTENTION_BACKEND", "FLASH_ATTN")
        env.setdefault("NCCL_IB_DISABLE", "1")
        env.setdefault("NCCL_P2P_DISABLE", "0")
        env.setdefault("TORCH_NCCL_BLOCKING_WAIT", "1")
        ray_root = os.environ.get("RAY_TMPDIR")
        env["RAY_TMPDIR"] = ray_root or str(Path(tempfile.gettempdir()) / f"i4ray-{os.getpid()}")
        Path(env["RAY_TMPDIR"]).mkdir(parents=True, exist_ok=True)
        for name in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"):
            env.pop(name, None)
        command = [sys.executable, "-m", "verl.trainer.main_ppo", "--config-path", str(output), "--config-name", "config"]
        command += ["hydra.run.dir=" + str(output)]
        if args.config_only:
            command += ["--cfg", "job"]
        command += overrides
        log_path = output / ("hydra-check.log" if args.config_only else "train.log")
        with log_path.open("wb") as log:
            result = subprocess.run(command, cwd=ROOT / "idea_source", env=env, stdout=log, stderr=subprocess.STDOUT)
        (output / "exit-status.json").write_text(json.dumps({"exit_code": result.returncode, "config_only": args.config_only}) + "\n")
        print(f"Exit code: {result.returncode}; combined stdout/stderr: {log_path}")
        return result.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
