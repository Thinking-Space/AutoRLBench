#!/usr/bin/env python3
"""Grade a fresh 60-step cache repair; historical replay is reviewer-only.

The official entry point always checks fresh outputs, report, effective config,
immutable model/data assets and a complete HF-only step-60 checkpoint. Trusted
reviewer code can call evaluate(..., replay=True) to calibrate recorded evidence.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import sys


FINAL_STEP = 60
EXPECTED_SOURCES = ("aime24", "aime25")
RESOURCE_ROOT = Path(__file__).resolve().parents[1] / "resources"
MANDATORY_CONTRACT = {
    "data.train_batch_size": 512,
    "data.max_prompt_length": 2048,
    "data.max_response_length": 8192,
    "data.filter_overlong_prompts": False,
    "data.truncation": "left",
    "data.shuffle": True,
    "data.seed": 42,
    "data.train_max_samples": -1,
    "data.val_max_samples": -1,
    "data.apply_chat_template_kwargs.enable_thinking": False,
    "data.custom_cls.name": "PromptDataset",
    "algorithm.adv_estimator": "grpo",
    "algorithm.use_kl_in_reward": False,
    "actor_rollout_ref.model.use_remove_padding": True,
    "actor_rollout_ref.model.override_config.attn_implementation": "flash_attention_2",
    "actor_rollout_ref.actor.strategy": "fsdp2",
    "actor_rollout_ref.actor.use_dynamic_bsz": True,
    "actor_rollout_ref.actor.ppo_mini_batch_size": 32,
    "actor_rollout_ref.actor.ppo_epochs": 1,
    "actor_rollout_ref.actor.shuffle": False,
    "actor_rollout_ref.actor.ppo_max_token_len_per_gpu": 10240,
    "actor_rollout_ref.actor.use_kl_loss": False,
    "actor_rollout_ref.actor.entropy_coeff": 0.0,
    "actor_rollout_ref.actor.optim.lr": 1e-6,
    "actor_rollout_ref.actor.optim.weight_decay": 0.01,
    "actor_rollout_ref.actor.optim.lr_scheduler_type": "constant",
    "actor_rollout_ref.actor.optim.lr_warmup_steps_ratio": 0.0,
    "actor_rollout_ref.actor.policy_loss.loss_mode": "vanilla",
    "actor_rollout_ref.actor.loss_agg_mode": "token-mean",
    "actor_rollout_ref.actor.fsdp_config.param_offload": False,
    "actor_rollout_ref.actor.fsdp_config.optimizer_offload": False,
    "actor_rollout_ref.actor.checkpoint.save_contents": ["hf_model"],
    "actor_rollout_ref.rollout.n": 8,
    "actor_rollout_ref.rollout.mode": "async",
    "actor_rollout_ref.rollout.temperature": 0.7,
    "actor_rollout_ref.rollout.top_p": 1.0,
    "actor_rollout_ref.rollout.top_k": -1,
    "actor_rollout_ref.rollout.tensor_model_parallel_size": 1,
    "actor_rollout_ref.rollout.gpu_memory_utilization": 0.7,
    "actor_rollout_ref.rollout.max_num_seqs": 256,
    "actor_rollout_ref.rollout.enforce_eager": False,
    "actor_rollout_ref.rollout.log_prob_use_dynamic_bsz": True,
    "actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu": 10240,
    "actor_rollout_ref.rollout.val_kwargs.n": 32,
    "actor_rollout_ref.rollout.val_kwargs.do_sample": True,
    "actor_rollout_ref.rollout.val_kwargs.temperature": 0.6,
    "actor_rollout_ref.rollout.val_kwargs.top_p": 0.95,
    "actor_rollout_ref.rollout.val_kwargs.top_k": 20,
    "trainer.total_training_steps": FINAL_STEP,
    "trainer.save_freq": FINAL_STEP,
    "trainer.test_freq": 10,
    "trainer.val_before_train": True,
    "trainer.nnodes": 2,
    "trainer.n_gpus_per_node": 4,
    "reward_model.reward_manager": "naive",
    "custom_reward_function.name": "compute_score",
}


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def jsonfile(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def nested(config, name):
    for part in name.split("."):
        config = config[part]
    return config


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def exact_path(value, expected, description):
    if isinstance(value, list):
        if len(value) != 1:
            raise ValueError(f"{description}: expected one path")
        value = value[0]
    if not isinstance(value, str) or Path(value).resolve() != expected.resolve():
        raise ValueError(f"Unexpected {description}")


def check_asset(path, description):
    expected_size = description.get("size_bytes", description.get("size"))
    expected_hash = description.get("sha256", "")
    if not isinstance(expected_size, int) or not re.fullmatch(r"[0-9a-f]{64}", expected_hash):
        raise ValueError("Trusted asset registry requires SHA256 and byte size")
    if not path.is_file() or path.stat().st_size != expected_size or sha256(path) != expected_hash:
        raise ValueError(f"Immutable asset modified or missing: {path.name}")


def verify_checkpoint(root):
    """Read safetensors headers and payload bounds; optimizer state is not required."""
    hf = root / "checkpoints/global_step_60/actor/huggingface"
    config = jsonfile(hf / "config.json")
    if config.get("model_type") != "qwen3":
        raise ValueError("Final checkpoint must be Qwen3")
    shards = sorted(hf.glob("*.safetensors"))
    if not shards:
        raise ValueError("Missing HF-only step60 checkpoint weights")
    widths = {"F64": 8, "F32": 4, "F16": 2, "BF16": 2, "I64": 8,
              "I32": 4, "I16": 2, "I8": 1, "U8": 1, "BOOL": 1}
    tensor_files = {}
    parameter_count = 0
    for shard in shards:
        with shard.open("rb") as handle:
            prefix = handle.read(8)
            if len(prefix) != 8:
                raise ValueError("Truncated safetensors header")
            header_size = int.from_bytes(prefix, "little")
            if not 2 <= header_size <= 64 * 1024 * 1024:
                raise ValueError("Invalid safetensors header size")
            header_raw = handle.read(header_size)
            if len(header_raw) != header_size:
                raise ValueError("Truncated safetensors header")
        header = json.loads(header_raw)
        payload_bytes = shard.stat().st_size - 8 - header_size
        intervals = []
        for name, entry in header.items():
            if name == "__metadata__":
                continue
            if name in tensor_files:
                raise ValueError("Duplicate checkpoint tensor")
            shape = entry["shape"]
            if not isinstance(shape, list) or any(type(dim) is not int or dim < 0 for dim in shape):
                raise ValueError("Invalid checkpoint tensor shape")
            count = math.prod(shape)
            offsets = entry["data_offsets"]
            if not isinstance(offsets, list) or len(offsets) != 2 or any(type(x) is not int for x in offsets):
                raise ValueError("Invalid checkpoint tensor offsets")
            start, end = offsets
            if start < 0 or end < start or end > payload_bytes:
                raise ValueError("Truncated checkpoint tensor payload")
            if end - start != count * widths[entry["dtype"]]:
                raise ValueError("Checkpoint dtype/shape/payload mismatch")
            intervals.append((start, end))
            tensor_files[name] = shard.name
            parameter_count += count
        cursor = 0
        for start, end in sorted(intervals):
            if start != cursor:
                raise ValueError("Checkpoint payload is overlapping or incomplete")
            cursor = end
        if cursor != payload_bytes:
            raise ValueError("Checkpoint payload length mismatch")
    index_path = hf / "model.safetensors.index.json"
    if index_path.exists() and jsonfile(index_path).get("weight_map") != tensor_files:
        raise ValueError("Checkpoint shard index does not match tensors")
    if len(shards) > 1 and not index_path.is_file():
        raise ValueError("Missing sharded checkpoint index")
    if parameter_count < 1_000_000_000:
        raise ValueError("Missing/incomplete HF checkpoint parameters")
    return {"format": "HF-only safetensors", "parameter_count": parameter_count, "shards": len(shards)}


def validate_fresh_contract(root, workspace, spec, resources):
    report = workspace / "report.md"
    if report.is_symlink() or not report.is_file() or not report.read_text(encoding="utf-8").strip():
        raise ValueError("Missing nonempty report.md")
    config = jsonfile(root / "effective_config.json")
    contract = jsonfile(resources / "contract.json")
    if not isinstance(contract, dict) or not contract:
        raise ValueError("Missing trusted configuration contract")
    for name, wanted in MANDATORY_CONTRACT.items():
        if nested(config, name) != wanted:
            raise ValueError(f"Effective configuration contract differs: {name}")
    for name, wanted in contract.items():
        if name in MANDATORY_CONTRACT and wanted != MANDATORY_CONTRACT[name]:
            raise ValueError(f"Trusted contract disagrees with 60-step recipe: {name}")
        if nested(config, name) != wanted:
            raise ValueError(f"Effective configuration contract differs: {name}")
    protected_code = jsonfile(resources / "protected_code.json")
    required_code = {"case/dapo_reward.py", "verl/utils/reward_score/math_dapo.py",
                     "verl/trainer/ppo/core_algos.py"}
    if not isinstance(protected_code, dict) or not required_code <= set(protected_code):
        raise ValueError("Trusted protected-code registry must cover fixed reward and algorithm sources")
    for relative, description in protected_code.items():
        if not isinstance(relative, str) or Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise ValueError("Invalid trusted protected-code path")
        try:
            check_asset(workspace / relative, description)
        except Exception as error:
            raise ValueError(f"Protected training code modified or missing: {relative}") from error
    exact_path(nested(config, "actor_rollout_ref.model.path"), workspace / "model", "model path")
    exact_path(nested(config, "data.custom_cls.path"), workspace / "case/dataset_cache.py", "dataset implementation path")
    registry = jsonfile(resources / "asset_registry.json")
    required = {"train.parquet", "val.parquet", "prompt-cache-v1.parquet", "prompt-cache-v1.json"}
    if not required <= set(registry.get("derived", {})):
        raise ValueError("Trusted registry must cover train, validation and both cache-seed inputs")
    paths = {"train.parquet": "data.train_files", "val.parquet": "data.val_files",
             "prompt-cache-v1.parquet": "data.cache_seed_path"}
    paths.update(spec.get("asset_config_paths", {}))
    for relative, description in registry["derived"].items():
        if Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise ValueError("Invalid trusted asset path")
        destination = workspace / "data" / relative
        if relative in paths:
            exact_path(nested(config, paths[relative]), destination, f"{relative} path")
        check_asset(destination, description)
    exact_path(nested(config, "data.cache_manifest_path"), workspace / "data/manifest.json", "cache manifest path")
    if not isinstance(registry.get("prepared_manifest"), dict):
        raise ValueError("Trusted registry must cover the prepared manifest")
    check_asset(workspace / "data/manifest.json", registry["prepared_manifest"])
    exact_path(nested(config, "data.cache_dir"), root / "processed-prompt-cache", "fresh processed-cache directory")
    model_files = registry.get("model", {}).get("files", [])
    if not model_files:
        raise ValueError("Missing pinned base model files in trusted registry")
    for description in model_files:
        relative = description["path"]
        if Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise ValueError("Invalid trusted model asset path")
        check_asset(workspace / "model" / relative, description)
    return verify_checkpoint(root)


def accuracy(data, source):
    aliases = (f"val-core/{source}/acc/mean@32", f"val-aux/{source}/reward/mean@32",
               f"val-aux/{source}/score/mean@32", f"val-aux/{source}/oracle_acc/mean@32")
    values = [data[name] for name in aliases if name in data]
    if not values or any(not finite(value) or not 0 <= value <= 1 for value in values):
        raise ValueError("Missing/invalid scheduled AIME evaluation")
    if max(values) - min(values) > 1e-9:
        raise ValueError("Logged AIME metric aliases disagree")
    return values[0]


def validate_items(items):
    if not isinstance(items, list) or len(items) != 60:
        raise ValueError("Trusted eval items must contain exactly 60 questions")
    expected = {}
    sources = Counter()
    for item in items:
        identity = item["problem_id"]
        if not isinstance(identity, str) or not identity or identity in expected:
            raise ValueError("Duplicate/invalid trusted evaluation question ID")
        if not isinstance(item["input"], str) or not re.fullmatch(r"[+-]?[0-9]+", str(item["ground_truth"])):
            raise ValueError("Invalid trusted evaluation input/gold")
        source = item.get("source", item.get("data_source"))
        if "source" in item and "data_source" in item and item["source"] != item["data_source"]:
            raise ValueError("Trusted evaluation source aliases disagree")
        expected[identity] = {**item, "source": source}
        sources[source] += 1
    if sources != Counter({"aime24": 30, "aime25": 30}):
        raise ValueError("Trusted evaluation must contain 30 questions per AIME year")
    return expected


def evaluate(results, spec, items, *, replay=False, workspace=None, resources=None):
    errors, detail = [], {}
    value = None
    resources = Path(resources) if resources is not None else RESOURCE_ROOT
    results = Path(results)
    root = results / "experiment_logs"
    if replay and not root.is_dir():
        root = results
    try:
        if spec.get("final_step") != FINAL_STEP or spec.get("metric_key") != "aime24_25_mean32_average" or spec.get("score_mode") != "final":
            raise ValueError("Unsupported grading specification; require final AIME average at step60")
        threshold, maximum = spec["required_final_score"], spec["required_final_score_max"]
        if not finite(threshold) or not finite(maximum) or not 0 < threshold <= maximum <= 1:
            raise ValueError("Invalid grading threshold")
        if not replay:
            if workspace is None:
                raise ValueError("Missing workspace for fresh-run verification")
            detail["checkpoint"] = validate_fresh_contract(root, Path(workspace), spec, resources)
        rows = load_rows(root / "metrics.jsonl")
        steps = [row["step"] for row in rows]
        if steps != list(range(FINAL_STEP + 1)) or any(type(step) is not int for step in steps):
            raise ValueError("Expected exactly one ordered metric record at every step 0..60")
        for row in rows:
            for key, number in row["data"].items():
                if isinstance(number, (int, float)) and not isinstance(number, bool) and not finite(number):
                    raise ValueError(f"Nonfinite metric: {key}")
            if row["step"]:
                data = row["data"]
                if not finite(data.get("actor/grad_norm")) or data.get("training/global_step") != row["step"]:
                    raise ValueError("Missing/inconsistent training step or gradient")
                if data.get("response/aborted_ratio") != 0:
                    raise ValueError("Missing/nonzero aborted response ratio")
        for step in range(0, FINAL_STEP + 1, 10):
            for source in EXPECTED_SOURCES:
                accuracy(rows[step]["data"], source)
        expected = validate_items(items)
        samples_path = root / "fresh_eval/60.jsonl"
        if replay and not samples_path.exists():
            samples_path = next((path for path in (
                root / "validation-60.jsonl", root / "frozen_eval/60.jsonl", root / "validation/60.jsonl"
            ) if path.is_file()), root / "validation-60.jsonl")
        samples = load_rows(samples_path)
        if len(samples) != 1920:
            raise ValueError("Final evaluation must contain 1920 completions")
        from score_math import score
        counts = Counter()
        correct = defaultdict(list)
        for sample in samples:
            identity = sample["problem_id"]
            if identity not in expected:
                raise ValueError("Unknown evaluation question")
            wanted = expected[identity]
            if (sample["input"] != wanted["input"] or str(sample["gts"]) != str(wanted["ground_truth"])
                    or sample["data_source"] != wanted["source"] or sample["step"] != FINAL_STEP):
                raise ValueError("Final evaluation question/answer identity mismatch")
            result = float(score(sample["output"], sample["gts"]))
            for key in ("score", "acc", "reward", "oracle_acc"):
                if key in sample and (not isinstance(sample[key], (int, float)) or float(sample[key]) != result):
                    raise ValueError("Per-response logged score differs from independent text rescoring")
            counts[identity] += 1
            correct[wanted["source"]].append(result)
        if set(counts) != set(expected) or set(counts.values()) != {32}:
            raise ValueError("Each of the 60 questions must have exactly 32 completions")
        for source in EXPECTED_SOURCES:
            observed = statistics.mean(correct[source])
            if abs(observed - accuracy(rows[-1]["data"], source)) > 1e-9:
                raise ValueError("Logged AIME accuracy differs from independent text rescoring")
            detail[source] = observed
        value = statistics.mean(detail[source] for source in EXPECTED_SOURCES)
        if value < threshold or value > maximum:
            errors.append("Metric outside accepted interval")
    except Exception as error:
        errors.append(f"{type(error).__name__}: {error}")
    return {"passed": not errors, "score": float(not errors), "case_score": float(not errors),
            "value": value, "independent_final_accuracy": {key: detail[key] for key in EXPECTED_SOURCES if key in detail},
            "verification": detail, "errors": errors, "historical_replay": replay,
            "fresh_gpu_run_verified": not replay and not errors}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path)
    parser.add_argument("--historical-replay", type=Path,
                        help="Reviewer-only recorded evidence; never accepted in the official harness")
    args = parser.parse_args()
    if args.historical_replay:
        if os.environ.get("AUTORLBENCH_REVIEWER_REPLAY") != "1" or any(
            os.environ.get(key) for key in ("AUTORLBENCH_RESULTS", "AUTORLBENCH_WORKSPACE", "AUTORLBENCH_VERIFIER")
        ):
            parser.error("Historical replay requires AUTORLBENCH_REVIEWER_REPLAY=1 outside the official harness")
        results = args.historical_replay
    else:
        results = args.results or Path(os.environ.get("AUTORLBENCH_RESULTS", "/results"))
    result = evaluate(results, jsonfile(RESOURCE_ROOT / "grader_spec.json"),
                      jsonfile(RESOURCE_ROOT / "eval_items.json"), replay=bool(args.historical_replay),
                      workspace=Path(os.environ.get("AUTORLBENCH_WORKSPACE", "/workspace")))
    destination = args.results if args.historical_replay else results
    if destination is not None:
        destination.mkdir(parents=True, exist_ok=True)
        for name in ("result.json", "metrics.json"):
            (destination / name).write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, allow_nan=False))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
