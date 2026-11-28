"""Verify two logged image-cache RL runs. Uses only the Python standard library."""

import argparse
import json
import math
import random
import re
import sys
from pathlib import Path


class InvalidRun(ValueError):
    """A run or the matched pair failed a check."""


COUNTERS = ("hits", "misses", "evictions", "requests", "encoder_calls", "encoded_images",
            "prefill_batches", "split_batches", "multi_item_split_batches")


PIN = "02b5e123c490c988d21526d0dd7837fefdd0e1da"


def sha256(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def check(ok, message):
    if not ok:
        raise InvalidRun(message)


def integer(value):
    return type(value) is int


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def read_run(path):
    records = []
    with path.open(encoding="utf-8") as stream:
        for i, line in enumerate(stream, 1):
            if line.strip():
                try:
                    records.append(json.loads(line))
                except ValueError as error:
                    raise InvalidRun(f"{path.name} line {i}: {error}") from error
    check(len(records) >= 3 and all(isinstance(row, dict) for row in records),
          f"{path.name}: expected a meta row, training steps and a final row")
    kinds = [row.get("type") for row in records]
    check(kinds == ["meta"] + ["step"] * (len(records) - 2) + ["final"],
          f"{path.name}: expected exactly meta, step(s), final in that order")
    meta, steps, final = records[0], records[1:-1], records[-1]
    check(integer(final.get("steps")) and final["steps"] == len(steps),
          f"{path.name}: final.steps does not match the number of step rows")
    accuracy = final.get("heldout_accuracy")
    check(finite(accuracy) and 0 <= accuracy <= 1, f"{path.name}: invalid heldout_accuracy")
    check(integer(meta.get("batch_size")) and meta["batch_size"] > 0
          and integer(meta.get("group_size")) and meta["group_size"] > 1
          and integer(meta.get("seed")) and isinstance(meta.get("device"), str)
          and sha256(meta.get("initial_policy_digest")), f"{path.name}: incomplete policy metadata")
    check(meta.get("model") == "HuggingFaceTB/SmolVLM-256M-Instruct" and
          meta.get("revision") == "d039ab0d4b9311b617e8ffec1d06c8464364aeb3" and
          meta.get("weights_sha256") == "74dea5904032e5ae99a2e0eef5179e6ac0f1dedc3ab0c7c2a5d4d387c843203e",
          f"{path.name}: unexpected public model")
    token_ids = meta.get("legal_token_ids")
    check(isinstance(token_ids, list) and len(token_ids) == 2 and len(set(token_ids)) == 2 and
          all(integer(t) for t in token_ids) and meta.get("legal_answers") == [" No", " Yes"],
          f"{path.name}: invalid answer vocabulary")
    native = meta.get("native_forward_check", {})
    check(native.get("n_logits") == 4 and finite(native.get("max_abs_diff")) and
          finite(native.get("tolerance")) and 0 <= native["max_abs_diff"] <= native["tolerance"],
          f"{path.name}: split decoder did not pass the native-forward check")
    last_digest = meta["initial_policy_digest"]
    for i, row in enumerate(steps):
        number = row.get("step")
        context = f"{path.name} step {number}"
        check(integer(number) and (number in (0, 1) if i == 0 else number == steps[i-1]["step"] + 1),
              f"{context}: expected consecutive training steps starting at 0 or 1")
        actions, rewards, logps = (row.get(key) for key in ("actions", "rewards", "logprobs"))
        check(all(isinstance(v, list) for v in (actions, rewards, logps)) and
              len(actions) == len(rewards) == len(logps) == meta["batch_size"] * meta["group_size"],
              f"{context}: action, reward and logprob counts must match")
        check(all(integer(v) and v >= 0 for v in actions), f"{context}: invalid action")
        check(all(finite(v) and v in (0, 1) for v in rewards), f"{context}: expected binary rewards")
        check(all(finite(v) and v <= 1e-7 for v in logps), f"{context}: invalid action logprob")
        scenes, prompts = row.get("scene_ids"), row.get("prompt_ids")
        check(isinstance(scenes, list) and isinstance(prompts, list) and
              len(scenes) == len(prompts) == meta["batch_size"] and all(integer(s) for s in scenes) and
              all(p in (0, 1) for p in prompts) and all(a in token_ids for a in actions),
              f"{context}: incomplete scenes, prompts or decoded actions")
        for idx, (scene, prompt) in enumerate(zip(scenes, prompts)):
            correct = token_ids[int(random.Random(scene).randrange(2) == prompt)]
            start = idx * meta["group_size"]
            end = start + meta["group_size"]
            check(rewards[start:end] == [int(a == correct) for a in actions[start:end]],
                  f"{context}: rewards disagree with the generated beacon scene")
        valid_groups = row.get("valid_groups")
        check(integer(valid_groups) and 0 < valid_groups <= min(rewards.count(0), rewards.count(1)),
              f"{context}: expected mixed-reward GRPO groups")
        group_size = row.get("group_size", meta.get("group_size"))
        if group_size is not None:
            check(integer(group_size) and group_size >= 2 and len(rewards) % group_size == 0,
                  f"{context}: incomplete or invalid rollout group")
            mixed = sum(len(set(rewards[start:start + group_size])) == 2
                        for start in range(0, len(rewards), group_size))
            check(mixed == valid_groups, f"{context}: valid_groups disagrees with grouped rewards")
        for key in ("grad_norm", "weight_delta"):
            check(finite(row.get(key)) and row[key] > 0, f"{context}: expected positive {key}")
        digest = row.get("policy_digest")
        check(isinstance(digest, str) and re.fullmatch(r"[0-9a-f]{64}", digest) and digest != last_digest,
              f"{context}: invalid or unchanged policy SHA-256")
        last_digest = digest
        cache = row.get("cache")
        check(isinstance(cache, dict) and all(integer(cache.get(k)) and cache[k] >= 0 for k in
              ("logical_bytes", "storage_bytes", "capacity_bytes")), f"{context}: invalid cache counts")
        check(cache["capacity_bytes"] > 0 and cache["logical_bytes"] <= cache["capacity_bytes"],
              f"{context}: invalid budget or logical bytes exceed the advertised budget")
        for key in COUNTERS:
            previous = steps[i-1]["cache"][key] if i else 0
            check(integer(cache.get(key)) and cache[key] >= previous,
                  f"{context}: missing or decreasing {key} counter")
        for per_step, total in (("encoder_calls", "encoder_calls"), ("encoder_images", "encoded_images")):
            if per_step in row:
                previous = steps[i-1]["cache"][total] if i else 0
                check(integer(row[per_step]) and previous + row[per_step] == cache[total],
                      f"{context}: vision encoder and scheduler disagree on {per_step}")
        check(cache.get("revision") == PIN and cache.get("scheduler_entry") == "_get_chunked_prefill_embedding"
              and cache.get("backend") in {"installed-sglang", "pinned-sglang-definitions"},
              f"{context}: missing or incorrect SGLang provenance")
        check(isinstance(cache.get("keys"), list) and cache["keys"] and all(integer(k) for k in cache["keys"])
              and len(set(cache["keys"])) == len(cache["keys"]) and sha256(cache.get("lookup_sha256")),
              f"{context}: missing or invalid LRU keys or lookup digest")
        check(isinstance(cache.get("device"), str)
              and cache["device"].split(":")[0] == meta["device"].split(":")[0],
              f"{context}: cache embeddings and policy device disagree")
        check(integer(cache.get("storage_count")) and 0 < cache["storage_count"] <= len(cache["keys"]),
              f"{context}: missing or invalid unique storage count")
        groups = cache.get("storage_groups")
        check(isinstance(groups, list) and len(groups) == cache["storage_count"], f"{context}: missing storage groups")
        positions = {key: i for i, key in enumerate(cache["keys"])}
        seen = set()
        first = -1
        for group in groups:
            check(isinstance(group, dict) and group.get("device") == cache["device"]
                  and integer(group.get("storage_bytes")) and group["storage_bytes"] > 0
                  and integer(group.get("logical_bytes")) and group["logical_bytes"] > 0
                  and isinstance(group.get("keys"), list) and group["keys"]
                  and integer(group.get("key_count")) and group["key_count"] == len(group["keys"]),
                  f"{context}: invalid storage group")
            group_keys = group["keys"]
            check(all(integer(k) and k in positions and k not in seen for k in group_keys)
                  and len(set(group_keys)) == len(group_keys), f"{context}: invalid storage key partition")
            indices = [positions[k] for k in group_keys]
            check(indices == sorted(indices) and indices[0] > first, f"{context}: unstable storage grouping")
            first = indices[0]
            seen.update(group_keys)
        check(seen == set(positions) and sum(g["storage_bytes"] for g in groups) == cache["storage_bytes"]
              and sum(g["logical_bytes"] for g in groups) == cache["logical_bytes"],
              f"{context}: storage totals do not match the live cache entries")
        shape = cache.get("assembled_embedding_shape")
        check(isinstance(shape, list) and len(shape) == 2 and all(integer(v) and v > 0 for v in shape)
              and isinstance(cache.get("assembled_embedding_dtype"), str) and cache["assembled_embedding_dtype"].startswith("torch.")
              and sha256(cache.get("assembled_embedding_sha256")), f"{context}: missing assembled embeddings")
        check(cache["split_batches"] == cache["encoder_calls"] and cache["multi_item_split_batches"] <= cache["split_batches"],
              f"{context}: split and encoder counts disagree")
    check(final.get("policy_digest") == last_digest, f"{path.name}: final policy differs from last update")
    return meta, steps, final


def cache_snapshot(row):
    cache = row["cache"]
    return {"step": row["step"], **{k: cache[k] for k in ("capacity_bytes", "logical_bytes", "storage_bytes")}}


def report_arm(meta, steps, final):
    peak = max(steps, key=lambda row: row["cache"]["storage_bytes"])
    over = [row for row in steps if row["cache"]["storage_bytes"] > row["cache"]["capacity_bytes"]]
    devices = sorted({str(row["cache"]["device"]) for row in steps if row["cache"].get("device") is not None})
    return {
        "device": meta.get("device"),
        "cache_devices": devices,
        "backend": steps[-1]["cache"].get("backend"),
        "revision": steps[-1]["cache"].get("revision"),
        "scheduler_entry": steps[-1]["cache"].get("scheduler_entry"),
        "final_counters": {key: steps[-1]["cache"][key] for key in COUNTERS if key in steps[-1]["cache"]},
        "steps_over_storage_budget": len(over),
        "first_over_storage_budget": cache_snapshot(over[0]) if over else None,
        "peak_storage": cache_snapshot(peak),
        "final_cache": cache_snapshot(steps[-1]),
        "heldout_accuracy": final["heldout_accuracy"],
    }


def verify(buggy_path, fixed_path):
    bm, bs, bf = read_run(buggy_path)
    fm, fs, ff = read_run(fixed_path)
    check(len(bs) == len(fs), "the two runs completed different numbers of RL steps")
    for key in ("initial_policy_digest", "device", "seed", "group_size", "batch_size", "weights_sha256",
                "model", "revision", "legal_token_ids", "temperature", "dtype", "trainable_parameters",
                "frozen_parameters", "native_forward_check", "initial_heldout_accuracy"):
        check(bm.get(key) == fm.get(key), f"meta.{key} differs between runs")
    logprob_max_diff = 0.0
    for b, f in zip(bs, fs):
        here = f"pair at step {b['step']}"
        for key in ("step", "actions", "rewards", "valid_groups", "policy_digest"):
            check(b[key] == f[key], f"{here}: {key} differs")
        for key in ("scene_ids", "prompt_ids", "encoder_calls", "encoder_images"):
            check(b.get(key) == f.get(key), f"{here}: {key} differs")
        delta = max(abs(x-y) for x, y in zip(b["logprobs"], f["logprobs"]))
        check(delta <= 1e-6, f"{here}: action logprobs differ by {delta}")
        logprob_max_diff = max(logprob_max_diff, delta)
        for key in ("grad_norm", "weight_delta"):
            check(math.isclose(b[key], f[key], rel_tol=1e-6, abs_tol=1e-12),
                  f"{here}: {key} differs")
        for key in ("logical_bytes", "capacity_bytes"):
            check(b["cache"][key] == f["cache"][key], f"{here}: cache {key} differs")
        for key in (*COUNTERS, "keys", "lookup_sha256", "backend", "revision", "scheduler_entry", "device",
                    "assembled_embedding_sha256", "assembled_embedding_shape", "assembled_embedding_dtype"):
            check(b["cache"].get(key) == f["cache"].get(key), f"{here}: cache {key} differs")
        for row, arm in ((b, "buggy"), (f, "fixed")):
            check(row["cache"].get("mode") == arm, f"{here}: expected {arm} cache mode")
        check(f["cache"]["storage_bytes"] <= f["cache"]["capacity_bytes"],
              f"{here}: fixed cache still retains more storage than its budget")
    buggy, fixed = report_arm(bm, bs, bf), report_arm(fm, fs, ff)
    check(buggy["steps_over_storage_budget"] > 0, "buggy run never exceeded its physical storage budget")
    counters = buggy["final_counters"]
    for key in ("hits", "evictions", "split_batches", "multi_item_split_batches"):
        if key in counters:
            check(counters[key] > 0, f"the experiment never observed cache {key}")
    if "encoded_images" in counters and "encoder_calls" in counters:
        check(counters["encoded_images"] > counters["encoder_calls"] > 0,
              "no evidence that multiple images were encoded in one batch")
    check(math.isclose(bf["heldout_accuracy"], ff["heldout_accuracy"], rel_tol=0, abs_tol=1e-12),
          "the runs differ on held-out accuracy")
    # Compare at one common time rather than subtracting independently sampled peaks.
    b, f = max(zip(bs, fs), key=lambda pair: pair[0]["cache"]["storage_bytes"] - pair[1]["cache"]["storage_bytes"])
    largest_saving = {
        "step": b["step"],
        "logical_bytes": b["cache"]["logical_bytes"],
        "capacity_bytes": b["cache"]["capacity_bytes"],
        "buggy_storage_bytes": b["cache"]["storage_bytes"],
        "fixed_storage_bytes": f["cache"]["storage_bytes"],
        "bytes_saved": b["cache"]["storage_bytes"] - f["cache"]["storage_bytes"],
    }
    return {
        "steps": len(bs),
        "rollout_actions_per_arm": sum(len(row["actions"]) for row in bs),
        "mixed_reward_groups_per_arm": sum(row["valid_groups"] for row in bs),
        "min_grad_norm": min(row["grad_norm"] for row in bs),
        "min_weight_delta": min(row["weight_delta"] for row in bs),
        "max_abs_action_logprob_diff": logprob_max_diff,
        "matching_assembled_embedding_batches": len(bs),
        "final_policy_sha256": bs[-1]["policy_digest"],
        "largest_paired_storage_saving": largest_saving,
        "buggy": buggy,
        "fixed": fixed,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("buggy", type=Path)
    parser.add_argument("fixed", type=Path)
    parser.add_argument("--out", type=Path, help="also write the verified summary as JSON")
    args = parser.parse_args()
    try:
        check(args.buggy.resolve() != args.fixed.resolve(), "pass two distinct run logs")
        if args.out:
            check(args.out.resolve() not in (args.buggy.resolve(), args.fixed.resolve()),
                  "--out must differ from the input logs")
        summary = verify(args.buggy, args.fixed)
        result = json.dumps(summary, indent=2, allow_nan=False) + "\n"
        if args.out:
            args.out.write_text(result, encoding="utf-8")
        sys.stdout.write(result)
    except (InvalidRun, OSError) as error:
        parser.exit(1, f"verification failed: {error}\n")


if __name__ == "__main__":
    main()
