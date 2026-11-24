"""Verify a native vocabulary continuation and grade the three held-out sets."""

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import math
from pathlib import Path
from random import Random
import re
import sys


BENCHMARKS = {"AIME24": 30, "AIME25": 30, "AMC23": 40}
DRAWS = 32
ARMS = ("baseline", "fault", "fixed")
TRAIN_METRICS = ("actor/grad_norm", "actor/lr", "rollout_corr/k3_kl", "timing_s/step", "perf/throughput")


def inputs(data):
    import pyarrow.parquet as parquet

    expected = {}
    for benchmark, questions in BENCHMARKS.items():
        rows = parquet.read_table(data / f"{benchmark}.parquet").to_pylist()
        if len(rows) != questions:
            raise ValueError(f"Incorrect number of questions for {benchmark}")
        for row in rows:
            message, = row["prompt"]
            if message["role"] != "user":
                raise ValueError("The native prompt is not a single user message")
            key = (message["content"].strip(), str(row["reward_model"]["ground_truth"]))
            if key in expected or row["data_source"] != benchmark:
                raise ValueError("Held-out dataset identity is ambiguous")
            expected[key] = benchmark
    return expected


def problem(rendered):
    prefix = "<｜User｜>"
    suffix = "<｜Assistant｜>"
    if prefix not in rendered or suffix not in rendered:
        raise ValueError("The rollout prompt is missing its native chat markers")
    return rendered.split(prefix, 1)[1].rsplit(suffix, 1)[0].strip()


def heldout(path, expected, step, *, native_score=None):
    count, correct = Counter(), Counter()
    digest = hashlib.sha256()
    with path.open("rb") as raw:
        for line in raw:
            digest.update(line)
            record = json.loads(line)
            key = (problem(record["input"]), str(record["gts"]))
            grade = record["acc"]
            if key not in expected or record["step"] != step or grade not in (0, 1):
                raise ValueError(f"Unexpected held-out question, step, or grade in {path}")
            if record["score"] != grade or record["reward"] != grade:
                raise ValueError("The native held-out reward and binary grade differ")
            if native_score is not None and native_score(expected[key], record["output"], key[1]) != grade:
                raise ValueError(f"{path}: the native reward disagrees at answer {count.total() + 1}")
            count[key] += 1
            correct[key] += int(grade)
    if count != Counter({key: DRAWS for key in expected}):
        raise ValueError(f"Expected {DRAWS} answers for every original question in {path}")
    return correct, digest.hexdigest()


def training(run, require_gradient, *, start_step, end_step=None, interrupted=False):
    if end_step is None:
        end_step = start_step + 2
    if end_step <= start_step:
        raise ValueError("The final update must follow the starting checkpoint")
    expected_steps = list(range(start_step + 1, end_step + 1))
    prompts = {}
    for step in expected_steps:
        path = run / "rollouts" / f"{step}.jsonl"
        inputs = Counter()
        with path.open(encoding="utf-8") as raw:
            for line in raw:
                row = json.loads(line)
                if row["step"] != step:
                    raise ValueError(f"Unexpected step in {path}")
                inputs[problem(row["input"])] += 1
        if sum(inputs.values()) != 512 or any(count % 8 for count in inputs.values()):
            raise ValueError(f"Original training workload was not completed in {path}")
        prompts[step] = hashlib.sha256(
            json.dumps(sorted(inputs.items()), ensure_ascii=False).encode()
        ).hexdigest()
    pattern = r"([\w/]+):(-?[0-9.]+(?:[eE][+-]?[0-9]+)?)"
    steps = {}
    for line in (run / "launcher.log").read_text(encoding="utf-8").splitlines():
        if "actor/grad_norm:" not in line:
            continue
        values = dict(re.findall(pattern, line))
        step = int(values.pop("step"))
        if interrupted and step > end_step:
            continue
        if step in steps:
            raise ValueError("Duplicate native optimizer step in the console log")
        steps[step] = {k: float(values[k]) for k in TRAIN_METRICS}
    if list(steps) != expected_steps or any(not math.isfinite(v) for d in steps.values() for v in d.values()):
        raise ValueError(f"Expected steps {start_step + 1}–{end_step} with finite native metrics")
    if require_gradient and any(d["actor/grad_norm"] <= 0 for d in steps.values()):
        raise ValueError("Repair updates must receive a real training gradient")
    checkpoint = run / "checkpoints"
    if (checkpoint / "latest_checkpointed_iteration.txt").read_text().strip() != str(end_step):
        raise ValueError(f"Missing native CP{end_step} marker")
    endpoint = checkpoint / f"global_step_{end_step}"
    files = ["data.pt"] + [
        f"actor/{kind}_world_size_2_rank_{rank}.pt"
        for kind in ("model", "optim", "extra_state") for rank in (0, 1)
    ]
    sizes = {name: (endpoint / name).stat().st_size for name in files}
    if not all(sizes.values()):
        raise ValueError(f"CP{end_step} is missing state needed for a complete resume")
    return {"optimizer_steps": steps, "checkpoint_sizes_bytes": sizes, "prompt_multiset_sha256": prompts}


def comparison(left, right, expected):
    deltas = {name: [] for name in BENCHMARKS}
    for key, name in expected.items():
        deltas[name].append(left[key] - right[key])
    result = {}
    for label, groups in [(name, [values]) for name, values in deltas.items()] + [
        ("overall", list(deltas.values()))
    ]:
        n = DRAWS * sum(map(len, groups))
        rng = Random(42)
        boot = sorted(sum(sum(rng.choices(v, k=len(v))) for v in groups) / n for _ in range(10000))
        result[label] = {
            "correct_difference": sum(map(sum, groups)),
            "response_count": n,
            "accuracy_difference": sum(map(sum, groups)) / n,
            "question_bootstrap_95_ci": [boot[250], boot[9749]],
        }
    return result


def verify_checkpoint_lineage(receipts, *, start_step):
    fixed, fault, baseline = (receipts[name] for name in ("fixed", "fault", "baseline"))
    for key in ("checkpoint", "checkpoint_receipt", "config_sha256", "source_manifest_sha256"):
        if fixed[key] != fault[key]:
            raise ValueError(f"Native continuation inputs differ at {key}")
    for key in ("config_sha256", "source_manifest_sha256"):
        if baseline[key] != fixed[key]:
            raise ValueError(f"Native baseline inputs differ at {key}")
    proof = baseline["checkpoint_receipt"]
    origin = Path(fixed["checkpoint"])
    if origin.name != f"global_step_{start_step}":
        raise ValueError("Native checkpoint does not match requested starting step")
    if proof.get("kind") != "validation-only model and extra restore":
        if any(baseline[key] != fixed[key] for key in ("checkpoint", "checkpoint_receipt")):
            raise ValueError("Expected a common native checkpoint")
        return

    # codex: The early evaluation links to the canonical model, RNG and data
    # files. Prove they are the same files after the completed checkpoint is published.
    val = Path(baseline["checkpoint"])
    identity = (val.parent / f"cp{start_step}-validation-identity.json").read_bytes()
    if hashlib.sha256(identity).hexdigest() != proof["identity_sha256"]:
        raise ValueError("Baseline identity differs from the frozen proof")
    expected = {"data.pt"} | {
        f"actor/{kind}_world_size_2_rank_{rank}.pt"
        for kind in ("model", "extra_state") for rank in (0, 1)
    }
    if json.loads(identity)["selected"] != proof["selected"] or set(proof["selected"]) != expected:
        raise ValueError("Baseline checkpoint proof is incomplete")
    payload = (origin.parent / f"cp{start_step}-transfer.json").read_bytes()
    if (proof["original_checkpoint"] != str(origin)
            or hashlib.sha256(payload).hexdigest() != fixed["checkpoint_receipt"]["transfer_receipt_sha256"]):
        raise ValueError("The published training checkpoint differs from the baseline origin")
    transfer = json.loads(payload)
    if not transfer["complete"]:
        raise ValueError("Both training arms require the fully published checkpoint")
    for name, selected in proof["selected"].items():
        if (transfer["sources"][name]["size"] != selected["bytes"]
                or transfer["chunks"][name] != selected["ranges"]
                or not (origin / name).samefile(val / name)
                or (origin / name).stat().st_mtime_ns != selected["mtime_ns"]):
            raise ValueError(f"Baseline and training checkpoint files differ at {name}")


def verify_lj_lineage(receipts, checkpoint, start_step, end_step):
    """Verify the checkpoint bound by the LJ launchers against each live receipt."""
    baseline = receipts["baseline"]
    if checkpoint.name != f"global_step_{start_step}":
        raise ValueError("LJ source checkpoint does not match the starting step")
    for name, receipt in receipts.items():
        is_baseline = name == "baseline"
        mode = "eval" if is_baseline else "train"
        arm = "fixed" if is_baseline else name
        evaluated_step = start_step if is_baseline else end_step
        first_update = None if is_baseline else start_step + 1
        if (receipt["mode"] != mode or receipt["arm"] != arm
                or receipt["evaluated_step"] != evaluated_step
                or receipt["first_update"] != first_update or receipt["gpus"] != 2
                or receipt["checkpoint"] != f"/resume/{checkpoint.name}"):
            raise ValueError(f"Unexpected LJ execution receipt for {name}")
        for field in ("manifest_sha256", "config_sha256"):
            if receipt[field] != baseline[field]:
                raise ValueError(f"LJ {name} differs from the baseline at {field}")
        expected = {"data.pt"} | {
            f"actor/{kind}_world_size_2_rank_{rank}.pt"
            for rank in (0, 1)
            for kind in (("model", "extra_state") if is_baseline else ("model", "extra_state", "optim"))
        }
        identity = receipt["checkpoint_identity"]
        if set(identity) != expected:
            raise ValueError(f"Incomplete LJ input checkpoint identity for {name}")
        for file, recorded in identity.items():
            path = checkpoint / file
            stat = path.stat()
            if stat.st_size != recorded["bytes"] or stat.st_mtime_ns != recorded["mtime_ns"]:
                raise ValueError(f"LJ checkpoint metadata changed at {path}")
            if file == "data.pt" or "/extra_state_" in file:
                if hashlib.sha256(path.read_bytes()).hexdigest() != recorded["sha256"]:
                    raise ValueError(f"LJ checkpoint contents changed at {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("--start-step", type=int, choices=(725, 975), default=975)
    parser.add_argument("--end-step", type=int, help="Final update, defaults to two steps after the start.")
    parser.add_argument("--native-workspace", type=Path,
                        help="Recompute each validation grade with this workspace's native reward.")
    parser.add_argument("--lj-checkpoint", type=Path,
                        help="Host path of the native checkpoint bound by the LJ launchers.")
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--fault", type=Path)
    parser.add_argument("--fixed", type=Path)
    args = parser.parse_args()
    end_step = args.start_step + 2 if args.end_step is None else args.end_step
    if end_step <= args.start_step:
        parser.error("--end-step must follow --start-step")
    if (args.fault is None) != (args.fixed is None):
        parser.error("Provide both training arms or evaluate the baseline alone.")
    arms = ("baseline",) if args.fault is None else ARMS
    expected = inputs(args.data)
    native_score = None
    native_reward_digest = None
    if args.native_workspace is not None:
        sys.path.insert(0, str((args.native_workspace / "framework").resolve()))
        reward_path = args.native_workspace / "reward.py"
        native_reward_digest = hashlib.sha256(reward_path.read_bytes()).hexdigest()
        spec = importlib.util.spec_from_file_location("autorl_native_reward", reward_path)
        reward = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(reward)
        native_score = reward.compute_score
    receipts, grades, runs = {}, {}, {}
    for name in arms:
        step = args.start_step if name == "baseline" else end_step
        run = getattr(args, name)
        completion = json.loads((run / "completed.json").read_text())
        arm = "fixed" if args.lj_checkpoint is not None and name == "baseline" else name
        checkpoint_step = None if name == "baseline" else end_step
        if (completion["arm"] != arm or completion["evaluated_step"] != step
                or completion["checkpoint_step"] != checkpoint_step):
            raise ValueError("Unexpected native completion receipt")
        receipts[name] = json.loads((run / "preflight.json").read_text())
        grades[name], digest = heldout(run / "validation" / f"{step}.jsonl", expected, step,
                                      native_score=native_score)
        correct = Counter()
        for key, grade in grades[name].items():
            correct[expected[key]] += grade
        scores = {
            label: {"correct": correct[label], "total": questions * DRAWS}
            for label, questions in BENCHMARKS.items()
        }
        scores["overall"] = {"correct": sum(correct.values()), "total": len(expected) * DRAWS}
        runs[name] = {"step": step, "heldout": scores, "validation_sha256": digest}
        if name != "baseline":
            runs[name]["training"] = training(run, require_gradient=name == "fixed",
                                              start_step=args.start_step, end_step=end_step)
    if args.lj_checkpoint is not None:
        verify_lj_lineage(receipts, args.lj_checkpoint, args.start_step, end_step)
        if ((args.baseline / "checkpoints").exists()
                or list((args.baseline / "rollouts").glob("*.jsonl"))
                or "actor/grad_norm:" in (args.baseline / "launcher.log").read_text(encoding="utf-8")):
            raise ValueError("LJ validation-only run entered training")
    if len(arms) == 1:
        baseline = receipts["baseline"]
        if args.lj_checkpoint is None:
            if (baseline["arm"] != "baseline" or baseline["step"] != args.start_step
                    or baseline["first_update"] is not None
                    or list((args.baseline / "rollouts").glob("*.jsonl"))):
                raise ValueError("Expected a validation-only native run")
            proof = baseline["checkpoint_receipt"]
            if proof.get("kind") == "validation-only model and extra restore":
                checkpoint = Path(baseline["checkpoint"])
                identity = (checkpoint.parent / f"cp{args.start_step}-validation-identity.json").read_bytes()
                if (hashlib.sha256(identity).hexdigest() != proof["identity_sha256"]
                        or json.loads(identity)["selected"] != proof["selected"]
                        or checkpoint.name != f"global_step_{args.start_step}"):
                    raise ValueError("Baseline checkpoint identity differs from the restoration receipt")
        report = {"runs": runs, "baseline_preflight_sha256":
                  hashlib.sha256((args.baseline / "preflight.json").read_bytes()).hexdigest()}
        if native_score is not None:
            report["native_rescore"] = {"reward_sha256": native_reward_digest,
                                        "answers_verified": len(expected) * DRAWS}
        print(json.dumps(report, indent=2))
        return
    if args.lj_checkpoint is None:
        verify_checkpoint_lineage(receipts, start_step=args.start_step)
    if receipts["baseline"]["actor_sha256"] != receipts["fixed"]["actor_sha256"]:
        raise ValueError("Common starting evaluation did not use the fixed evaluation workspace")
    if receipts["fault"]["actor_sha256"] == receipts["fixed"]["actor_sha256"]:
        raise ValueError("The actor repair was not isolated to the fixed arm")
    if runs["fault"]["training"]["prompt_multiset_sha256"] != runs["fixed"]["training"]["prompt_multiset_sha256"]:
        raise ValueError("The native arms did not train on the same prompts")
    pairs = {
        f"{left}_minus_{right}": comparison(grades[left], grades[right], expected)
        for left, right in (("fixed", "baseline"), ("fault", "baseline"), ("fixed", "fault"))
    }
    result = {"runs": runs, "paired_question_estimates": pairs}
    if native_score is not None:
        result["native_rescore"] = {"reward_sha256": native_reward_digest,
                                    "answers_verified": len(runs) * len(expected) * DRAWS}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
