import json
import os
from pathlib import Path

import numpy as np


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def runner_integrity():
    """The rerun stage executes the workspace copies of the runner, so grading compares
    them against the verifier's pristine copies. A replay of packaged evidence by hand
    runs without the stage environment and skips the check."""
    workspace = os.environ.get("AUTORLBENCH_WORKSPACE", "")
    verifier = os.environ.get("AUTORLBENCH_VERIFIER", "")
    if not workspace or not verifier:
        return None, {}
    pairs = {
        "run_training.sh": (Path(workspace) / "run_training.sh", Path(verifier) / "run_training.sh"),
        "train.py": (Path(workspace) / "train.py", Path(verifier) / "resources" / "train.py"),
    }
    details = {}
    for name, (staged, pristine) in pairs.items():
        try:
            details[name] = staged.read_bytes() == pristine.read_bytes()
        except OSError:
            details[name] = False
    return all(details.values()), details


def evaluate(directory, case, required_score=23.5):
    final_path = directory / "final_metrics.json"
    final = json.loads(final_path.read_text()) if final_path.exists() else {}
    metrics = rows(directory / "metrics_rank0.jsonl")
    training = [row for row in metrics if "grad_norm" in row]
    complete = final.get("step") == 24 and [row["step"] for row in training] == list(range(1, 25))
    nonzero_updates = sum(row["grad_norm"] > 0 for row in training)
    integrity_ok, integrity_details = runner_integrity()
    result = {
        "completed_steps": final.get("step", 0),
        "nonzero_updates": nonzero_updates,
        "runner_integrity": integrity_ok if integrity_ok is not None else "not checked (no stage environment)",
        "runner_integrity_files": integrity_details,
    }
    if integrity_ok is False:
        complete = False
    if case == "normalization":
        correct = {step: [] for step in range(24)}
        losses = {step: [] for step in range(24)}
        ratios = []
        windows = {}
        metadata_ok = True
        for path in sorted((directory / "token_records").glob("*.npz")):
            record = np.load(path)
            ratio = np.exp(record["logps"].astype(np.float64) - record["old_logps"])
            advantage = record["advantages"][:, None]
            clipped = np.clip(ratio, 1 - record["clip_low"], 1 + record["clip_high"])
            numerator = (-np.minimum(ratio * advantage, clipped * advantage) * record["mask"]).sum()
            denominator = max(float(record["generation_token_count"]), 1.0) / float(record["world_size"])
            denominator *= float(record["accumulation_steps"]) / float(record["generation_steps"])
            expected = numerator / denominator
            observed = float(record["loss"])
            step = int(path.name.split("_")[0][4:])
            batch = int(path.name.split("_")[1][5:])
            correct[step].append(bool(np.isclose(observed, expected, rtol=5e-4, atol=2e-6)))
            losses[step].append(observed)
            if abs(expected) > 1e-5:
                ratios.append(observed / expected)
            # The reference denominator is built from run-shape metadata, so pin it to the
            # shape the runner fixes (1 process, 4 microbatches per generation batch, 2 per
            # optimizer step): a record-only edit of any of these fields would reconcile a
            # never-repaired loss with the reference.
            if float(record["world_size"]) != 1.0 or float(record["generation_steps"]) != 4.0 or float(record["accumulation_steps"]) != 2.0:
                metadata_ok = False
            windows[batch] = (int(record["mask"].sum()), float(record["generation_token_count"]))
        # Each generation batch spans 4 consecutive microbatches, so their token counts
        # add up to the recorded generation_token_count; forging that field alone breaks
        # the sum.
        window_ok = len(windows) == 48
        window_matches = 0
        for start in range(0, 48, 4):
            group = [windows[b] for b in range(start, start + 4) if b in windows]
            if len(group) == 4 and len({gtc for _, gtc in group}) == 1 and abs(sum(m for m, _ in group) - group[0][1]) < 0.5:
                window_matches += 1
            else:
                window_ok = False
        # The trainer's own step loss anchors the records: it is the sum of that step's
        # two microbatch losses, so rewriting the npz loss field without also forging the
        # logged metrics breaks the agreement.
        logged_loss = {row["step"]: row["loss"] for row in training if "loss" in row}
        anchor_matches = sum(
            step + 1 in logged_loss
            and len(losses[step]) == 2
            and bool(np.isclose(logged_loss[step + 1], sum(losses[step]), rtol=5e-4, atol=2e-6))
            for step in range(24)
        )
        score = sum(len(values) == 2 and all(values) for values in correct.values())
        complete = complete and sum(map(len, correct.values())) == 48 and len(ratios) > 0 and metadata_ok and window_ok and anchor_matches == 24
        result.update(
            correct_objective_steps=score,
            nonzero_comparisons=len(ratios),
            loss_ratios=ratios,
            run_shape_pinned=metadata_ok,
            generation_windows_consistent=window_matches,
            loss_anchor_matches=anchor_matches,
        )
    else:
        samples = rows(directory / "rollouts_rank0.jsonl")
        logged = [row for row in training if "answer_parse_rate" in row]
        pairs = []
        for row in logged:
            batch = [item for item in samples if item["step"] == row["step"] - 1]
            expected = sum(item["parsed"] is not None for item in batch) / len(batch) if batch else None
            pairs.append({"step": row["step"], "observed": row["answer_parse_rate"], "expected": expected})
        metric_ok = len(pairs) == 12 and all(pair["expected"] is not None and np.isclose(pair["observed"], pair["expected"], atol=1e-6, rtol=1e-6) for pair in pairs)
        rank1 = rows(directory / "metrics_rank1.jsonl")
        other = [row["step"] for row in rank1 if "grad_norm" in row]
        # The upstream repair gathers the union of pending metric keys across ranks, so
        # rank 1 logs answer_parse_rate as well, with the same reduced value. Rank-0-only
        # logging is the defect; a repair that stops at rank 0 fails here.
        expected_by_step = {pair["step"]: pair["expected"] for pair in pairs}
        rank1_pairs = [
            {"step": row["step"], "observed": row["answer_parse_rate"], "expected": expected_by_step.get(row["step"])}
            for row in rank1 if "answer_parse_rate" in row
        ]
        # The 12 rank-1 events must land on the same steps as rank 0's, not merely be 12
        # values that match some expected step: concentrating or duplicating events on a
        # subset would otherwise pass.
        rank1_steps = sorted(pair["step"] for pair in rank1_pairs)
        metric_rank1_ok = len(rank1_pairs) == 12 and rank1_steps == sorted(expected_by_step) and all(
            pair["expected"] is not None and np.isclose(pair["observed"], pair["expected"], atol=1e-6, rtol=1e-6)
            for pair in rank1_pairs
        )
        complete = complete and other == list(range(1, 25)) and metric_ok and metric_rank1_ok
        score = final.get("step", 0) if metric_ok and metric_rank1_ok else 0
        result.update(
            metric_preserved=metric_ok,
            metric_rank1_events=len(rank1_pairs),
            metric_both_ranks=metric_rank1_ok,
            metric_comparisons=pairs,
            rank1_metric_comparisons=rank1_pairs,
            synchronized_steps=score,
        )
    result["score_value"] = score
    passed = bool(complete and nonzero_updates > 0 and score >= required_score)
    return {"passed": passed, "score": float(passed), "metrics": result}


def main():
    results = Path(os.environ["AUTORLBENCH_RESULTS"])
    spec = json.loads((Path(__file__).resolve().parents[1] / "resources/grader_spec.json").read_text())
    outcome = evaluate(results / "experiment_logs", spec["case"], float(spec["required_final_score"]))
    for name in ["experiment_metrics.json", "metrics.json"]:
        (results / name).write_text(json.dumps(outcome["metrics"], indent=2) + "\n")
    (results / "result.json").write_text(json.dumps(outcome, indent=2) + "\n")
    print(json.dumps(outcome))
    return 0 if outcome["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
