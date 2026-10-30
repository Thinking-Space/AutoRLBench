import json
import os
from pathlib import Path

import numpy as np


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def evaluate(directory, case, required_score=23.5):
    final_path = directory / "final_metrics.json"
    final = json.loads(final_path.read_text()) if final_path.exists() else {}
    metrics = rows(directory / "metrics_rank0.jsonl")
    training = [row for row in metrics if "grad_norm" in row]
    complete = final.get("step") == 24 and [row["step"] for row in training] == list(range(1, 25))
    nonzero_updates = sum(row["grad_norm"] > 0 for row in training)
    result = {"completed_steps": final.get("step", 0), "nonzero_updates": nonzero_updates}
    if case == "normalization":
        correct = {step: [] for step in range(24)}
        ratios = []
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
            correct[step].append(bool(np.isclose(observed, expected, rtol=5e-4, atol=2e-6)))
            if abs(expected) > 1e-5:
                ratios.append(observed / expected)
        score = sum(len(values) == 2 and all(values) for values in correct.values())
        complete = complete and sum(map(len, correct.values())) == 48 and len(ratios) > 0
        result.update(correct_objective_steps=score, nonzero_comparisons=len(ratios), loss_ratios=ratios)
    else:
        samples = rows(directory / "rollouts_rank0.jsonl")
        logged = [row for row in training if "answer_parse_rate" in row]
        pairs = []
        for row in logged:
            batch = [item for item in samples if item["step"] == row["step"] - 1]
            expected = sum(item["parsed"] is not None for item in batch) / len(batch) if batch else None
            pairs.append({"step": row["step"], "observed": row["answer_parse_rate"], "expected": expected})
        metric_ok = len(pairs) == 12 and all(pair["expected"] is not None and np.isclose(pair["observed"], pair["expected"], atol=1e-6, rtol=1e-6) for pair in pairs)
        other = [row["step"] for row in rows(directory / "metrics_rank1.jsonl") if "grad_norm" in row]
        complete = complete and other == list(range(1, 25)) and metric_ok
        score = final.get("step", 0) if metric_ok else 0
        result.update(metric_preserved=metric_ok, metric_comparisons=pairs, synchronized_steps=score)
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
