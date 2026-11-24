"""Grade fresh native training output after both reported defects are repaired."""

import json
import math
import os
from pathlib import Path
import re
from statistics import median


METRIC = re.compile(r"(?:^|\s)(actor/pg_clipfrac|actor/grad_norm|critic/score/mean):([^\s]+)")
STEP = re.compile(r"\bstep:(\d+) - ")
SPEC = Path(__file__).resolve().parents[1] / "resources/grader_spec.json"


def training_metrics(log_paths, final_step):
    rows = {}
    for path in log_paths:
        with path.open(encoding="utf-8", errors="replace") as stream:
            for line in stream:
                step = STEP.search(line)
                if not step:
                    continue
                fields = {name: float(raw) for name, raw in METRIC.findall(line)}
                if not fields:
                    continue
                index = int(step[1])
                if index in rows or len(fields) != 3 or not all(map(math.isfinite, fields.values())):
                    raise ValueError(f"Invalid or duplicate native metrics at step {index}")
                rows[index] = fields
    if list(rows) != list(range(1, final_step + 1)):
        raise ValueError(f"Expected all {final_step} native training steps in order")
    return rows


def score_training(rows, spec):
    positives, clip = 0, []
    for step in range(spec["scored_from_step"], spec["final_step"] + 1):
        row = rows[step]
        positive = (row["critic/score/mean"] + 1) * spec["responses_per_step"] / 2
        if not positive.is_integer() or not 0 <= positive <= spec["responses_per_step"]:
            raise ValueError(f"Invalid reward count at step {step}")
        fraction = row["actor/pg_clipfrac"]
        if not 0 <= fraction <= 1 or row["actor/grad_norm"] < 0:
            raise ValueError(f"Invalid actor metrics at step {step}")
        positives += int(positive)
        clip.append(fraction)
    middle = median(clip)
    passed = (positives >= spec["minimum_positive_rewards"]
              and middle >= spec["minimum_median_ppo_clip_fraction"])
    return dict(passed=passed, score=float(passed), positive_rewards=positives, median_ppo_clip_fraction=middle)


def grade(results, workspace, spec):
    if not (workspace / "report.md").read_text(encoding="utf-8").strip():
        raise ValueError("Missing repair explanation in report.md")
    logs = results / "experiment_logs"
    segments = [logs / name for name in ("steps-01-08", "steps-09-30")]
    if any((segment / "exit_code").read_text().strip() != "0" for segment in segments):
        raise ValueError("Native training did not exit successfully")
    checkpoints = segments[-1] / "run/checkpoints"
    step = spec["final_step"]
    if (checkpoints / "latest_checkpointed_iteration.txt").read_text().strip() != str(step):
        raise ValueError(f"Expected native checkpoint marker {step}")
    checkpoint = checkpoints / f"global_step_{step}"
    parts = [checkpoint / "data.pt"]
    parts.extend(checkpoint / "actor" / f"{kind}_world_size_4_rank_{rank}.pt"
                 for kind in ("model", "optim", "extra_state") for rank in range(4))
    if any(not part.is_file() or part.stat().st_size == 0 for part in parts):
        raise ValueError("Incomplete final native checkpoint")
    evaluation = json.loads((logs / "evaluation/E30.json").read_bytes())
    correct = evaluation["correct"]
    questions = evaluation["predictions"]
    if (evaluation["benchmark"] != "aime_math"
            or evaluation["kv_cache_dtype_requested"] != "turboquant_k2v3_nc"
            or evaluation["tensor_parallel_size"] != 1
            or evaluation["sampling"] != {
                "temperature": 0.7, "top_p": 0.95, "n": 8, "max_tokens": 32768,
                "stop_token_ids": [151643], "seed": 42,
            }
            or type(correct) is not int or evaluation["total_samples"] != 240 or len(questions) != 30
            or len({q["id"] for q in questions}) != 30
            or any(len(q["samples"]) != 8 or {s["sample"] for s in q["samples"]} != set(range(8))
                   for q in questions)
            or any(s["correct"] not in (0, 1) for q in questions for s in q["samples"])
            or correct != sum(s["correct"] for q in questions for s in q["samples"])
            or not 0 <= correct <= 240):
        raise ValueError("Expected the full AIME24 evaluation at step 30")
    rows = training_metrics([segment / "run/train.stdout" for segment in segments], step)
    report = score_training(rows, spec)
    passed = report["passed"] and correct > spec["initial_aime24_correct"]
    report.update(passed=passed, score=float(passed), aime24_correct=correct)
    return report


def main():
    results = Path(os.environ.get("AUTORLBENCH_RESULTS", "/results"))
    workspace = Path(os.environ.get("AUTORLBENCH_WORKSPACE", "/workspace"))
    try:
        report = grade(results, workspace, json.loads(SPEC.read_bytes()))
        code = 0 if report["passed"] else 1
    except (OSError, ValueError, KeyError, TypeError) as error:
        report = dict(passed=False, score=0.0, errors=[str(error)])
        code = 2
    (results / "result.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
