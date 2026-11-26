#!/usr/bin/env python3
"""Build a deterministic, balanced train/test trajectory set for KL probes.

The train half is sampled from the exact 9,600-example replay set shared by
the 150-step 1.5B runs.  It contains two examples from each Skywork difficulty
1..15.  The fixed continuations are the base-FP teacher rollouts used by the
off-policy arm.

The test half contains ten examples each from AIME 2024, AMC23, and
OlympiadBench.  Their fixed continuations are sample 0 from the existing
base-FP benchmark reports.  Prompts are rebuilt through the same evaluator
instruction and model chat template used by those reports.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import pyarrow.parquet as pq
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

BASE_MODEL = Path("/historical/shared/qat/models/Nemotron-Research-Reasoning-Qwen-1.5B-brorl")
REPLAY = Path("/historical/shared/qat/data/processed/skywork_replay_subset_9600.parquet")
TEACHER = Path("/historical/shared/qat/data/rollouts/teacher-1p5b-9600/trajectories.parquet")
AIME_DATA = Path("/historical/shared/qat/data/raw/aime_2024/data/train-00000-of-00001.parquet")
AIME_REPORT = Path(
    "/historical/shared/qat/runs/qat-eval-1p5b-base-aime24-avg16/eval/"
    "base_fp_aime24_avg16_auto.json"
)
AMC_DATA = Path(
    "/historical/shared/qat/vendor/simpleRL-reason/examples/simplelr_math_eval/data/amc23/test.jsonl"
)
AMC_REPORT = Path(
    "/historical/shared/qat/runs/qat-eval-1p5b-multibench/amc23/eval/base-fp.json"
)
OLY_DATA = Path(
    "/historical/shared/qat/vendor/simpleRL-reason/examples/simplelr_math_eval/data/"
    "olympiadbench/test.jsonl"
)
OLY_REPORT = Path(
    "/historical/shared/qat/runs/qat-eval-1p5b-multibench/olympiad_bench/eval/base-fp.json"
)


def _stable_rank(seed: int, text: str) -> str:
    return hashlib.sha256(f"{seed}:{text}".encode()).hexdigest()


def _quantiles_evenly(items: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    """Take deterministic evenly spaced order statistics, including endpoints."""
    if len(items) < count:
        raise ValueError(f"need {count} eligible rows, found {len(items)}")
    if count == 1:
        return [items[len(items) // 2]]
    indices = [round(i * (len(items) - 1) / (count - 1)) for i in range(count)]
    if len(set(indices)) != count:
        raise RuntimeError(f"even selection produced duplicate indices: {indices}")
    return [items[i] for i in indices]


def _train_samples(seed: int) -> list[dict[str, Any]]:
    replay = pq.read_table(REPLAY, columns=["example_id", "data_source", "extra_info"]).to_pylist()
    teacher_rows = pq.read_table(
        TEACHER,
        columns=[
            "example_id",
            "data_source",
            "prompt_text",
            "prompt_token_ids",
            "response_token_ids",
            "response_tokens",
            "finish_reason",
            "truncated",
        ],
    ).to_pylist()
    teacher = {row["example_id"]: row for row in teacher_rows}
    if len(teacher) != len(teacher_rows):
        raise RuntimeError("teacher trajectories contain duplicate example_id values")

    by_difficulty: dict[int, list[dict[str, Any]]] = {d: [] for d in range(1, 16)}
    for row in replay:
        difficulty = (row.get("extra_info") or {}).get("model_difficulty") or {}
        difficulty = difficulty.get("DeepSeek-R1-Distill-Qwen-7B")
        traj = teacher.get(row["example_id"])
        if difficulty not in by_difficulty or traj is None:
            continue
        if traj.get("truncated") or traj.get("finish_reason") != "stop":
            continue
        if not traj.get("response_token_ids"):
            continue
        merged = dict(traj)
        merged["difficulty"] = int(difficulty)
        by_difficulty[int(difficulty)].append(merged)

    selected: list[dict[str, Any]] = []
    for difficulty in range(1, 16):
        rows = sorted(
            by_difficulty[difficulty],
            key=lambda row: _stable_rank(seed, str(row["example_id"])),
        )
        if len(rows) < 2:
            raise RuntimeError(f"difficulty {difficulty} has only {len(rows)} eligible rows")
        # Prefer two different source datasets when the stratum permits it.
        first = rows[0]
        second = next(
            (row for row in rows[1:] if row["data_source"] != first["data_source"]),
            rows[1],
        )
        for row in (first, second):
            selected.append(
                {
                    "split": "train",
                    "stratum": f"difficulty_{difficulty:02d}",
                    "benchmark": "skywork_replay",
                    "example_id": row["example_id"],
                    "data_source": row["data_source"],
                    "difficulty": difficulty,
                    "prompt_text": row["prompt_text"],
                    "prompt_token_ids": [int(x) for x in row["prompt_token_ids"]],
                    "response_token_ids": [int(x) for x in row["response_token_ids"]],
                    "recorded_response_tokens": int(row["response_tokens"]),
                    "finish_reason": row["finish_reason"],
                    "truncated": bool(row["truncated"]),
                    "trajectory_source": str(TEACHER),
                }
            )
    return selected


def _test_samples(tokenizer, seed: int) -> list[dict[str, Any]]:
    from scripts.eval_aime import DEFAULT_INSTRUCTION as AIME_INSTRUCTION, load_aime_records
    from scripts.eval_math_benchmarks import DEFAULT_INSTRUCTION as MATH_INSTRUCTION, load_math_records

    specs = [
        ("aime_math", AIME_DATA, AIME_REPORT, AIME_INSTRUCTION),
        ("amc23", AMC_DATA, AMC_REPORT, MATH_INSTRUCTION),
        ("olympiad_bench", OLY_DATA, OLY_REPORT, MATH_INSTRUCTION),
    ]
    selected: list[dict[str, Any]] = []
    for benchmark, data_path, report_path, instruction in specs:
        if benchmark == "aime_math":
            records = load_aime_records(data_path)
        else:
            records = load_math_records(data_path, benchmark)
        report = json.loads(report_path.read_text())
        predictions = report["predictions"]
        if len(records) != len(predictions):
            raise RuntimeError(
                f"{benchmark}: data/report count mismatch {len(records)} != {len(predictions)}"
            )

        eligible: list[dict[str, Any]] = []
        for position, (record, pred) in enumerate(zip(records, predictions)):
            samples = pred.get("samples") or []
            if not samples:
                continue
            sample = samples[0]
            if sample.get("truncated") or sample.get("finish_reason") != "stop":
                continue
            response = str(sample.get("response") or "")
            if not response:
                continue
            problem = str(record["problem"])
            messages = [{"role": "user", "content": f"{instruction}\n\n{problem}"}]
            prompt_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
            response_ids = tokenizer(response, add_special_tokens=False)["input_ids"]
            if not response_ids:
                continue
            eligible.append(
                {
                    "position": position,
                    "split": "test",
                    "stratum": benchmark,
                    "benchmark": benchmark,
                    "example_id": f"{benchmark}:{pred.get('id', position)}",
                    "data_source": str(data_path),
                    "difficulty": record.get("difficulty"),
                    "prompt_text": prompt_text,
                    "prompt_token_ids": [int(x) for x in prompt_ids],
                    "response_token_ids": [int(x) for x in response_ids],
                    "recorded_response_tokens": int(sample.get("response_tokens", len(response_ids))),
                    "retokenized_response_tokens": len(response_ids),
                    "finish_reason": sample.get("finish_reason"),
                    "truncated": bool(sample.get("truncated")),
                    "trajectory_source": str(report_path),
                }
            )
        # Stable pseudo-random ordering followed by evenly spaced order
        # statistics avoids taking a contiguous slice of benchmark ordering.
        eligible.sort(key=lambda row: _stable_rank(seed, str(row["example_id"])))
        selected.extend(_quantiles_evenly(eligible, 10))
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--model", type=Path, default=BASE_MODEL)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    samples = _train_samples(args.seed) + _test_samples(tokenizer, args.seed)
    counts = {split: sum(row["split"] == split for row in samples) for split in ("train", "test")}
    if counts != {"train": 30, "test": 30}:
        raise RuntimeError(f"unexpected split counts: {counts}")

    payload = {
        "schema_version": 1,
        "seed": args.seed,
        "model_tokenizer": str(args.model),
        "selection": {
            "train": "2 per DeepSeek-R1-Distill-Qwen-7B difficulty, levels 1..15",
            "test": "10 each from AIME 2024, AMC23, OlympiadBench",
            "trajectory": "base-FP fixed continuation; full response; sample 0 for test reports",
        },
        "counts": counts,
        "samples": samples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "counts": counts}, ensure_ascii=False))


if __name__ == "__main__":
    main()
