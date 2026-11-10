#!/usr/bin/env python3
"""Re-score the fresh run's training rollout dumps with an external judge and assign a binary score.

The task's defect is in the training data: a large fraction of train rows carry
natively wrong expected outputs, so a rollout that implements the correct
algorithm is still scored 0 by the binary all-tests-pass reward. The logged
training reward therefore cannot separate a repaired run from a defective one
by rising alone; what separates them is whether correct solutions keep being
punished.

The grader reads `experiment_logs/rollout_data/<step>.jsonl` (the trainer's
per-step rollout dumps: input, output, score, step) for the last
`last_k_dumps` steps. For every dumped response whose problem is in the
corrected-outputs table (resources/corrected_outputs.json, built during the
task's data purification from the problems' own agreeing bundled solutions)
and whose logged score is 0, it executes the response's code on that
problem's recorded failing test and compares the actual output with the
corrected output's stored sha256 digest. A match is a "correct solution
scored 0" sample. The metric
is the maximum over the last `last_k_dumps` steps of
(correct-but-zero samples / rows in that step's dump).

The defective arm's rate is materially above zero and does not fall with
training; a run trained on clean rows only has none of these problems left,
so its rate is ~0 (residual: whatever wrong rows a partial repair missed).
Pass requires metric <= required_final_score_max.

The rerun is anchored: live grading byte-compares the workspace launch script
and `verl/utils/reward_score/taco.py` against the verifier's pristine copies,
pins `taco_verl/test.parquet` by sha256, requires the repaired
`taco_verl/train.parquet` to keep at least `min_train_rows` rows, and checks
that no training prompt equals a held-out prompt (training on the validation
set fails). Run-reality evidence is the dump sequence itself: one file per
training step 1..final_step, each with `min_rows` rows whose `step` field
matches its file -- never the score values. A replay of packaged evidence by
hand runs without the stage environment and skips the workspace checks.

Executed response code is untrusted: every execution is a fresh `python3 -I`
subprocess with CPU/AS/NPROC/FSIZE rlimits and a wall-clock kill; it runs in
the verifier container exactly like the training run's own reward executions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import resource
import subprocess
import sys
import tempfile
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

MAX_FILE_BYTES = 512 * 1024 * 1024
REQUIRED_SPEC_KEYS = {
    "schema_version",
    "task_id",
    "metric_key",
    "final_step",
    "historical_final_baseline",
    "required_final_score",
    "log_roots",
    "dump_subdir",
    "min_rows",
    "pinned_sha256",
}

# The prompt template the training parquets carry (byte-verified against the
# sibling task's published data; the literal two-character \n before 'Use
# Standard Input format' is in the upstream data generation).
PREFIX = (
    "Please write a python program to address the following QUESTION. "
    "Your ANSWER should be in a code block format like this: "
    "```python # Write your code here ```. \nQUESTION:\n"
)
STDIN_SUFFIX = " \\nUse Standard Input format\nANSWER:\n"
FENCE_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def load_spec(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    missing = REQUIRED_SPEC_KEYS - set(value)
    if missing:
        raise ValueError(f"grader_spec.json missing keys: {sorted(missing)}")
    if int(value["schema_version"]) != 3:
        raise ValueError("grader_spec.json schema_version must be 3")
    value["final_step"] = int(value["final_step"])
    value["min_rows"] = int(value["min_rows"])
    return value


def extract_question(input_field: str) -> str | None:
    if not input_field.startswith("system") and PREFIX not in input_field:
        start = input_field.find(PREFIX)
        if start < 0:
            return None
    start = input_field.find(PREFIX)
    if start < 0:
        return None
    rest = input_field[start + len(PREFIX):]
    end = rest.find(STDIN_SUFFIX)
    if end < 0:
        return None
    return rest[:end]


def extract_code(response: str) -> str | None:
    """Mirror taco.py's post_process: first fenced block, language tag stripped."""
    blocks = FENCE_RE.findall(response or "")
    if blocks:
        return blocks[0]
    if "```" in (response or ""):
        return response.split("```")[1]
    return None


def _limits() -> None:
    """Best-effort rlimits for the untrusted-response subprocess.

    Each limit is applied independently: some platforms (macOS) reject
    RLIMIT_AS/RLIMIT_NPROC outright; the verifier container (Linux) applies
    all of them. The wall-clock kill and `python3 -I` isolation hold either
    way.
    """
    for res_, limit in (
        (resource.RLIMIT_AS, (2 << 30, 2 << 30)),
        (resource.RLIMIT_CPU, (120, 120)),
        (resource.RLIMIT_NPROC, (128, 128)),
        (resource.RLIMIT_FSIZE, (32 << 20, 32 << 20)),
    ):
        try:
            resource.setrlimit(res_, limit)
        except (OSError, ValueError):
            pass


def run_response(code: str, stdin_text: str, timeout_s: int) -> list[str] | None:
    """Execute one response on one test input, isolated; None on any failure."""
    with tempfile.TemporaryDirectory(prefix="taco-grade-") as tmp:
        sol = Path(tmp) / "sol.py"
        sol.write_text(code, encoding="utf-8")
        try:
            proc = subprocess.run(
                [sys.executable, "-I", str(sol)],
                input=stdin_text,
                capture_output=True,
                text=True,
                timeout=timeout_s,
                preexec_fn=_limits,
            )
        except subprocess.TimeoutExpired:
            return None
        if proc.returncode != 0:
            return None
        return proc.stdout.splitlines()


def training_reality(results: Path, spec: dict[str, Any]) -> tuple[bool, dict[str, Any], str | None]:
    """The scored dumps are written by the trainer itself, so a run that actually
    executed the shipped launch script leaves one rollout dump per training step
    1..final_step, each with the fixed batch of `min_rows` rows whose `step`
    field matches the file. Only that shape is checked; the scores are judged
    separately by re-execution, never read as ground truth."""
    final_step = spec["final_step"]
    details: dict[str, Any] = {"dump_files": 0, "rows_last": 0, "steps_continuous": False}
    base = None
    for root in spec["log_roots"]:
        cand = results / root / spec["dump_subdir"]
        if cand.is_dir():
            base = cand
            break
    if base is None:
        return False, details, (
            f"missing {spec['dump_subdir']}/ under {spec['log_roots']} (the launch script's "
            "trainer.rollout_data_dir as shipped writes one dump per training step)"
        )
    indices = []
    for path in base.glob("*.jsonl"):
        try:
            indices.append(int(path.stem))
        except ValueError:
            continue
    indices.sort()
    details["dump_files"] = len(indices)
    details["steps_continuous"] = indices == list(range(1, final_step + 1))
    if not details["steps_continuous"]:
        return False, details, (
            f"{base.relative_to(results)} does not show a complete run: expected dumps for "
            f"steps 1..{final_step}, found {len(indices)} files"
            + (f" (max step {indices[-1]})" if indices else "")
        )
    return True, details, None


def read_dump(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict) and "input" in row and "output" in row:
                rows.append(row)
    return rows


def stage_integrity(spec: dict[str, Any]) -> tuple[bool | None, dict[str, Any]]:
    """The rerun stage executes the agent's workspace copy of the launch script on
    the agent's data, so live grading compares: the launch script and taco.py
    byte-for-byte against the pristine copies in resources/, test.parquet by
    sha256, train.parquet by row floor, and train/test prompt disjointness. A
    replay of packaged evidence by hand runs without the stage environment and
    skips the check."""
    workspace = os.environ.get("AUTORLBENCH_WORKSPACE", "")
    verifier = os.environ.get("AUTORLBENCH_VERIFIER", "")
    if not workspace or not verifier:
        return None, {}
    ws = Path(workspace)
    details: dict[str, Any] = {}
    try:
        details["launch_script_bytes"] = (
            (ws / "examples/grpo_trainer/run_qwen3_4b_instruct_2507_taco_fsdp.sh").read_bytes()
            == (Path(verifier) / "resources/run_qwen3_4b_instruct_2507_taco_fsdp.sh").read_bytes()
        )
    except OSError:
        details["launch_script_bytes"] = False
    try:
        details["taco_py_bytes"] = (
            (ws / "verl/utils/reward_score/taco.py").read_bytes()
            == (Path(verifier) / "resources/taco.py").read_bytes()
        )
    except OSError:
        details["taco_py_bytes"] = False
    for rel, expected in spec.get("pinned_sha256", {}).items():
        try:
            digest = hashlib.sha256((ws / rel).read_bytes()).hexdigest()
        except OSError:
            digest = ""
        details[f"pinned:{rel}"] = digest == expected
    # train.parquet floor + train/test disjointness (the repair may and must
    # change train.parquet, so it is not pinned; it just cannot shrink to a
    # token set or become the validation set).
    try:
        import pyarrow.parquet as pq

        train = pq.read_table(ws / "taco_verl/train.parquet", columns=["prompt"])
        test = pq.read_table(ws / "taco_verl/test.parquet", columns=["prompt"])
        train_prompts = {json.dumps(p, sort_keys=True) for p in train["prompt"].to_pylist()}
        test_prompts = {json.dumps(p, sort_keys=True) for p in test["prompt"].to_pylist()}
        details["train_rows"] = len(train_prompts)
        details["train_rows_ok"] = len(train_prompts) >= int(spec.get("min_train_rows", 0))
        details["train_test_disjoint"] = not (train_prompts & test_prompts)
    except Exception as exc:  # pyarrow missing or parquet unreadable
        details["train_rows_ok"] = False
        details["train_parquet_error"] = repr(exc)
    return all(v for v in details.values() if isinstance(v, bool)), details


def grade(spec: dict[str, Any], results: Path, resources: Path) -> int:
    final_step = spec["final_step"]
    required_max = Decimal(str(spec["required_final_score_max"]))
    errors: list[str] = []
    exec_timeout = int(spec.get("exec_timeout_s", 60))
    last_k = int(spec.get("last_k_dumps", 5))

    corrected = json.loads((resources / "corrected_outputs.json").read_text(encoding="utf-8"))
    dump_dir = None
    for root in spec["log_roots"]:
        cand = results / root / spec["dump_subdir"]
        if cand.is_dir():
            dump_dir = cand
            break
    if dump_dir is None:
        print(json.dumps({"errors": [f"missing {spec['dump_subdir']}/ under {spec['log_roots']}"]}))
        return 1

    steps = sorted(
        (int(p.stem) for p in dump_dir.glob("*.jsonl") if p.stem.isdigit()), reverse=True
    )[:last_k]
    if not steps:
        print(json.dumps({"errors": ["no rollout dumps found"]}))
        return 1

    rates = {}
    sample_detail = []
    for step in steps:
        path = dump_dir / f"{step}.jsonl"
        rows = read_dump(path)
        if len(rows) < spec["min_rows"]:
            errors.append(
                f"{path.name} has {len(rows)} rows, fewer than the {spec['min_rows']} required"
            )
            continue
        judged = 0
        for i, row in enumerate(rows):
            if row.get("score") not in (0, 0.0):
                continue
            question = extract_question(row.get("input", ""))
            if question is None:
                continue
            qhash = hashlib.sha256(question.encode("utf-8", "replace")).hexdigest()
            entry = corrected.get(qhash)
            if not entry:
                continue
            code = extract_code(row.get("output", ""))
            stdin_text = entry.get("fail_input") or ""
            corrected_sha = entry.get("corrected_output_sha256")
            # The corrected output is stored as the sha256 of its canonical
            # JSON line list (the same hash of the executed response's stdout
            # lines). Empty outputs are never present: an empty-vs-empty match
            # would vacuously "confirm" any response that prints nothing, and
            # the strict selection produced none.
            if code is None or not corrected_sha:
                continue
            actual = run_response(code, stdin_text, exec_timeout)
            ok = actual is not None and hashlib.sha256(
                json.dumps(actual).encode("utf-8")
            ).hexdigest() == corrected_sha
            if ok:
                judged += 1
                if len(sample_detail) < 8:
                    sample_detail.append(
                        {"step": step, "row": i, "name": entry.get("name"),
                         "fail_index": entry.get("fail_index")}
                    )
        rates[step] = judged / len(rows)

    if not rates:
        print(json.dumps({"errors": errors or ["no scorable dumps"]}))
        return 1
    metric = max(rates.values())

    _, reality_details, reality_error = training_reality(results, spec)
    if reality_error:
        errors.append(reality_error)
    integrity_ok, integrity_details = stage_integrity(spec)
    if integrity_ok is False:
        errors.append(
            "workspace integrity failed: "
            + json.dumps({k: v for k, v in integrity_details.items()}, sort_keys=True)
        )

    passed = not errors and metric <= required_max
    result = {
        "task_id": spec["task_id"],
        "metric_key": spec["metric_key"],
        "metric": float(metric),
        "required_final_score_max": str(required_max),
        "per_step_rates": {str(k): float(v) for k, v in sorted(rates.items())},
        "correct_but_zero_samples": sample_detail,
        "run_reality": reality_details,
        "integrity": integrity_details,
        "passed": passed,
        "errors": errors,
    }
    write_json(results / "experiment_metrics.json", result)
    write_json(
        results / "metrics.json",
        {
            "metric": float(metric),
            "required_final_score_max": str(required_max),
            "passed": passed,
            "errors": errors,
        },
    )
    write_json(results / "result.json", {"pass": passed, "score": 1.0 if passed else 0.0})
    if errors:
        for err in errors:
            print(f"error: {err}", file=sys.stderr)
    print(
        f"[grade] correct-but-scored-zero rate (max over steps {sorted(rates)}) = {metric:.4f}; "
        f"required <= {required_max}; {'PASS' if passed else 'FAIL'}"
    )
    return 0 if passed else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results", default=os.environ.get("AUTORLBENCH_RESULTS", "."))
    ap.add_argument(
        "--resources",
        default=str(Path(__file__).resolve().parent.parent / "resources"),
    )
    args = ap.parse_args()
    spec = load_spec(Path(args.resources) / "grader_spec.json")
    resources = Path(args.resources)
    if not (resources / "corrected_outputs.json").is_file():
        print(
            "error: resources/corrected_outputs.json missing (it is produced by the "
            "author-side purification pipeline, see environment/data_prep/README.md)",
            file=sys.stderr,
        )
        return 1
    return grade(spec, Path(args.results), resources)


if __name__ == "__main__":
    sys.exit(main())
