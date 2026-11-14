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
task's data purification from the problems' own agreeing bundled solutions),
it executes the response's code on that problem's recorded failing test and
compares the actual output with the corrected output's stored sha256 digest.
A match on a row the run logged as score 0 is a "correct solution scored 0"
sample; the metric is the maximum over the last `last_k_dumps` steps of
(correct-but-zero samples / rows in that step's dump). The dump writer is
not a pinned file, so the logged score itself is never trusted: a nonzero
(or missing) logged score on a pool problem whose re-execution does not
reproduce the corrected output is a falsified sample, and a falsified rate
above `falsified_rate_max` fails the run (an unrepaired run whose dumps were
edited to claim passes).

The defective arm's rate is materially above zero and does not fall with
training; a run trained on clean rows only has none of these problems left,
so its rate is ~0 (residual: whatever wrong rows a partial repair missed).
Pass requires metric <= required_final_score_max.

The rerun is anchored: live grading byte-compares the workspace launch script
and `verl/utils/reward_score/taco.py` against the verifier's pristine copies,
pins `taco_verl/test.parquet` by sha256, requires the repaired
`taco_verl/train.parquet` to keep at least `min_train_rows` rows, and checks
that no training prompt equals a held-out prompt (training on the validation
set fails). The external judge finds each dumped response's problem by
parsing the prompt the run trained on, so the repaired train.parquet is also
checked against blinding: every row must keep the shipped prompt-template
shape, and no more than `max_residual_dirty_rows` rows may still carry one of
the recorded wrong-expected-output signatures (a row whose test at the
recorded failing index still has the recorded failing input and the original
wrong expected output -- a row whose problem statement was rewritten to hide
it from the lookup while its wrong label stayed in place counts exactly like
an unrepaired one). A non-empty `/workspace/report.md` is required. Dump-side,
every score-0 rollout row must parse back to its problem; unparseable rows
mean the dumps did not come from the anchored pipeline. Run-reality evidence
is the dump sequence itself: one file per training step 1..final_step, each
with `min_rows` rows whose `step` field matches its file -- never the score
values. A replay of packaged evidence by hand runs without the stage
environment and skips the workspace checks.

Executed response code is untrusted: every execution is a fresh `python3 -I`
subprocess with CPU/AS/NPROC/FSIZE rlimits and a wall-clock kill; it runs in
the verifier container exactly like the training run's own reward executions.
Row verdicts are independent, so they are computed in a forked process pool
(each worker single-threaded, keeping the preexec_fn rlimits safe) and fall
back to a sequential loop.
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


def as_stdin_text(raw) -> str:
    """Normalize a test input into stdin text -- the same normalization the
    purification stage applied when it recorded each problem's failing test
    (environment/data_prep/exec_problem.py, copied verbatim) so the signature
    check below compares byte-for-byte against the recorded fail_input."""
    if isinstance(raw, list):
        return "\n".join(str(x) for x in raw)
    if isinstance(raw, str):
        s = raw.strip()
        if s.startswith("[") and s.endswith("]"):
            try:
                v = json.loads(s)
            except ValueError:
                return raw
            if isinstance(v, list):
                return "\n".join(str(x) for x in v)
        return raw
    return str(raw)


def canonical_sha(value) -> str:
    """Byte-stable digest of one test's expected output (str or list form);
    both sides of the comparison load the value from JSON, so the dumps are
    identical for identical values."""
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


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


# ---------------------------------------------------------------------------
# Per-row judging. Rows are independent, and a response on an unsolved or
# adversarial problem can burn the full exec timeout, so verdicts are computed
# in a forked process pool -- each worker is single-threaded, which keeps the
# untrusted-execution preexec_fn rlimits safe -- and fall back to a sequential
# loop when no pool can be created. The corrected-outputs table (~5 MB)
# travels via a module global inherited at fork, not per-row pickles.

_JUDGE_CORRECTED: dict[str, Any] = {}
_JUDGE_TIMEOUT = 60


def _judge_one(row: dict[str, Any]) -> tuple[str, str | None, int | None]:
    """Judge one dumped rollout row: 'judged' when it is a
    correct-but-scored-0 sample, 'falsified' when a nonzero logged score on a
    corrected-pool problem is contradicted by independent re-execution,
    otherwise why it is neither."""
    question = extract_question(row.get("input", ""))
    if question is None:
        return "unparseable", None, None
    qhash = hashlib.sha256(question.encode("utf-8", "replace")).hexdigest()
    entry = _JUDGE_CORRECTED.get(qhash)
    if not entry:
        return "not-dirty", None, None
    code = extract_code(row.get("output", ""))
    stdin_text = entry.get("fail_input") or ""
    corrected_sha = entry.get("corrected_output_sha256")
    # The corrected output is stored as the sha256 of its canonical JSON line
    # list (the same hash of the executed response's stdout lines). Empty
    # outputs are never present: an empty-vs-empty match would vacuously
    # "confirm" any response that prints nothing, and the strict selection
    # produced none.
    if code is None or not corrected_sha:
        return "no-code", None, None
    actual = run_response(code, stdin_text, _JUDGE_TIMEOUT)
    ok = actual is not None and hashlib.sha256(
        json.dumps(actual).encode("utf-8")
    ).hexdigest() == corrected_sha
    if row.get("score") in (0, 0.0):
        if ok:
            return "judged", entry.get("name"), entry.get("fail_index")
        return "wrong-output", None, None
    # Nonzero (or missing) logged score on a pool problem: the run claims the
    # response passed its reward scoring. Cross-check the claim with the same
    # re-execution -- a claimed pass that does not reproduce the corrected
    # output is a falsified score, not a repair. A missing score field counts
    # as a nonzero claim for the same reason. The dump writer is not a pinned
    # file, so the metric must not trust what the run logs about itself.
    if ok:
        return "score-consistent", None, None
    return "falsified", entry.get("name"), entry.get("fail_index")


def _judge_one_indexed(
    item: tuple[int, dict[str, Any]],
) -> tuple[int, str, str | None, int | None]:
    index, row = item
    status, name, fail_index = _judge_one(row)
    return index, status, name, fail_index


def _judge_rows(
    rows: list[dict[str, Any]], corrected: dict[str, Any], exec_timeout: int
) -> list[tuple[int, str, str | None, int | None]]:
    """Verdicts for every row, input order preserved (pool.map is ordered)."""
    global _JUDGE_CORRECTED, _JUDGE_TIMEOUT
    _JUDGE_CORRECTED = corrected
    _JUDGE_TIMEOUT = exec_timeout
    indexed = list(enumerate(rows))
    try:
        import warnings
        from multiprocessing import get_context

        workers = min(8, os.cpu_count() or 1)
        if workers < 2:
            raise ValueError("not enough CPUs for a pool")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            ctx = get_context("fork")
            with ctx.Pool(workers) as pool:
                return pool.map(_judge_one_indexed, indexed)
    except Exception:
        return [_judge_one_indexed(item) for item in indexed]


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
        if path.is_symlink():
            continue
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


def stage_integrity(
    spec: dict[str, Any], corrected: dict[str, Any]
) -> tuple[bool | None, dict[str, Any]]:
    """The rerun stage executes the agent's workspace copy of the launch script on
    the agent's data, so live grading compares: the launch script and taco.py
    byte-for-byte against the pristine copies in resources/, test.parquet by
    sha256, train.parquet by row floor, train/test prompt disjointness, the
    shipped prompt-template shape on every train row, a bound on rows still
    carrying a recorded wrong-expected-output signature, and the required
    non-empty report.md. A replay of packaged evidence by hand runs without
    the stage environment and skips the check."""
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
    train = None
    try:
        import pyarrow.parquet as pq

        train = pq.read_table(
            ws / "taco_verl/train.parquet", columns=["prompt", "reward_model"]
        )
        test = pq.read_table(ws / "taco_verl/test.parquet", columns=["prompt"])
        train_prompts = {json.dumps(p, sort_keys=True) for p in train["prompt"].to_pylist()}
        test_prompts = {json.dumps(p, sort_keys=True) for p in test["prompt"].to_pylist()}
        details["train_rows"] = len(train_prompts)
        details["train_rows_ok"] = len(train_prompts) >= int(spec.get("min_train_rows", 0))
        details["train_test_disjoint"] = not (train_prompts & test_prompts)
    except Exception as exc:  # pyarrow missing or parquet unreadable
        details["train_rows_ok"] = False
        details["train_parquet_error"] = repr(exc)

    # Blinding guard. The external judge identifies each dumped response's
    # problem by parsing the prompt the run trained on (PREFIX ... question
    # ... STDIN_SUFFIX) and hashing the question. A repaired train.parquet
    # whose rows no longer parse -- or whose problems were rewritten to hide
    # them from that lookup while their recorded wrong expected outputs
    # stayed in place -- would score 0 without being repaired, so the
    # template shape is enforced on every row and rows still carrying a
    # recorded wrong-expected signature (the recorded failing input AND the
    # original wrong expected output, both at the recorded failing index --
    # question-intact or question-rewritten alike) are bounded by
    # max_residual_dirty_rows.
    if train is not None:
        try:
            by_fail_index: dict[int, list[dict[str, Any]]] = {}
            for entry in corrected.values():
                if entry.get("expected_sha256") is not None and entry.get("fail_index") is not None:
                    by_fail_index.setdefault(int(entry["fail_index"]), []).append(entry)
            prompts = train["prompt"].to_pylist()
            grounds = [
                rm.get("ground_truth") if isinstance(rm, dict) else None
                for rm in train["reward_model"].to_pylist()
            ]
            bad_template = 0
            residual: list[dict[str, Any]] = []
            for idx, (prompt, gt) in enumerate(zip(prompts, grounds)):
                content = (
                    prompt[0].get("content")
                    if isinstance(prompt, list)
                    and len(prompt) == 1
                    and isinstance(prompt[0], dict)
                    and prompt[0].get("role") == "user"
                    else None
                )
                if not (
                    isinstance(content, str)
                    and content.startswith(PREFIX)
                    and content.endswith(STDIN_SUFFIX)
                    and len(content) > len(PREFIX) + len(STDIN_SUFFIX)
                ):
                    bad_template += 1
                try:
                    in_outs = json.loads(gt) if isinstance(gt, str) else None
                except ValueError:
                    in_outs = None
                inputs = (in_outs or {}).get("inputs") or []
                outputs = (in_outs or {}).get("outputs") or []
                for k in range(min(len(inputs), len(outputs))):
                    for entry in by_fail_index.get(k, ()):
                        if (
                            as_stdin_text(inputs[k]) == entry.get("fail_input")
                            and canonical_sha(outputs[k]) == entry.get("expected_sha256")
                        ):
                            residual.append(
                                {"row": idx, "name": entry.get("name"), "fail_index": k}
                            )
                            break
            details["train_prompt_template_ok"] = bad_template == 0
            if bad_template:
                details["train_prompt_template_bad_rows"] = bad_template
            tolerance = int(spec.get("max_residual_dirty_rows", 0))
            details["train_residual_dirty_rows"] = len(residual)
            details["train_residual_dirty_ok"] = len(residual) <= tolerance
            if residual:
                details["train_residual_dirty_examples"] = residual[:3]
        except Exception as exc:
            details["train_residual_dirty_ok"] = False
            details["train_content_error"] = repr(exc)

    # The instruction requires a written report; a missing or empty one fails
    # the run rather than only the score.
    report = ws / "report.md"
    try:
        details["report_md_ok"] = report.is_file() and report.stat().st_size > 0
    except OSError:
        details["report_md_ok"] = False
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
        (
            int(p.stem)
            for p in dump_dir.glob("*.jsonl")
            if p.stem.isdigit() and not p.is_symlink()
        ),
        reverse=True,
    )[:last_k]
    if not steps:
        print(json.dumps({"errors": ["no rollout dumps found"]}))
        return 1

    rates = {}
    falsified_rates = {}
    sample_detail = []
    falsified_detail = []
    for step in steps:
        path = dump_dir / f"{step}.jsonl"
        if path.is_symlink():
            continue
        rows = read_dump(path)
        if len(rows) < spec["min_rows"]:
            errors.append(
                f"{path.name} has {len(rows)} rows, fewer than the {spec['min_rows']} required"
            )
            continue
        judged = 0
        falsified = 0
        unparseable = 0
        for i, status, name, fail_index in _judge_rows(rows, corrected, exec_timeout):
            if status == "judged":
                judged += 1
                if len(sample_detail) < 8:
                    sample_detail.append(
                        {"step": step, "row": i, "name": name, "fail_index": fail_index}
                    )
            elif status == "falsified":
                falsified += 1
                if len(falsified_detail) < 8:
                    falsified_detail.append(
                        {"step": step, "row": i, "name": name, "fail_index": fail_index}
                    )
            elif status == "unparseable":
                unparseable += 1
        rates[step] = judged / len(rows)
        falsified_rates[step] = falsified / len(rows)
        # A dump the anchored pipeline wrote renders every parquet prompt,
        # and every train row is checked to keep the parseable template, so
        # score-0 rows that do not parse back to a problem mean the dumps were
        # not produced by that pipeline -- blinding, not repairing.
        if unparseable:
            errors.append(
                f"{path.name}: {unparseable} score-0 rollout rows could not be "
                "parsed back to their problems; the dumps do not match the "
                "anchored prompt template"
            )

    if not rates:
        print(json.dumps({"errors": errors or ["no scorable dumps"]}))
        return 1
    metric = max(rates.values())
    falsified_max = float(spec.get("falsified_rate_max", "0.05"))
    falsified_metric = max(falsified_rates.values()) if falsified_rates else 0.0
    if falsified_metric > falsified_max:
        errors.append(
            f"falsified nonzero-score rate (max over steps {sorted(falsified_rates)}) = "
            f"{falsified_metric:.4f} > {falsified_max}: dumped scores that independent "
            "re-execution contradicts"
        )

    _, reality_details, reality_error = training_reality(results, spec)
    if reality_error:
        errors.append(reality_error)
    integrity_ok, integrity_details = stage_integrity(spec, corrected)
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
        "falsified_rate_max": falsified_max,
        "falsified_metric": float(falsified_metric),
        "per_step_rates": {str(k): float(v) for k, v in sorted(rates.items())},
        "per_step_falsified_rates": {
            str(k): float(v) for k, v in sorted(falsified_rates.items())
        },
        "correct_but_zero_samples": sample_detail,
        "falsified_samples": falsified_detail,
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
            "falsified_metric": float(falsified_metric),
            "falsified_rate_max": falsified_max,
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
        f"required <= {required_max}; falsified rate = {falsified_metric:.4f} "
        f"(max {falsified_max}); {'PASS' if passed else 'FAIL'}"
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
