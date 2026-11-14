#!/usr/bin/env python3
"""One dirty-pool candidate problem, judged under a loose timeout.

Runs INSIDE a disposable container (see data_prep/README.md); reads one problem
JSON produced by build_dirty_pool.py, executes up to --max-solutions of its
bundled solutions under the task's own judge (`verl/utils/reward_score/taco.py`,
the exact scorer the training run uses), and emits one JSON line on stdout.

The verdict that matters:

  native_wrong_expected_output
      >= --agree-k textually distinct bundled solutions all run the problem to
      completion, all fail the SAME test index, and their actual outputs on
      that test are identical to each other but differ from the dataset's
      expected output. That is the "correct solution is deterministically
      scored 0" family: the solutions agree with each other, so the residual
      explanation is that the expected output itself is wrong.

Everything else is a rejection with a reason:

  passes_loose    some bundled solution passes all tests under the LOOSE
                  timeout -> the problem is not in the false-negative family
                  at all (it only timed out under the verified set's stricter
                  budget); it must not enter the dirty manifest.
  timeout        no solution produced a verdict within the per-test/global
                  budgets -> pure slowness, wrong family.
  compile        every runnable solution failed to compile.
  disagree       solutions failed at different indices or with different
                  outputs -> no defensible "expected output is wrong" claim.
  no_tests / harness_timeout / error
                  bookkeeping failures.

taco.py is imported from the task workspace and patched only in two places:
the per-test alarm (module global `timeout`, 4 s in the shipped file) is raised
to --timeout, and the debug clip is widened so the receipt records the full
actual/expected outputs rather than an 800-character prefix. The comparison
logic itself is untouched -- receipts are produced by the same judge the
training run scores with.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path


def as_stdin_text(raw) -> str:
    """Normalize taco.py's debug `input` field into plain stdin text.

    For stdin problems taco stores the raw test input in ctx["inputs"] before
    joining, and _clip_debug json-encodes a list, so the receipt may hold
    '["3\\n3\\n"]' rather than '3\\n3\\n'. The grader re-feeds this text on a
    subprocess stdin, so store it already joined.
    """
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


def load_taco(taco_path: str):
    spec = importlib.util.spec_from_file_location("taco_judge", taco_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["taco_judge"] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--problem", required=True, help="problem JSON from build_dirty_pool.py")
    ap.add_argument("--taco", required=True, help="path to verl/utils/reward_score/taco.py")
    ap.add_argument("--timeout", type=float, default=30.0, help="per-test alarm seconds")
    ap.add_argument("--global-cap", type=float, default=240.0,
                    help="cap on the whole-suite join timeout for one solution")
    ap.add_argument("--max-solutions", type=int, default=4)
    ap.add_argument("--agree-k", type=int, default=2,
                    help="distinct solutions that must agree on the same failing test")
    args = ap.parse_args()

    problem = json.loads(Path(args.problem).read_text(encoding="utf-8"))
    # Train-pool records carry taco_index; test-pool records carry test_index
    # (the position in the held-out split). Echo whichever key is present.
    out = {
        "taco_index": problem.get("taco_index", problem.get("test_index")),
        "split": "test" if "test_index" in problem else "train",
        "name": problem.get("name"),
        "difficulty": problem.get("difficulty"),
        "n_tests": problem.get("n_tests"),
        "n_solutions": problem.get("n_solutions"),
        "question_sha256": hashlib.sha256(
            problem.get("question", "").encode("utf-8", "replace")
        ).hexdigest(),
        "verdict": "error",
        "evidence": [],
    }

    taco = load_taco(args.taco)
    # Loose budget for the re-execution (the verified set used 10 s; the brief
    # calls for 30-60 s to separate wrong-expected-output from timeouts).
    taco.timeout = int(args.timeout)
    taco.TIMEOUT = int(args.timeout)
    # Receipts carry full outputs; clip only truly pathological sizes.
    taco._clip_debug = lambda obj, limit=2_000_000: (  # noqa: E731
        obj if obj is None else (obj if isinstance(obj, (str, list, int, float, bool)) else str(obj))
    )
    # check_correctness joins the child for `timeout + 1`; keep the per-test
    # alarm as the real limit and scale the global join with the suite size,
    # like compute_score does (4*n+1), under a hard cap.
    n_tests = int(problem.get("n_tests") or 0)
    global_join = min(max(args.timeout, 4 * max(n_tests, 1) + 1), args.global_cap)

    sample = {"input_output": problem["input_output"]}
    evidence = []
    seen_solution_hashes = set()
    errored = False
    try:
        for sol_idx, sol in enumerate(problem.get("solutions", [])[: args.max_solutions]):
            sol_hash = hashlib.sha256(sol.encode("utf-8", "replace")).hexdigest()
            if sol_hash in seen_solution_hashes:
                continue
            seen_solution_hashes.add(sol_hash)
            # compute_score() strips the fence with post_process() BEFORE
            # check_correctness(); check_correctness itself expects bare code.
            # Feed the raw solution through the same post_process a model
            # response gets, so the receipts match the training judge exactly.
            code = taco.post_process("```python\n" + sol + "\n```")
            try:
                results, fail_debug = taco.check_correctness(
                    sample, code, timeout=global_join, debug=False
                )
            except Exception as exc:  # defensive: harness itself blew up
                evidence.append({"sol": sol_hash[:16], "stage": "harness", "error": repr(exc)})
                continue

            if results and all(r is True for r in results):
                out["verdict"] = "passes_loose"
                out["evidence"] = evidence
                print(json.dumps(out, ensure_ascii=False))
                return 0

            if not results:
                out["verdict"] = "no_tests"
                out["evidence"] = evidence
                print(json.dumps(out, ensure_ascii=False))
                return 0

            dbg = fail_debug or {}
            raw_err = dbg.get("error")
            err = str(raw_err) if raw_err is not None else None
            if err and "global_timeout" in err:
                evidence.append({"sol": sol_hash[:16], "stage": "global_timeout"})
                continue
            if any(r == -2 for r in results) or (err or "").startswith("compile"):
                evidence.append({"sol": sol_hash[:16], "stage": "compile"})
                continue
            evidence.append({
                "sol": sol_hash[:16],
                "stage": "mismatch",
                "fail_index": dbg.get("fail_index"),
                "input": as_stdin_text(dbg.get("input"))[:65536],
                "expected": dbg.get("expected"),
                "output": dbg.get("output"),
                "error": err,
            })

            # Early accept: --agree-k distinct solutions, same failing test,
            # identical NON-EMPTY actual outputs. A "mismatch" entry only
            # counts when the solution ran the test to completion (no runtime
            # error) and actually produced output: an empty agreed output
            # means every solution failed to even parse the test input (the
            # GfG-style "N = 8\narr[] = {...}" corruption family) -- that
            # says nothing about the expected output being wrong, and an
            # empty-vs-empty match would vacuously "confirm" any
            # print-nothing response downstream. taco.py has already judged
            # the (non-empty) output unequal to the expected one, which is
            # the false-negative signal itself.
            def _clean(e):
                return (
                    e.get("stage") == "mismatch"
                    and e.get("error") is None
                    and e.get("output")
                )

            last = evidence[-1]
            agreeing = [
                e for e in evidence
                if _clean(e)
                and e.get("fail_index") == last["fail_index"]
                and e.get("output") == last["output"]
            ]
            if len({e["sol"] for e in agreeing}) >= args.agree_k:
                out["verdict"] = "native_wrong_expected_output"
                out["fail_index"] = evidence[-1]["fail_index"]
                out["expected"] = evidence[-1]["expected"]
                out["corrected_output"] = evidence[-1]["output"]
                out["fail_input"] = as_stdin_text(dbg.get("input"))
                out["agreeing_solutions"] = [e["sol"] for e in agreeing]
                break
    except Exception as exc:
        out["verdict"] = "error"
        out["error"] = repr(exc)
        errored = True

    if errored:
        out["evidence"] = evidence
    elif evidence and any(
        e.get("stage") == "mismatch" and e.get("error") is None and e.get("output")
        for e in evidence
    ):
        mismatches = [
            e for e in evidence
            if e.get("stage") == "mismatch" and e.get("error") is None and e.get("output")
        ]
        same_idx = {}
        for e in mismatches:
            same_idx.setdefault((e.get("fail_index"), json.dumps(e.get("output"), ensure_ascii=False)), []).append(e)
        best = max(same_idx.values(), key=len)
        if len({e["sol"] for e in best}) >= args.agree_k:
            out["verdict"] = "native_wrong_expected_output"
            out["fail_index"] = best[0]["fail_index"]
            out["expected"] = best[0]["expected"]
            out["corrected_output"] = best[0]["output"]
            out["fail_input"] = best[0].get("input")
            out["agreeing_solutions"] = [e["sol"] for e in best]
        else:
            out["verdict"] = "disagree"
        out["evidence"] = evidence
    elif evidence:
        stages = {e.get("stage") for e in evidence}
        if stages <= {"compile"}:
            out["verdict"] = "compile"
        elif stages <= {"global_timeout"}:
            out["verdict"] = "timeout"
        elif evidence and all(
            e.get("stage") == "mismatch" and "timeout" in str(e.get("error") or "").lower()
            for e in evidence
        ):
            out["verdict"] = "timeout"
        else:
            out["verdict"] = "unverified"
        out["evidence"] = evidence
    else:
        out["verdict"] = "no_runnable_solution"

    print(json.dumps(out, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
