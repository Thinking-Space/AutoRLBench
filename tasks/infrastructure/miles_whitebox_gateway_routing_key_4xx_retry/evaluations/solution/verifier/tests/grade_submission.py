#!/usr/bin/env python3
"""Grade a repaired workspace against the fresh rerun and sealed checks.

Three gates, all of which must pass:

1. sealed-file integrity — the routing proxy, the rollout plugin, the recipe,
   the experiment driver, the dataset, and the recorded evidence must be
   byte-identical to the pristine copies shipped with the task;
2. the fresh rerun's ``experiment_logs/summary.json`` — the training-level
   contract (iterations complete, no routing-key rejections, no inner
   retry storm);
3. the retry-policy unit check — the workspace's ``_post`` must fail fast on
   a deterministic 4xx while still retrying transport errors, 5xx, 408 and
   429. This axis is invisible at the training level (a gateway fix alone
   removes every deterministic 400), so it is checked directly.

Writes result.json into $AUTORLBENCH_RESULTS and exits 0 only on a pass.
"""

from __future__ import annotations

import asyncio
import hashlib
import importlib
import json
import os
import sys
from pathlib import Path

WORKSPACE = Path(os.environ.get("AUTORLBENCH_WORKSPACE", "/workspace"))
RESULTS = Path(os.environ.get("AUTORLBENCH_RESULTS", "/results"))

REQUIRED_ITERATIONS = 5
MAX_PROXY_REJECTIONS = 0
MAX_INNER_RETRIES = 0

# task-root-relative -> sha256 of the pristine shipped copy
SEALED_FILES = {
    "routing_proxy/strict_router.py": "8dc0b8c8b23463020de5112dd9e98a677242d1aba49706c7fba652490135998e",
    "rollout/gateway_generate.py": "cadb35be0a68718a18963d091914f3a8f16a1d7173093169f0b19050dfcafb04",
    "recipes/run_training.py": "dcbca37c0394f888a3335a56721b21b4faae32cde8e8cdda7ca1cfb21f1ecbee",
    "scripts/run_experiment.sh": "c089c699baf5322512c69088af89da45ef566349c3536d95c857c8573c6426ce",
    "data/dapo_math_17k_dedup.jsonl": "8e16ebfd017010aa6c6dc0da024d524c59d96abadababac0c50c4f972f5a359c",
    "gateway/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "rollout/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "routing_proxy/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "recorded_run/train.log.gz": "f1ce656e8aee37769711672482f04e281691dd2f2796bd08e79d8540fc464be8",
    "recorded_run/gateway.log.gz": "b323dfb99cec2a74c2462ea2704673f5e8ca64e6eab0659e67d56e9c84dafd61",
    "recorded_run/shim.log.gz": "7deddf6dcc15ffb220a3b241bfc5339266892adc392200bbf2fa1d7577862364",
    "recorded_run/launch.log": "a821a992743febc49034092aa9002016d9877ef9b500d4a97de3ef6e654a21cf",
}


def fail(reason: str) -> None:
    print(f"grade_submission: {reason}", file=sys.stderr)


def write_result(passed: bool, metrics: dict) -> int:
    payload = {
        "passed": passed,
        "score": 1.0 if passed else 0.0,
        "metrics": metrics,
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "result.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    return 0 if passed else 1


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_sealed() -> tuple[bool, list[str]]:
    """Hash the sealed files against the pristine copies shipped with the task."""
    problems = []
    for rel, expected in SEALED_FILES.items():
        live = WORKSPACE / rel
        if not live.is_file():
            problems.append(f"sealed file deleted: {rel}")
            continue
        if digest(live) != expected:
            problems.append(f"sealed file modified: {rel}")
    return not problems, problems


def check_rerun_summary() -> tuple[bool, list[str], dict]:
    summary_path = RESULTS / "experiment_logs" / "summary.json"
    if not summary_path.is_file():
        return False, [f"fresh rerun summary missing: {summary_path}"], {}
    try:
        summary = json.loads(summary_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return False, [f"unreadable summary: {exc}"], {}

    problems = []
    iterations = int(summary.get("rollout_iterations_completed", 0))
    rejections = int(summary.get("proxy_rejections", 0))
    inner_retries = int(summary.get("gateway_inner_retry_lines", 0))
    if iterations < REQUIRED_ITERATIONS:
        problems.append(
            f"fresh run completed {iterations} rollout iterations, need >= {REQUIRED_ITERATIONS}"
        )
    if rejections > MAX_PROXY_REJECTIONS:
        problems.append(
            f"fresh run still has {rejections} routing-key rejections (need <= {MAX_PROXY_REJECTIONS})"
        )
    if inner_retries > MAX_INNER_RETRIES:
        problems.append(
            f"fresh run logged {inner_retries} gateway inner retry lines (need <= {MAX_INNER_RETRIES})"
        )
    metrics = {
        "rollout_iterations_completed": iterations,
        "proxy_rejections": rejections,
        "gateway_inner_retry_lines": inner_retries,
        "outer_retry_lines": int(summary.get("outer_retry_lines", 0)),
        "window_s": summary.get("window_s"),
        "elapsed_s": summary.get("elapsed_s"),
    }
    return not problems, problems, metrics


class _ScriptedHttpClient:
    """Replays pre-built responses/exceptions in order and counts attempts."""

    def __init__(self, script):
        self._script = list(script)
        self.attempts = 0

    async def post(self, url, json=None, headers=None):
        self.attempts += 1
        item = self._script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class _FastSleep:
    """Replaces asyncio.sleep inside _post so retries cost no wall clock."""

    async def __call__(self, _seconds):
        return None


def _make_response(status_code: int):
    import httpx

    return httpx.Response(status_code, request=httpx.Request("POST", "http://engine/generate"))


def check_retry_policy() -> tuple[bool, list[str], dict]:
    """Import the workspace's http_utils and exercise _post directly."""
    miles_root = WORKSPACE / "miles"
    if not (miles_root / "miles" / "utils" / "http_utils.py").is_file():
        return False, ["workspace miles tree missing"], {}

    for stale in [m for m in list(sys.modules) if m == "miles" or m.startswith("miles.")]:
        del sys.modules[stale]
    sys.path.insert(0, str(miles_root))
    try:
        http_utils = importlib.import_module("miles.utils.http_utils")
    except Exception as exc:  # noqa: BLE001 - any import failure is a failed repair
        return False, [f"cannot import miles.utils.http_utils from the workspace: {exc}"], {}
    finally:
        try:
            sys.path.remove(str(miles_root))
        except ValueError:
            pass

    problems = []
    metrics = {}

    async def run_case(script):
        client = _ScriptedHttpClient(script)
        original_sleep = http_utils.asyncio.sleep
        http_utils.asyncio.sleep = _FastSleep()
        try:
            try:
                output = await http_utils._post(client, "http://engine/generate", {}, max_retries=5)
                return "ok", output, client.attempts
            except Exception as exc:  # noqa: BLE001 - the raised type is part of the case
                return type(exc).__name__, exc, client.attempts
        finally:
            http_utils.asyncio.sleep = original_sleep

    # deterministic client errors: exactly one attempt, then raise
    for status in (400, 401, 403, 404, 422):
        kind, exc, attempts = asyncio.run(run_case([_make_response(status)]))
        metrics[f"post_{status}_attempts"] = attempts
        if attempts != 1 or kind != "HTTPStatusError":
            problems.append(
                f"_post on {status}: expected 1 attempt then HTTPStatusError, got {attempts} attempt(s) -> {kind}"
            )

    # transient statuses and transport errors: retried, then succeed
    for status in (408, 429, 500, 503):
        import httpx

        kind, _, attempts = asyncio.run(
            run_case([_make_response(status), _make_response(200)])
        )
        metrics[f"post_{status}_then_200_attempts"] = attempts
        if attempts != 2 or kind != "ok":
            problems.append(
                f"_post on {status} then 200: expected 2 attempts then success, got {attempts} attempt(s) -> {kind}"
            )
        kind, _, attempts = asyncio.run(
            run_case([httpx.ConnectError("boom"), _make_response(200)])
        )
        metrics["post_connect_error_then_200_attempts"] = attempts
        if attempts != 2 or kind != "ok":
            problems.append(
                f"_post on transport error then 200: expected 2 attempts then success, got {attempts} attempt(s) -> {kind}"
            )

    return not problems, problems, metrics


def main() -> int:
    sealed_ok, sealed_problems = check_sealed()
    rerun_ok, rerun_problems, rerun_metrics = check_rerun_summary()
    retry_ok, retry_problems, retry_metrics = check_retry_policy()

    passed = sealed_ok and rerun_ok and retry_ok
    problems = sealed_problems + rerun_problems + retry_problems
    metrics = {
        **rerun_metrics,
        **retry_metrics,
        "sealed_files_ok": sealed_ok,
        "retry_policy_ok": retry_ok,
    }
    if problems:
        metrics["problems"] = problems
    return write_result(passed, metrics)


if __name__ == "__main__":
    raise SystemExit(main())
