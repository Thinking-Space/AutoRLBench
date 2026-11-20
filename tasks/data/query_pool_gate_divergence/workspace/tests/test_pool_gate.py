"""Unit tests of the pool gate.

The gate is a shell script with an inline validator, so these tests exercise it end to end on
small fixtures: a pool is written to a temporary directory, the launcher is run against it with
`POOL_VALIDATE_ONLY=1`, and the exit status is the assertion.

    python3 -m pytest tests -q
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[1]
LAUNCHER = WORKSPACE / "scripts" / "launch_run.sh"
TOOLS = ("execute_bash", "think", "complete_plan", "finish")


def write_pool(root: Path, counts: dict[str, int], total: int | None = None) -> Path:
    pool = root / "pool"
    pool.mkdir(parents=True, exist_ok=True)
    rows = []
    for source, count in counts.items():
        for index in range(count):
            rows.append({
                "prompt": f"{source} problem {index}",
                "tools": [{"type": "function", "function": {"name": name}} for name in TOOLS],
                "metadata": {"pool_source": source, "instance_id": f"{source}__{index}", "repo": "demo/demo"},
            })
    (pool / "queries.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )
    manifest = {
        "version": 2,
        "seed": 1,
        "source_counts": counts,
        "excluded_instance_counts": {source: 0 for source in counts},
    }
    if total is not None:
        manifest["total_queries"] = total
    (pool / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return pool


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_gate(pool: Path, registry: Path, **env: str) -> subprocess.CompletedProcess:
    environment = dict(os.environ)
    environment.pop("RUN_TAG", None)
    environment.update({
        "POOL_DIR": str(pool),
        "POOL_REGISTRY": str(registry),
        "POOL_SHA256": digest(pool / "queries.jsonl"),
        "POOL_MANIFEST_SHA256": digest(pool / "manifest.json"),
        "POOL_REGISTRY_SHA256": digest(registry),
        "POOL_VALIDATE_ONLY": "1",
        "JOB_ID": "000001",
        # The fixtures are tens of queries, not thousands: size the batches for them.
        "OVER_SAMPLING_BATCH_SIZE": "8",
        "ROLLOUT_BATCH_SIZE": "4",
    })
    environment.update(env)
    return subprocess.run(["bash", str(LAUNCHER)], cwd=str(WORKSPACE), env=environment,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


def empty_registry(root: Path) -> Path:
    path = root / "registry.json"
    path.write_text(json.dumps({"selected": []}) + "\n", encoding="utf-8")
    return path


def test_accepts_a_consistent_pool(tmp_path):
    pool = write_pool(tmp_path, {"swe_smith": 6, "swe_rebench": 4}, total=10)
    assert run_gate(pool, empty_registry(tmp_path)).returncode == 0


def test_rejects_a_total_that_disagrees_with_the_source_counts(tmp_path):
    pool = write_pool(tmp_path, {"swe_smith": 6, "swe_rebench": 4}, total=11)
    assert run_gate(pool, empty_registry(tmp_path)).returncode != 0


def test_rejects_a_pool_whose_rows_went_missing(tmp_path):
    pool = write_pool(tmp_path, {"swe_smith": 6, "swe_rebench": 4}, total=10)
    lines = (pool / "queries.jsonl").read_text(encoding="utf-8").splitlines(keepends=True)
    (pool / "queries.jsonl").write_text("".join(lines[:-1]), encoding="utf-8")
    assert run_gate(pool, empty_registry(tmp_path)).returncode != 0


def test_rejects_a_pool_with_an_unexpected_tool_list(tmp_path):
    pool = write_pool(tmp_path, {"swe_smith": 4}, total=4)
    rows = [json.loads(line) for line in (pool / "queries.jsonl").read_text(encoding="utf-8").splitlines()]
    rows[0]["tools"] = rows[0]["tools"][:2]
    (pool / "queries.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )
    assert run_gate(pool, empty_registry(tmp_path)).returncode != 0


def test_rejects_a_pool_holding_an_unusable_instance(tmp_path):
    pool = write_pool(tmp_path, {"swe_smith": 4}, total=4)
    registry = tmp_path / "registry.json"
    registry.write_text(json.dumps({"selected": [
        {"source": "swe_smith", "instance_id": "swe_smith__1"}]}) + "\n", encoding="utf-8")
    manifest = json.loads((pool / "manifest.json").read_text(encoding="utf-8"))
    manifest["excluded_instance_counts"] = {"swe_smith": 1}
    (pool / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    assert run_gate(pool, registry).returncode != 0


def test_rejects_a_corrupted_pool(tmp_path):
    pool = write_pool(tmp_path, {"swe_smith": 4}, total=4)
    assert run_gate(pool, empty_registry(tmp_path), POOL_SHA256="0" * 64).returncode != 0
