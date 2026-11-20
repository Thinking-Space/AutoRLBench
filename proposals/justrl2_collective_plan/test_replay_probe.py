# SPDX-License-Identifier: Apache-2.0
"""Runner contracts; native snapshots stay in maintainer tests, never the runner."""
import ast
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

HERE = Path(__file__).resolve().parent
RUNNER = HERE / "replay_probe.py"


def probe():
    assert RUNNER.is_file(), "standalone runner is not implemented"
    spec = importlib.util.spec_from_file_location("replay_probe", RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def native_source(tmp_path):
    # Reuse only the public native-function snapshot, not the verifier/oracle.
    snapshots = json.loads((HERE / "native_functions.json").read_text())["functions"]
    root = tmp_path / "source"
    for relative, text in snapshots.items():
        if relative.endswith(("ppo_utils.py", "distributed_utils.py")):
            text = "from miles.utils.collective_plan import reduce_sum\n" + text.replace("dist.all_reduce(", "reduce_sum(")
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    (root / "miles/utils/collective_plan.py").write_text((HERE / "collective_plan.py").read_text())
    return root


def fixed(source):
    path = source / "miles/utils/collective_plan.py"
    text = path.read_text()
    assert "key = _signature(tensor, group)" in text
    path.write_text(text.replace("key = _signature(tensor, group)", "key = (group, *_signature(tensor, group))"))


@pytest.fixture
def native_indices(native_source):
    namespace = {}
    path = native_source / "miles/backends/training_utils/cp_utils.py"
    exec(compile("from __future__ import annotations\n" + path.read_text(), str(path), "exec"), namespace)
    def indices(sample, cp, rank):
        namespace["get_parallel_state"] = lambda: SimpleNamespace(cp=SimpleNamespace(size=cp, rank=rank))
        return namespace["slice_log_prob_with_cp"](list(range(sample["length"])), sample["total"], sample["length"])
    return indices


def test_generated_samples_keep_terminal_response_on_native_cp0(native_indices):
    runner = probe()
    for cp in (1, 2, 4):
        for replica in range(3):
            for length in (3, 1, 2, 7, 11, 17, 4096):
                samples = runner._samples(length, cp, replica, 1729, 0, "ones")
                assert len(samples) == 1
                for sample in samples:
                    assert length - 1 in native_indices(sample, cp, 0), (cp, replica, sample)
                    assert sample["total"] % (2 * cp) == 0
                    assert sample["total"] // (2 * cp) >= 2


@pytest.mark.parametrize("cp,dp,lengths", [
    (1, 2, (3, 7, 11, 17)), (2, 2, (3, 7, 11, 17)), (2, 3, (3, 7, 11, 17)),
    (4, 2, (7, 11, 17)), (4, 3, (7, 11, 17)),
])
def test_supported_matrix_has_valid_masks_on_nonempty_native_slices(native_indices, cp, dp, lengths):
    runner = probe()
    for seed in (0, 1, 2, 3, 1729):
        for step, length in enumerate(lengths):
            counts = [0] * cp
            for replica in range(dp):
                for sample in runner._samples(length, cp, replica, seed, step, "sparse"):
                    for rank in range(cp):
                        owned = native_indices(sample, cp, rank)
                        count = sum(sample["masks"][i] for i in owned)
                        assert not owned or count > 0, (cp, replica, length, rank, seed)
                        counts[rank] += count
            assert all(count > 0 for count in counts), (cp, dp, length, seed, counts)


@pytest.mark.parametrize("options", [
    {"cp": 0}, {"cp": 3}, {"dp": 1}, {"dp": 4}, {"steps": []},
    {"steps": [0]}, {"steps": [-1]}, {"steps": [True]},
    {"steps": [4097]}, {"steps": [3] * 17},
    {"order": [0, 1, 1, 3]}, {"order": [0, 1]},
    {"order": [1, 0, 2, 3]}, {"mask_mode": "bad"},
])
def test_invalid_arguments_fail_closed(tmp_path, options):
    with pytest.raises((ValueError, TypeError)):
        probe().run_probe(tmp_path / "source", tmp_path / "out", **options)
    assert not (tmp_path / "out").exists()


def test_missing_source_is_infrastructure_error(tmp_path):
    result = probe().run_probe(tmp_path / "absent", tmp_path / "out")
    assert result["status"] == "infrastructure_error"
    assert result["exit_code"] == 2
    assert result == json.loads((tmp_path / "out/summary.json").read_text())


def test_replay_size_boundaries_are_accepted():
    config = probe()._configuration(4, 3, None, [1, 4096] * 8, "ones", 1729, False)
    assert len(config["steps"]) == 16 and max(config["steps"]) == 4096


def test_source_changed_during_execution_is_infrastructure_error(native_source, tmp_path):
    path = native_source / "miles/utils/collective_plan.py"
    path.write_text(path.read_text() + '''
from pathlib import Path
_original_reduce_sum = reduce_sum
_source_changed = False
def reduce_sum(tensor, group=None):
    global _source_changed
    if dist.get_rank() == 0 and not _source_changed:
        path = Path(__file__)
        path.write_text(path.read_text() + "\\n# concurrent edit\\n")
        _source_changed = True
    return _original_reduce_sum(tensor, group=group)
''')
    result = probe().run_probe(native_source, tmp_path / "out", cp=1, steps=[3])
    assert all(row["status"] == "completed" for row in result["ranks"]), result
    assert result["status"] == "infrastructure_error", result
    assert result["exit_code"] == 2 and not result["outer_timeout"]
    assert result["source_changed"] == ["miles/utils/collective_plan.py"]
    assert result["source_sha256_after"] != result["source_sha256"]
    assert "source changed" in result["error"]


def test_cli_invalid_arguments_and_fresh_output(tmp_path):
    for args in (["--cp", "0"], ["--steps", "11,,3"], ["--order", "0,1,1,3"],
                 ["--steps", "4097"], ["--steps", ",".join(["3"] * 17)]):
        result = subprocess.run([sys.executable, str(RUNNER), "--output", str(tmp_path / "new"), *args], capture_output=True, text=True)
        assert result.returncode == 2
        assert not (tmp_path / "new").exists()
    with pytest.raises(FileExistsError):
        probe().run_probe(tmp_path / "source", tmp_path)


def test_cp1_completion_records_native_outputs_without_routes(native_source, tmp_path):
    result = probe().run_probe(native_source, tmp_path / "out", cp=1, dp=2)
    assert result["status"] == "completed", result
    assert result["exit_code"] == 0
    assert result["source_sha256_after"] == result["source_sha256"]
    rows = [json.loads(p.read_text()) for p in sorted((tmp_path / "out").glob("rank*.json"))]
    assert len(rows) == 2
    for row in rows:
        assert row["done"] and row["status"] == "completed"
        assert [s["response_length"] for s in row["steps"]] == [11, 3]
        for step in row["steps"]:
            assert step["status"] == "completed"
            assert step["samples"] and step["sliced_values"]
            assert "raw_advantages" in step and "raw_returns" in step
            assert "whitening" in step and "advantages" in step and "returns" in step
        text = json.dumps(row)
        assert "requested_members" not in text and "actual_members" not in text
    assert not list((tmp_path / "out").glob("*.jsonl"))


@pytest.mark.parametrize("cp,dp,order,steps", [
    (2, 2, [0, 2, 1, 3], (11, 3)), (2, 3, None, (11, 3)),
    (4, 2, [0, 2, 4, 6, 1, 3, 5, 7], (7, 11, 17)), (4, 3, None, (11, 17)),
])
def test_fixed_plan_completes_and_trace_records_real_routes(native_source, tmp_path, cp, dp, order, steps):
    fixed(native_source)
    result = probe().run_probe(native_source, tmp_path / "out", cp=cp, dp=dp, order=order, steps=steps, trace=True, mask_mode="sparse")
    assert result["status"] == "completed", result
    for path in (tmp_path / "out").glob("*.trace.jsonl"):
        events = [json.loads(line) for line in path.read_text().splitlines()]
        assert events and [e["sequence"] for e in events] == list(range(len(events)))
        actual = [e for e in events if e["operation"] == "all_reduce"]
        assert {e["phase"] for e in actual} == {"gae", "whiten"}
        assert {tuple(e["shape"]) for e in actual if e["phase"] == "gae"} == {(1, length) for length in steps}
        assert all(e["requested_members"] == e["actual_members"] for e in actual)
        assert all(e["event"] in ("enter", "return") for e in events)
        assert all(e["numel"] >= 0 and e["caller"] for e in events)
        assert not any("timeout" in e for e in events)


def test_native_functions_really_depend_on_candidate_reduce_sum(native_source, tmp_path):
    path = native_source / "miles/utils/collective_plan.py"
    path.write_text("import torch.distributed as dist\n_calls = []\ndef clear():\n    _calls.clear()\ndef reduce_sum(tensor, group=None):\n    _calls.append(tuple(tensor.shape))\n    raise RuntimeError('candidate dependency reached')\n")
    result = probe().run_probe(native_source, tmp_path / "out", trace=True)
    assert result["status"] == "candidate_error", result
    assert result["exit_code"] == 1 and not result["outer_timeout"]
    rows = [json.loads(p.read_text()) for p in (tmp_path / "out").glob("rank*.json")]
    assert all("candidate dependency reached" in r["steps"][0]["error"] for r in rows)
    for path in (tmp_path / "out").glob("*.jsonl"):
        events = [json.loads(line) for line in path.read_text().splitlines()]
        assert all(e["operation"] == "reduce_sum" for e in events)
        assert not any("timeout" in e for e in events)


def test_cached_wrong_routes_are_observed_not_predicted(native_source, tmp_path):
    result = probe().run_probe(native_source, tmp_path / "out", trace=True)
    assert result["status"] == "candidate_error", result
    assert not result["outer_timeout"]
    events = [json.loads(line) for path in (tmp_path / "out").glob("*.jsonl") for line in path.read_text().splitlines()]
    assert any(e["operation"] == "all_reduce" and e["requested_members"] != e["actual_members"] for e in events)
    errors = [e for e in events if e["operation"] == "all_reduce" and e["event"] == "error"]
    assert errors
    assert any("timed out" in e["error"].lower() or "timeout" in e["error"].lower() for e in errors)
    assert all("connection closed by peer" not in e["error"].lower() for e in errors)
