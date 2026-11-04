"""Offline behavioral tests; no GPU, SSH connection or automation is created."""

import contextlib
import io
import json
from pathlib import Path
import shlex
import subprocess
import tempfile
import unittest
from unittest.mock import patch


import gpu_queue as queue


def gpu(index="0", **changes):
    value = dict(index=index, uuid="GPU-" + index, name="A100", total_mib=81920,
                 used_mib=0, utilization_percent=0, compute_process_present=False)
    value.update(changes)
    return value


class QueueTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="gpu-queue-test-")
        self.root = Path(self.directory.name).resolve()
        self.assertEqual(self.root.parent, Path(tempfile.gettempdir()).resolve())
        self.state = self.root / "queue.json"
        self.now = 10000
        self.invoke("init", "--gpu-count", "1")

    def tearDown(self):
        self.directory.cleanup()

    def invoke(self, command, *args, inventory=None, error=None):
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream), patch.object(queue.time, "time", return_value=self.now), \
                patch.object(queue, "probe", return_value=inventory or [gpu()], side_effect=error) as probe:
            code = queue.main([command, "--state", str(self.state), *args])
        return code, json.loads(stream.getvalue()), probe.call_count

    def test_explicit_check_queries_each_time(self):
        code, first, calls = self.invoke("check")
        self.assertEqual((code, first["status"], calls), (0, "ready", 1))
        self.now += 1
        _, repeated, calls = self.invoke("check")
        self.assertEqual(calls, 1)
        self.assertEqual(repeated["status"], "ready")

    def test_fixed_interval_when_busy(self):
        for _ in range(5):
            _, result, calls = self.invoke("check", inventory=[gpu(compute_process_present=True)])
            self.assertEqual((result["status"], calls), ("waiting", 1))
            self.assertEqual(result["next_check_epoch"] - self.now, 900)
            self.now = result["next_check_epoch"]

    def test_query_error_is_not_idle(self):
        code, result, calls = self.invoke("check", error=RuntimeError("SSH unavailable"))
        self.assertEqual((code, result["status"], calls), (2, "error", 1))
        self.assertEqual(result["selected_gpus"], [])
        self.assertEqual(result["next_check_epoch"] - self.now, 900)
        self.now += 1
        _, _, calls = self.invoke("check")
        self.assertEqual(calls, 1)

    def make_ready(self):
        self.invoke("check")

    def test_claim_rechecks_and_is_single_use(self):
        self.make_ready()
        code, result, calls = self.invoke("claim")
        self.assertEqual((code, result["status"], calls), (0, "claimed", 1))
        self.assertTrue(result["claim_id"])
        _, again, calls = self.invoke("claim")
        self.assertEqual(calls, 0)
        self.assertFalse(again["probed"])
        self.assertEqual(result["claim_id"], again["claim_id"])

    def test_resources_taken_before_claim(self):
        self.make_ready()
        _, result, calls = self.invoke("claim", inventory=[gpu(compute_process_present=True)])
        self.assertEqual((result["status"], calls), ("waiting", 1))
        self.assertNotIn("claim_id", result)

    def test_unlimited_wait_and_cancel(self):
        self.now += 365 * 24 * 3600
        _, result, calls = self.invoke("check", inventory=[gpu(compute_process_present=True)])
        self.assertEqual((result["status"], calls), ("waiting", 1))
        self.invoke("cancel")
        _, result, calls = self.invoke("check")
        self.assertEqual((result["status"], calls), ("cancelled", 0))

    def test_config_change_requires_new_state(self):
        code, _, calls = self.invoke("init", "--gpu-count", "2")
        self.assertEqual((code, calls), (2, 0))
        self.assertEqual(json.loads(self.state.read_text())["config"]["gpu_count"], 1)

    def test_filters_and_process_evidence(self):
        state = json.loads(self.state.read_text())
        config = state["config"]
        config.update(gpu_name="A100", min_free_mib=40000, gpu_count=2)
        available = [gpu("0"), gpu("1", compute_process_present=True), gpu("2", name="V100"),
                     gpu("3", used_mib=5121), gpu("4", utilization_percent=20), gpu("5")]
        self.assertEqual([item["index"] for item in queue.select_gpus(available, config)], ["0", "5"])
        parsed = queue.parse_inventory("0, GPU-0, A100, 81920, 0, 0\n", "GPU-0, 12345\n")
        self.assertTrue(parsed[0]["compute_process_present"])
        with self.assertRaises(ValueError):
            queue.parse_inventory("0, GPU-0, A100, 81920, N/A, 0\n", "")

    def test_default_memory_limit_boundary(self):
        _, accepted, _ = self.invoke("check", inventory=[gpu(used_mib=5120)])
        self.assertEqual(accepted["status"], "ready")
        _, rejected, _ = self.invoke("check", inventory=[gpu(used_mib=5121)])
        self.assertEqual(rejected["status"], "waiting")

    def test_remote_executable_is_quoted_as_one_command(self):
        config = {"ssh_target": "gpu-host", "nvidia_smi": "/opt/NVIDIA tools/nvidia-smi"}
        query = "--query-compute-apps=gpu_uuid,pid"
        with patch.object(queue.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, "", "")) as run:
            queue.command(config, query)
        argv = run.call_args.args[0]
        self.assertEqual(argv[-2], "gpu-host")
        self.assertEqual(shlex.split(argv[-1]), [config["nvidia_smi"], query, "--format=csv,noheader,nounits"])


if __name__ == "__main__":
    unittest.main()
