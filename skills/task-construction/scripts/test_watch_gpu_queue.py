"""Offline event delivery tests; Codex message submission is mocked."""

import contextlib
import io
import json
from pathlib import Path
import subprocess
import tempfile
import time
import unittest
from unittest.mock import patch

import gpu_queue as queue
import watch_gpu_queue as watcher


THREAD = "01234567-89ab-cdef-0123-456789abcdef"


class EventWakeTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="gpu-event-test-")
        self.root = Path(self.directory.name).resolve()
        self.assertEqual(self.root.parent, Path(tempfile.gettempdir()).resolve())
        self.state = self.root / "state.json"
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(queue.main(["init", "--state", str(self.state), "--gpu-count", "1"]), 0)
        self.change(status="ready")

    def tearDown(self):
        self.directory.cleanup()

    def change(self, **changes):
        state = queue.load(self.state)
        state.update(changes)
        queue.save(self.state, state)

    def test_each_delivery_submits_a_message(self):
        with patch.object(watcher.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, "queued", "")) as send:
            watcher.deliver(self.state, THREAD, ["codex"], str(self.root))
            watcher.deliver(self.state, THREAD, ["codex"], str(self.root))
        self.assertEqual(send.call_count, 2)
        argv = send.call_args.args[0]
        self.assertEqual(argv[:5], ["codex", "queue", "--thread", THREAD, "--message"])
        self.assertIn(str(self.state), argv[5])

    def test_busy_or_error_never_calls_a_model(self):
        for status in ("waiting", "error"):
            with self.subTest(status=status):
                self.change(status=status, next_check_epoch=time.time() + 900)
                with patch.object(watcher.time, "sleep", side_effect=lambda seconds: self.change(status="ready")) as sleep, \
                        patch.object(watcher, "poll") as probe, \
                        patch.object(watcher, "deliver", return_value={"status": "delivered"}) as send:
                    result = watcher.watch(self.state, THREAD, ["codex"], str(self.root))
                self.assertEqual(sleep.call_count, 1)
                probe.assert_not_called()
                send.assert_called_once()
                self.assertTrue(result["woken"])

    def test_first_eligible_probe_wakes(self):
        clock = [10000]
        self.change(status="waiting", next_check_epoch=clock[0])
        inventory = [{"index": "0", "uuid": "GPU-0", "name": "A100", "total_mib": 81920,
                      "used_mib": 0, "utilization_percent": 0, "compute_process_present": False}]
        def sleep(seconds):
            clock[0] += seconds
        with contextlib.redirect_stdout(io.StringIO()), \
                patch.object(watcher.time, "time", side_effect=lambda: clock[0]), \
                patch.object(watcher.time, "sleep", side_effect=sleep), \
                patch.object(queue, "probe", return_value=inventory) as probe, \
                patch.object(watcher.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, "queued", "")) as send:
            result = watcher.watch(self.state, THREAD, ["codex"], str(self.root))
        self.assertTrue(result["woken"])
        self.assertEqual(probe.call_count, 1)
        self.assertEqual(clock[0], 10000)
        send.assert_called_once()

    def test_terminal_states_do_not_send(self):
        for status in ("claimed", "cancelled"):
            self.change(status=status)
            with patch.object(watcher, "deliver") as send:
                result = watcher.watch(self.state, THREAD, ["codex"], str(self.root))
            send.assert_not_called()
            self.assertFalse(result["woken"])

    def test_failed_delivery_can_be_called_again(self):
        with patch.object(watcher.subprocess, "run", return_value=subprocess.CompletedProcess([], 1, "", "unreachable session")) as send:
            with self.assertRaises(RuntimeError):
                watcher.deliver(self.state, THREAD, ["codex"], str(self.root))
            with self.assertRaises(RuntimeError):
                watcher.deliver(self.state, THREAD, ["codex"], str(self.root))
        self.assertEqual(send.call_count, 2)

    def test_preserves_state_changed_by_resumed_agent(self):
        def send(*args, **kwargs):
            self.change(status="claimed", claim_id="example-claim")
            return subprocess.CompletedProcess([], 0, "queued", "")
        with patch.object(watcher.subprocess, "run", side_effect=send):
            watcher.deliver(self.state, THREAD, ["codex"], str(self.root))
        state = queue.load(self.state)
        self.assertEqual(state["status"], "claimed")
        self.assertEqual(state["claim_id"], "example-claim")

    def test_invalid_target_does_not_start_cli(self):
        with contextlib.redirect_stderr(io.StringIO()), patch.object(watcher.subprocess, "run") as run:
            code = watcher.main(["--state", str(self.state), "--thread", "latest-session"])
        self.assertEqual(code, 2)
        run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
