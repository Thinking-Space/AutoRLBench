from __future__ import annotations

import hashlib
import io
import tempfile
import unittest
import subprocess
from contextlib import redirect_stderr
from unittest.mock import patch
from pathlib import Path

from autorlbench.cli import build_parser
from autorlbench.core.types import (
    DATASETS_DIR,
    MODELS_DIR,
    PREPARED_DIR,
    AUTORLBENCH_ENVIRONMENT_DIR,
    CommandResult,
    EvaluationConfig,
)
from autorlbench.evaluation.evaluator import TaskEvaluator
from autorlbench.runtimes import (
    EnvironmentArtifact,
    SandboxInstance,
    SandboxRequest,
    SandboxRuntime,
)
from autorlbench.runtimes.base import stream_process_to_log
from autorlbench.runtimes.providers.docker import DockerRuntime


def result(label: str, returncode: int = 0) -> CommandResult:
    return CommandResult([label], returncode, label, "", False, 0.0, "/workspace")


class RecordingRuntime(SandboxRuntime):
    name = "recording"

    def __init__(self) -> None:
        self.events: list[str] = []

    @property
    def needs_environment_image(self) -> bool:
        return False

    def ensure_environment(self, image, *, rebuild=False):
        return EnvironmentArtifact(self.name)

    def path(self, request, name):
        return {
            "task_root": "/task",
            "visible_task": "/task",
            "workspace": "/workspace",
            "results": "/results",
            "environment": AUTORLBENCH_ENVIRONMENT_DIR,
            "prepared": PREPARED_DIR,
            "models": MODELS_DIR,
            "datasets": DATASETS_DIR,
        }[name]

    def run(self, request, spec, *, artifact=None):
        raise AssertionError("persistent lifecycle should be used")

    def create_sandbox(self, request, *, artifact):
        self.events.append(f"create:{request.stage}")
        return SandboxInstance(self.name, request.stage, request)

    def exec(self, sandbox, spec):
        self.events.append(f"exec:{sandbox.request.stage}")
        return result("exec")

    def close_sandbox(self, sandbox):
        self.events.append(f"close:{sandbox.request.stage}")


class SandboxLifecycleTests(unittest.TestCase):
    def test_sensitive_docker_env_uses_name_only(self):
        argv, process_env = DockerRuntime._docker_env(
            {"OPENAI_API_KEY": "secret", "AUTORLBENCH_STAGE": "agent"}
        )
        self.assertEqual(
            argv,
            ["-e", "AUTORLBENCH_STAGE=agent", "-e", "OPENAI_API_KEY"],
        )
        self.assertEqual(process_env["OPENAI_API_KEY"], "secret")
        self.assertNotIn("secret", " ".join(argv))

    def test_rerun_cli_defaults(self):
        args = build_parser().parse_args(["rerun", "task"])
        self.assertFalse(hasattr(args, "agent_backend"))
        self.assertEqual(args.backend, "docker")
        self.assertEqual(args.log_level, "quiet")
        self.assertIsNone(args.huggingface_mirror)
        self.assertIsNone(args.timeout_s)

    def test_rerun_cli_rejects_verifier_and_judge_flags(self):
        # Verifier and judge configuration moved to harbor-evaluate; the rerun
        # command runs no verifier, so those flags must not parse.
        for flag in (
            "--judge-model",
            "--model",
            "--verifier-env",
            "--verifier-cmd",
            "--evaluation-level",
        ):
            with self.assertRaises(SystemExit):
                build_parser().parse_args(["rerun", "task", flag, "value"])

    def test_rerun_cli_accepts_mirrors_and_caches(self):
        args = build_parser().parse_args(
            [
                "rerun",
                "task",
                "--huggingface-mirror",
                "https://hf-mirror.com/",
                "--models-dir",
                "/cache/models",
                "--datasets-dir",
                "/cache/datasets",
            ]
        )

        self.assertEqual(args.huggingface_mirror, "https://hf-mirror.com/")
        self.assertEqual(args.models_dir, "/cache/models")
        self.assertEqual(args.datasets_dir, "/cache/datasets")

    def test_failed_stage_summary_carries_the_error_tail(self):
        from autorlbench.evaluation.run_log import RunLog

        with tempfile.TemporaryDirectory() as temp_dir:
            run_log = RunLog.create("task", Path(temp_dir))
            TaskEvaluator("local")._log_stage_summary(
                run_log,
                "prepare",
                CommandResult(
                    ["false"],
                    2,
                    "setup output\n",
                    "installation failed\n",
                    False,
                    1.25,
                    "/workspace",
                ),
            )
            contents = (run_log.root / "run.log").read_text(encoding="utf-8")

        self.assertIn("[FAIL] prepare exit=2", contents)
        self.assertIn("error_tail: installation failed", contents)

    def test_docker_proxy_is_opt_in_for_all_stages(self):
        # _proxy_env applies to every stage alike; one stage proves the mechanism.
        request = SandboxRequest(
            stage="rerun",
            task_id="task",
            host_task_root=Path("/task"),
            host_visible_task=Path("/visible"),
            host_workspace=Path("/workspace"),
            host_results=Path("/results"),
        )
        with patch.dict(
            "os.environ",
            {"AUTORLBENCH_DOCKER_PROXY": "http://host.docker.internal:7890"},
            clear=False,
        ):
            env = DockerRuntime._proxy_env(request, {"AUTORLBENCH_STAGE": "rerun"})
        self.assertEqual(env["HTTP_PROXY"], "http://host.docker.internal:7890")
        self.assertEqual(env["HTTPS_PROXY"], "http://host.docker.internal:7890")
        self.assertEqual(env["ALL_PROXY"], "http://host.docker.internal:7890")

    def test_docker_stream_process_writes_live_stage_logs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            # `sh -c`, not `sh -lc`: a login shell sources the host's profile, and
            # a cluster that prints a quota banner at login would show up here as
            # extra stdout. What is under test is the streaming, not the shell.
            process = subprocess.Popen(
                ["sh", "-c", "echo setup-out; echo setup-error >&2"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            stdout, stderr, timed_out = DockerRuntime._stream_process(
                process,
                timeout_s=5,
                results_dir=Path(temp_dir),
                log_name="prepare",
            )
            self.assertFalse(timed_out)
            self.assertEqual(stdout, "setup-out\n")
            self.assertIn("setup-error\n", stderr)
            self.assertEqual(
                (Path(temp_dir) / "prepare.stdout.log").read_text(),
                stdout,
            )
            self.assertEqual(
                (Path(temp_dir) / "prepare.stderr.log").read_text(),
                stderr,
            )

    def test_docker_stream_log_file_combines_both_streams(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            log_path = Path(temp_dir) / "nested" / "prepare.log"
            process = subprocess.Popen(
                ["sh", "-c", "echo setup-out; echo setup-error >&2"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            stdout, stderr, timed_out = DockerRuntime._stream_log_file(
                process,
                timeout_s=5,
                log_path=log_path,
            )
            self.assertFalse(timed_out)
            # stdout and stderr stay separately captured for the CommandResult ...
            self.assertEqual(stdout, "setup-out\n")
            self.assertIn("setup-error\n", stderr)
            # ... while the persistent log file holds both, in one file, parent created.
            combined = log_path.read_text()
            self.assertIn("setup-out\n", combined)
            self.assertIn("setup-error\n", combined)

    def test_stream_log_file_mirrors_to_the_terminal_when_asked(self):
        # The rerun command's --log-level debug runs with live_log_to_stderr; the
        # live_log_path branch must honor it too, not only write the persistent file.
        with tempfile.TemporaryDirectory() as temp_dir:
            log_path = Path(temp_dir) / "rerun.log"
            process = subprocess.Popen(
                ["sh", "-c", "echo train-step; echo train-error >&2"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            with redirect_stderr(io.StringIO()) as terminal:
                stdout, stderr, timed_out = stream_process_to_log(
                    process,
                    timeout_s=5,
                    log_path=log_path,
                    to_stderr=True,
                )
            self.assertFalse(timed_out)
            self.assertIn("train-step", terminal.getvalue())
            self.assertIn("train-error", terminal.getvalue())
            # The persistent file still gets both streams.
            combined = log_path.read_text()
            self.assertIn("train-step\n", combined)
            self.assertIn("train-error\n", combined)

    def test_run_directory_prefix_is_readable_and_safe(self):
        prefix = TaskEvaluator._run_directory_prefix(
            "swe_slugfix/001: trial",
            "issue",
        )
        # The prefix is the bind-mount source the agent can read from /proc/mounts, so it
        # carries the task id only as a digest -- task ids often spell out the diagnosis.
        self.assertRegex(
            prefix,
            rf"^autorlbench-run-{hashlib.sha256('swe_slugfix/001: trial'.encode()).hexdigest()[:8]}"
            r"-issue-\d{8}-\d{6}-\d{6}-$",
        )
        self.assertNotIn("swe_slugfix", prefix)

    def test_huggingface_mirror_reaches_the_prepare_stage_as_hf_endpoint(self):
        # Downloads happen in the prepare stage, so mirror settings can only enter through that stage's env vars.
        env = TaskEvaluator._prepare_mirror_env(
            EvaluationConfig(huggingface_mirror="https://hf-mirror.com")
        )
        self.assertEqual(env, {"HF_ENDPOINT": "https://hf-mirror.com"})
        self.assertEqual(TaskEvaluator._prepare_mirror_env(EvaluationConfig()), {})

