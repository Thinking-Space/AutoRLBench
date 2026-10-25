from __future__ import annotations

import tempfile
import unittest
import subprocess
from unittest.mock import patch
from pathlib import Path

from autorlbench.cli import build_parser
from autorlbench.core.types import (
    DATASETS_DIR,
    MODELS_DIR,
    PREPARED_DIR,
    AUTORLBENCH_ENVIRONMENT_DIR,
    AgentSpec,
    CommandResult,
    EnvironmentPlan,
    EvaluationConfig,
)
from autorlbench.evaluation.evaluator import TaskEvaluator
from autorlbench.runtimes import (
    EnvironmentArtifact,
    SandboxInstance,
    SandboxRequest,
    SandboxRuntime,
)
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
            "oracle": "/task/oracle",
            "verifier": "/task/verifier",
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

    def test_network_defaults_allow_agent_and_verifier_access(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            request = SandboxRequest(
                stage="verifier",
                task_id="task",
                host_task_root=root,
                host_visible_task=root,
                host_workspace=root,
                host_results=root,
            )
        args = build_parser().parse_args(["evaluate", "task"])
        self.assertEqual(request.network, "bridge")
        self.assertEqual(EvaluationConfig().verifier_network, "bridge")
        self.assertFalse(hasattr(args, "agent_backend"))
        self.assertEqual(args.verifier_network, "bridge")
        judge_args = build_parser().parse_args(
            ["evaluate", "task", "--model", "agent-model", "--judge-model", "judge-model"]
        )
        self.assertEqual(judge_args.model, "agent-model")
        self.assertEqual(judge_args.judge_model, "judge-model")

    def test_evaluate_cli_accepts_stage_env_and_mirrors(self):
        args = build_parser().parse_args(
            [
                "evaluate",
                "task",
                "--verifier-env",
                "HTTPS_PROXY=http://proxy:7890",
                "--debian-mirror",
                "https://mirrors.tuna.tsinghua.edu.cn/",
                "--huggingface-mirror",
                "https://hf-mirror.com/",
            ]
        )

        self.assertEqual(args.verifier_env, ["HTTPS_PROXY=http://proxy:7890"])
        self.assertEqual(args.debian_mirror, "https://mirrors.tuna.tsinghua.edu.cn/")
        self.assertEqual(args.huggingface_mirror, "https://hf-mirror.com/")

    def test_docker_verifier_receives_judge_configuration(self):
        env = TaskEvaluator("docker")._verifier_judge_env(
            AgentSpec(
                kind="codex",
                model="agent-model",
                judge_model="judge-model",
                base_url="https://api.example.test/v1",
                api_key="secret-key",
            )
        )
        argv, process_env = DockerRuntime._docker_env(env)
        self.assertEqual(env["AUTORLBENCH_JUDGE_MODEL"], "judge-model")
        self.assertEqual(env["OPENAI_BASE_URL"], "https://api.example.test/v1")
        self.assertIn("-e", argv)
        self.assertNotIn("secret-key", " ".join(argv))
        self.assertEqual(process_env["OPENAI_API_KEY"], "secret-key")

    def test_verifier_env_is_merged_before_judge_configuration(self):
        class VerifierRuntime(RecordingRuntime):
            def create_sandbox(self, request, *, artifact):
                self.request = request
                return super().create_sandbox(request, artifact=artifact)

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for name in (
                "task",
                "visible",
                "workspace",
                "results",
                "oracle",
                "verifier",
                "environment",
                "prepared",
            ):
                (root / name).mkdir()
            runtime = VerifierRuntime()
            evaluator = TaskEvaluator("local")
            evaluator.runtime = runtime
            evaluator._run_verifier(
                type(
                    "Package",
                    (),
                    {
                        "task_id": "task",
                        "environment": EnvironmentPlan(),
                        # _environment_mount reads paths.environment.
                        "paths": type("Paths", (), {"environment": root / "environment"})(),
                    },
                )(),
                type(
                    "Evaluation",
                    (),
                    {
                        "root": root / "task",
                        "level": "solution",
                        "oracle": root / "oracle",
                        "verifier": root / "verifier",
                    },
                )(),
                type(
                    "Run",
                    (),
                    {
                        "visible_task": root / "visible",
                        "workspace": root / "workspace",
                        "results": root / "results",
                        "prepared": root / "prepared",
                        "models": root / "models",
                        "datasets": root / "datasets",
                    },
                )(),
                EvaluationConfig(
                    backend="local",
                    verifier_env={"HTTPS_PROXY": "http://proxy:7890"},
                ),
                EnvironmentArtifact("recording"),
                AgentSpec(
                    kind="codex",
                    model="gpt-5.5",
                    base_url="https://api.example.test/v1",
                ),
            )

        self.assertEqual(runtime.request.env["HTTPS_PROXY"], "http://proxy:7890")
        self.assertEqual(runtime.request.env["AUTORLBENCH_JUDGE_MODEL"], "gpt-5.5")
        self.assertEqual(
            runtime.request.env["OPENAI_BASE_URL"],
            "https://api.example.test/v1",
        )

    def test_failed_stage_summary_points_to_debug_logs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            results = Path(temp_dir)
            TaskEvaluator("local")._append_command_summary(
                results,
                "agent_setup",
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
            run_log = (results / "run.log").read_text(encoding="utf-8")

        self.assertIn("[FAIL] agent_setup exit=2", run_log)
        self.assertIn("error_tail: installation failed", run_log)
        self.assertIn(
            "logs: agent_setup.stdout.log, agent_setup.stderr.log",
            run_log,
        )

    def test_docker_proxy_is_opt_in_for_all_stages(self):
        request = SandboxRequest(
            stage="agent",
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
            env = DockerRuntime._proxy_env(request, {"AUTORLBENCH_STAGE": "agent"})
        self.assertEqual(env["HTTP_PROXY"], "http://host.docker.internal:7890")
        self.assertEqual(env["HTTPS_PROXY"], "http://host.docker.internal:7890")
        self.assertEqual(env["ALL_PROXY"], "http://host.docker.internal:7890")

        verifier_request = SandboxRequest(
            stage="verifier",
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
            verifier_env = DockerRuntime._proxy_env(
                verifier_request, {"AUTORLBENCH_STAGE": "verifier"}
            )
        self.assertEqual(
            verifier_env["HTTPS_PROXY"], "http://host.docker.internal:7890"
        )

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
                log_name="agent_setup",
            )
            self.assertFalse(timed_out)
            self.assertEqual(stdout, "setup-out\n")
            self.assertIn("setup-error\n", stderr)
            self.assertEqual(
                (Path(temp_dir) / "agent_setup.stdout.log").read_text(),
                stdout,
            )
            self.assertEqual(
                (Path(temp_dir) / "agent_setup.stderr.log").read_text(),
                stderr,
            )

    def test_run_directory_prefix_is_readable_and_safe(self):
        prefix = TaskEvaluator._run_directory_prefix(
            "swe_slugfix/001: trial",
            "issue",
        )
        self.assertRegex(
            prefix,
            r"^autorlbench-run-swe_slugfix-001-trial-issue-\d{8}-\d{6}-\d{6}-$",
        )

    def test_huggingface_mirror_reaches_the_prepare_stage_as_hf_endpoint(self):
        # Downloads happen in the prepare stage, so mirror settings can only enter through that stage's env vars.
        env = TaskEvaluator._prepare_mirror_env(
            EvaluationConfig(huggingface_mirror="https://hf-mirror.com")
        )
        self.assertEqual(env, {"HF_ENDPOINT": "https://hf-mirror.com"})
        self.assertEqual(TaskEvaluator._prepare_mirror_env(EvaluationConfig()), {})

