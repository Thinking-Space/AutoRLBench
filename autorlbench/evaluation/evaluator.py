from __future__ import annotations

import json
import re
import shlex
import shutil
import sys
import tempfile
from datetime import datetime
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, Optional

from ..core.types import (
    DOCKER_DIR,
    AUTORLBENCH_ENVIRONMENT_DIR,
    TOOLING_INSTALLER,
    AgentSpec,
    TaskPackage,
    CommandResult,
    EvaluationConfig,
    EvaluationPaths,
    EvaluationResult,
)
from ..tasks import SchemaError, load_task_package
from ..runtimes import (
    CommandSpec,
    EnvironmentArtifact,
    SandboxRequest,
    create_runtime,
)
from ..utils.filesystem import copy_tree, ensure_dir, tree_digest, write_text
from .run_spec import REPORT_FILENAME, load_run_spec


@dataclass(frozen=True)
class PreparedRun:
    """Isolated staging directory for one evaluation."""

    root: Path
    visible_task: Path
    workspace: Path
    results: Path
    # Where prepare-stage outputs land. Writable in prepare; agent / verifier mount the same directory read-only.
    prepared: Path
    # Download caches shared across runs and tasks. Writable in prepare, read-only for agent / verifier.
    models: Path
    datasets: Path


class TaskEvaluator:
    """Orchestrates the execution order of task, agent, runtime, and verifier."""

    def __init__(self, backend: str = "docker"):
        self.backend = backend
        self.log_level = "quiet"
        # The evaluator only picks a runtime and never handles Docker or local process details directly.
        self.runtime = create_runtime(backend)

    def load(self, task_root: str | Path, strict: bool = True) -> TaskPackage:
        return load_task_package(Path(task_root), strict=strict)

    def validate(self, task_root: str | Path, strict: bool = True) -> Dict[str, Any]:
        return self.load(task_root, strict=strict).report.to_dict()

    def evaluate(
        self,
        task_root: str | Path,
        config: EvaluationConfig | None = None,
        strict: bool = True,
    ) -> EvaluationResult:
        """Run one prepare + rerun + verifier evaluation.

        At the solution level the order is agent -> rerun -> verifier: the agent repairs code and
        submits report.md but **does not train**; the rerun is the harness re-running the experiment
        on the workspace the agent left behind, per the task's run_spec.json; and the verifier still
        only ever reads this pass's fresh logs under $AUTORLBENCH_RESULTS. The issue level has no
        rerun. This entry point itself runs no agent -- it is the verifier-side path -- so at the
        solution level it only gets past the report gate on a workspace that already carries one.
        """
        config = config or EvaluationConfig(backend=self.backend)
        self.log_level = config.log_level
        if self.log_level not in {"quiet", "debug"}:
            raise ValueError("log_level must be 'quiet' or 'debug'")
        if config.backend != self.backend:
            return TaskEvaluator(config.backend).evaluate(task_root, config=config, strict=strict)

        package = self.load(task_root, strict=strict)
        if not package.report.ok:
            raise SchemaError("task package failed validation")
        evaluation = self._select_evaluation(package, config.evaluation_level)

        agent_spec = config.agent
        run = self._prepare_run(package, evaluation, config)
        self._append_run_log(
            run.results,
            f"AutoRLBench task={package.task_id} backend={self.runtime.name} "
            f"evaluation_level={evaluation.level}",
        )
        try:
            artifact = None
            if self.runtime.needs_environment_image:
                self._append_run_log(
                    run.results,
                    f"[START] environment_prepare backend={self.runtime.name}",
                )
                image = package.environment.environment_image
                if image is None:
                    raise SchemaError(
                        f"{package.paths.task_toml}: metadata.environment.image "
                        f"is required by the {self.runtime.name} backend"
                    )
                # Images are shared by environment name: reuse if present, otherwise build from
                # docker/images/<image>.Dockerfile.
                artifact = self.runtime.ensure_environment(image)
                self._append_run_log(
                    run.results,
                    f"[OK] environment_prepare backend={self.runtime.name} "
                    f"artifact={artifact.reference or '-'}",
                )

            workspace_before_hash = tree_digest(package.paths.workspace)
            prepare_result = self._run_prepare(
                package, evaluation, run, config, artifact
            )
            if prepare_result is not None and prepare_result.returncode != 0:
                # If we kept going after prepare failed, the verifier would grade a workspace with no model,
                # and the final failure would be blamed on the task: a misleading lead that wastes hours.
                self._append_run_log(
                    run.results, "[SKIP] verifier reason=prepare_failed"
                )
                raise RuntimeError(
                    f"task {package.task_id!r}: prepare failed with exit "
                    f"{prepare_result.returncode}; see prepare.stderr.log"
                )

            # The solution level's submission gate: a repair is only finished once /workspace/report.md
            # exists. Without it the protocol was not completed, so neither the rerun nor the verifier
            # runs -- above all for tasks whose verifier drives the work itself, where skipping this
            # check would let a workspace that submitted nothing still collect a score.
            report_file = run.workspace / REPORT_FILENAME
            has_report = report_file.is_file() and report_file.stat().st_size > 0
            rerun_result = None
            verifier_result = None
            if evaluation.level == "solution" and not has_report:
                self._append_run_log(
                    run.results, f"[SKIP] rerun reason=no_{REPORT_FILENAME}"
                )
                self._append_run_log(
                    run.results, f"[SKIP] verifier reason=no_{REPORT_FILENAME}"
                )
            else:
                if has_report:
                    # Archive the report into results: the workspace is cleaned up after the run, and
                    # reviewing "what did the agent actually change" needs something to point at.
                    shutil.copy2(report_file, run.results / REPORT_FILENAME)
                rerun_result = self._run_rerun(
                    package, evaluation, run, config, artifact=artifact
                )
                verifier_result = self._run_verifier(
                    package,
                    evaluation,
                    run,
                    config,
                    artifact=artifact,
                    agent_spec=agent_spec,
                )
            workspace_after_hash = tree_digest(run.workspace)
            metrics = self._extract_metrics(run.results, verifier_result)
            rerun_ok = rerun_result is None or rerun_result.returncode == 0
            if evaluation.level == "solution" and not has_report:
                metrics["passed"] = False
                metrics["failure"] = (
                    f"agent did not submit {REPORT_FILENAME} in the workspace"
                )
            result = EvaluationResult(
                task_id=package.task_id,
                evaluation_level=evaluation.level,
                passed=rerun_ok and bool(metrics.get("passed", False)),
                score=self._extract_score(metrics) if rerun_ok else 0.0,
                metrics=metrics,
                report_dir=run.results,
                agent_spec=agent_spec,
                agent_setup=None,
                agent=None,
                rerun=rerun_result,
                verifier=verifier_result,
                workspace_before_hash=workspace_before_hash,
                workspace_after_hash=workspace_after_hash,
                metadata=package.metadata,
            )
            self._write_report(run.results / "report.json", result)
            self._append_run_log(
                run.results,
                f"[RESULT] passed={str(result.passed).lower()} score={result.score}",
            )
            return result
        except Exception as exc:
            self._append_run_log(
                run.results,
                f"[ERROR] orchestration {type(exc).__name__}: {exc}",
            )
            raise
        finally:
            if not config.keep_artifacts:
                shutil.rmtree(run.root, ignore_errors=True)

    @staticmethod
    def _select_evaluation(
        package: TaskPackage,
        level: str,
    ) -> EvaluationPaths:
        try:
            return package.paths.evaluations[level]
        except KeyError as exc:
            available = ", ".join(package.evaluation_levels) or "none"
            raise SchemaError(
                f"evaluation level {level!r} is unavailable; available levels: {available}"
            ) from exc

    def _prepare_run(
        self,
        package: TaskPackage,
        evaluation: EvaluationPaths,
        config: EvaluationConfig,
    ) -> PreparedRun:
        """Create the run directory, exposing only the instruction and a workspace copy to the agent."""
        parent = config.work_dir
        if parent is not None:
            parent = parent.resolve()
            parent.mkdir(parents=True, exist_ok=True)
        root = Path(
            tempfile.mkdtemp(
                prefix=self._run_directory_prefix(package.task_id, evaluation.level),
                dir=None if parent is None else str(parent),
            )
        ).resolve()
        visible_task = ensure_dir(root / "visible_task")
        workspace = ensure_dir(root / "workspace")
        results = ensure_dir(root / "results")
        prepared = ensure_dir(root / "prepared")
        # Caches only make sense outside the run directory: it is deleted afterwards, and models would be redownloaded.
        # EvaluationConfig guarantees both values are set (defaulting to models/ and datasets/ at the repo root),
        # so there is no "no cache this time" branch here.
        models = ensure_dir(config.models_dir)
        datasets = ensure_dir(config.datasets_dir)
        # The agent modifies this copy, not the task directory in the repo.
        copy_tree(package.paths.workspace, workspace)
        write_text(
            visible_task / "instruction.md",
            evaluation.instruction_md.read_text(encoding="utf-8"),
        )
        return PreparedRun(
            root, visible_task, workspace, results, prepared, models, datasets
        )

    @staticmethod
    def _run_directory_prefix(task_id: str, evaluation_level: str) -> str:
        """Build a readable, sortable prefix while retaining mkdtemp uniqueness."""
        safe_task_id = re.sub(r"[^A-Za-z0-9_.-]+", "-", task_id).strip("-._")
        safe_task_id = safe_task_id or "task"
        timestamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S-%f")
        return f"autorlbench-run-{safe_task_id}-{evaluation_level}-{timestamp}-"

    @staticmethod
    def _environment_mount(package: TaskPackage) -> Optional[Path]:
        """The task's environment/ directory, or nothing to mount if absent."""
        return package.paths.environment if package.paths.environment.is_dir() else None

    @staticmethod
    def _prepare_script(package: TaskPackage) -> Optional[str]:
        """Container path of prepare.sh, or None if the task has no such script.

        The script must live in the task's `environment/` directory, since that is the only copy mounted into the container.
        A path outside it would become "file not found" in the container, far from the real cause, so
        it is rejected here.
        """
        prepare = package.environment.prepare
        if prepare is None:
            return None
        if not prepare.is_file():
            raise SchemaError(
                f"task {package.task_id!r}: metadata.environment.prepare points to "
                f"{prepare}, which does not exist"
            )
        try:
            relative = prepare.resolve().relative_to(package.paths.environment.resolve())
        except ValueError as exc:
            raise SchemaError(
                f"task {package.task_id!r}: metadata.environment.prepare must stay under "
                f"{package.paths.environment}; only that directory is mounted into the "
                f"prepare container"
            ) from exc
        return f"{AUTORLBENCH_ENVIRONMENT_DIR}/{relative.as_posix()}"

    def _run_prepare(
        self,
        package: TaskPackage,
        evaluation: EvaluationPaths,
        run: PreparedRun,
        config: EvaluationConfig,
        artifact: Optional[EnvironmentArtifact],
    ) -> Optional[CommandResult]:
        """Run the task's prepare.sh: download models and datasets, write prepared/env.

        This is the **only** one of the three stages with network. It runs before the agent instead of being baked
        into the image because downloads are per task; baking them in would mean a tens-of-GB image layer per task,
        which is exactly what this redesign removes.
        """
        script = self._prepare_script(package)
        if script is None:
            return None
        request = SandboxRequest(
            stage="prepare",
            task_id=package.task_id,
            host_task_root=run.visible_task,
            host_visible_task=run.visible_task,
            host_workspace=run.workspace,
            host_results=run.results,
            evaluation_level=evaluation.level,
            host_environment=self._environment_mount(package),
            host_prepared=run.prepared,
            host_models=run.models,
            host_datasets=run.datasets,
            # The repo's docker/ is mounted read-only into the prepare container; used to install the latest fetch/dataset tools first.
            # If the directory is missing (docker/ may not ship with a wheel install), skip the mount; the call below also skips.
            host_tooling=DOCKER_DIR if DOCKER_DIR.is_dir() else None,
            env=self._prepare_mirror_env(config),
            # Prepare is the only stage that should have network; cut it too and nobody downloads the model.
            network="bridge",
            timeout_s=config.timeout_s,
            workdir="workspace",
            task_env=dict(package.environment.env),
        )
        if artifact is None:
            if self.runtime.needs_environment_image:
                raise ValueError("prepare runtime needs a built environment artifact")
            artifact = EnvironmentArtifact(backend=self.runtime.name)
        self._append_run_log(
            run.results, f"[START] prepare script={script}"
        )
        sandbox = self.runtime.create_sandbox(request, artifact=artifact)
        try:
            # The autorlbench_fetch_* / autorlbench_dataset commands prepare.sh calls are shims that exec
            # tools under /usr/local/libexec. The copy baked into the image may be stale, so if the repo's
            # docker/ is mounted (host_tooling), run the installer first to install the current version, then the script.
            # If the installer is absent (not mounted), skip it, matching the old behavior.
            shell = f"bash {shlex.quote(script)}"
            if request.host_tooling is not None:
                shell = (
                    f"if [ -f {shlex.quote(TOOLING_INSTALLER)} ]; then "
                    f"bash {shlex.quote(TOOLING_INSTALLER)}; fi; " + shell
                )
            result = self.runtime.exec(
                sandbox,
                CommandSpec(
                    shell=shell,
                    timeout_s=config.timeout_s,
                    live_log_name="prepare",
                    live_log_to_stderr=config.log_level == "debug",
                ),
            )
        finally:
            self.runtime.close_sandbox(sandbox)
        self._write_stage_logs(run.results, "prepare", result)
        self._append_command_summary(run.results, "prepare", result)
        return result

    @staticmethod
    def _prepare_mirror_env(config: EvaluationConfig) -> Dict[str, str]:
        """Pass mirror settings to the prepare stage.

        These used to be build args because downloads happened in the Dockerfile; once downloads moved to prepare,
        they had to move into this stage's environment too, or machines without direct access could not fetch any model.
        """
        env: Dict[str, str] = {}
        if config.huggingface_mirror:
            env["HF_ENDPOINT"] = config.huggingface_mirror
        return env

    def _run_rerun(
        self,
        package: TaskPackage,
        evaluation: EvaluationPaths,
        run: PreparedRun,
        config: EvaluationConfig,
        artifact: Optional[EnvironmentArtifact],
    ) -> Optional[CommandResult]:
        """Re-run the experiment on the workspace the agent left behind, per run_spec.json.

        Returns None when the task has no run_spec.json (a verifier that drives the work itself, or
        a pure protocol repair such as verl_aime25_single_sample_protocol): those tasks go straight
        from the submitted report to the verifier, and the agent-side constraints are carried by
        instruction.md alone. A run_spec.json that exists but is invalid is a task-definition bug,
        and load_run_spec raises RunSpecError on the spot -- not halfway into a training run.
        """
        if evaluation.level != "solution":
            return None
        spec = load_run_spec(evaluation.root, package.task_id)
        if spec is None:
            return None
        # Grading only ever sees this pass: clear the log roots the spec declares first. Old logs an
        # agent left behind by training against the protocol do not count.
        for log_root in spec.log_roots:
            shutil.rmtree(run.results / log_root, ignore_errors=True)
        timeout_s = spec.timeout_s if spec.timeout_s is not None else config.timeout_s
        # References like $AUTORLBENCH_RESULTS in spec.env have to become this backend's real paths
        # (/results under docker, a host directory under local). standard_env only reads the request's
        # path fields and never its env, so building the request before expanding is not circular.
        request = SandboxRequest(
            stage="rerun",
            task_id=package.task_id,
            host_task_root=run.visible_task,
            host_visible_task=run.visible_task,
            host_workspace=run.workspace,
            host_results=run.results,
            evaluation_level=evaluation.level,
            host_environment=self._environment_mount(package),
            # The rerun reads what prepare delivered too, mounted read-only: the agent cannot change it,
            # which is the only thing that makes "the experiment ran on prepare's assets" hold.
            host_prepared=run.prepared,
            env={},
            # Ray clusters and local retrieval servers inside an experiment need loopback; being offline
            # is enforced by HF_HUB_OFFLINE and friends in _network_policy_env, not by cutting the network.
            network="bridge",
            timeout_s=timeout_s,
            workdir="workspace",
            task_env=dict(package.environment.env),
        )
        resolved = {
            key: value
            for key, value in self.runtime.standard_env(request).items()
            if key.startswith("AUTORLBENCH_") and value
        }
        request = replace(request, env=spec.expand_env(resolved))
        if artifact is None:
            if self.runtime.needs_environment_image:
                raise ValueError("rerun runtime needs a built environment artifact")
            artifact = EnvironmentArtifact(backend=self.runtime.name)
        self._append_run_log(run.results, "[START] rerun")
        sandbox = self.runtime.create_sandbox(request, artifact=artifact)
        try:
            result = self.runtime.exec(
                sandbox,
                CommandSpec(
                    shell=spec.command,
                    timeout_s=timeout_s,
                    # Training runs for hours: stream to disk like prepare does, so when it dies
                    # rerun.stdout.log shows which step it died on.
                    live_log_name="rerun",
                    live_log_to_stderr=config.log_level == "debug",
                ),
            )
        finally:
            self.runtime.close_sandbox(sandbox)
        self._write_stage_logs(run.results, "rerun", result)
        self._append_command_summary(run.results, "rerun", result)
        return result

    def _run_verifier(
        self,
        package: TaskPackage,
        evaluation: EvaluationPaths,
        run: PreparedRun,
        config: EvaluationConfig,
        artifact: Optional[EnvironmentArtifact],
        agent_spec: Optional[AgentSpec],
    ) -> Any:
        verifier_env = dict(config.verifier_env)
        verifier_env.update(self._verifier_judge_env(agent_spec))
        request = SandboxRequest(
            stage="verifier",
            task_id=package.task_id,
            host_task_root=evaluation.root,
            host_visible_task=run.visible_task,
            host_workspace=run.workspace,
            host_results=run.results,
            evaluation_level=evaluation.level,
            host_oracle=evaluation.oracle,
            host_verifier=evaluation.verifier,
            env=verifier_env,
            network=config.verifier_network,
            timeout_s=config.timeout_s,
            workdir="task_root",
            host_environment=self._environment_mount(package),
            # The verifier also needs to see what prepare delivered: grading scripts often read a checkpoint
            # or match samples against a dataset. Mounted read-only, so what it reads is always what prepare wrote.
            host_prepared=run.prepared,
            # Download caches are mounted read-only: useful when grading checks against the original model or dataset,
            # and read-only ensures it is not checking a copy the agent can change.
            host_models=run.models,
            host_datasets=run.datasets,
            # Task-declared env goes to both stages: the verifier often loads the checkpoint on the same number of GPUs,
            # and without CUDA_VISIBLE_DEVICES it would not see the same machine the agent did.
            task_env=dict(package.environment.env),
        )
        command = CommandSpec(
            shell=config.verifier_cmd or 'bash "$AUTORLBENCH_VERIFIER/test.sh"',
            timeout_s=config.timeout_s,
        )
        if artifact is None:
            if self.runtime.needs_environment_image:
                raise ValueError("verifier runtime needs a built environment artifact")
            artifact = EnvironmentArtifact(backend=self.runtime.name)
        sandbox = self.runtime.create_sandbox(request, artifact=artifact)
        try:
            self._append_run_log(run.results, "[START] verifier")
            result = self.runtime.exec(sandbox, command)
            self._write_stage_logs(run.results, "verifier", result)
            self._append_command_summary(run.results, "verifier", result)
            return result
        finally:
            self.runtime.close_sandbox(sandbox)

    def _append_run_log(self, results_dir: Path, line: str) -> None:
        path = results_dir / "run.log"
        existing = path.read_text(encoding="utf-8") if path.exists() else ""
        write_text(path, existing + line.rstrip() + "\n")
        if self.log_level == "debug":
            print(line.rstrip(), file=sys.stderr, flush=True)

    @staticmethod
    def _write_stage_logs(
        results_dir: Path,
        stage: str,
        result: CommandResult,
    ) -> None:
        write_text(results_dir / f"{stage}.stdout.log", result.stdout or "")
        write_text(results_dir / f"{stage}.stderr.log", result.stderr or "")

    def _append_command_summary(
        self,
        results_dir: Path,
        stage: str,
        result: CommandResult,
    ) -> None:
        status = "OK" if result.returncode == 0 and not result.timed_out else "FAIL"
        self._append_run_log(
            results_dir,
            f"[{status}] {stage} exit={result.returncode} "
            f"timed_out={str(result.timed_out).lower()} "
            f"duration={result.duration_s:.2f}s",
        )
        error = (result.stderr or "").strip()
        if not error and result.returncode != 0:
            error = (result.stdout or "").strip()
        if error:
            tail = " | ".join(error.splitlines()[-5:])
            self._append_run_log(results_dir, f"  error_tail: {tail}")
        if status == "FAIL":
            self._append_run_log(
                results_dir,
                f"  logs: {stage}.stdout.log, {stage}.stderr.log",
            )

    def _verifier_judge_env(self, agent_spec: Optional[AgentSpec]) -> Dict[str, str]:
        """Expose the configured model endpoint to API-backed verifiers."""
        if agent_spec is None:
            return {}
        env = {
            "AUTORLBENCH_JUDGE_PROVIDER": "openai-compatible",
        }
        judge_model = agent_spec.judge_model or agent_spec.model
        if judge_model:
            env["AUTORLBENCH_JUDGE_MODEL"] = judge_model
        env["OPENAI_BASE_URL"] = agent_spec.base_url or "https://api.openai.com/v1"
        if agent_spec.api_key:
            env["OPENAI_API_KEY"] = agent_spec.api_key
        return env

    def _write_report(self, path: Path, result: EvaluationResult) -> None:
        write_text(path, json.dumps(result.to_dict(), indent=2, ensure_ascii=False))

    def _extract_metrics(self, results_dir: Path, verifier_result: Any) -> Dict[str, Any]:
        """Prefer the verifier's JSON result; otherwise fall back to the exit code."""
        metrics: Dict[str, Any] = {}
        for filename in ("result.json", "metrics.json"):
            path = results_dir / filename
            if not path.exists():
                continue
            try:
                with open(path, "r", encoding="utf-8") as file:
                    payload = json.load(file)
            except Exception as exc:
                metrics["result_parse_error"] = str(exc)
                continue
            if not isinstance(payload, dict):
                continue
            nested = payload.get("metrics")
            if isinstance(nested, dict):
                metrics.update(nested)
            for key in ("passed", "score"):
                if key in payload:
                    metrics[key] = payload[key]
            for key, value in payload.items():
                if key not in {"metrics", "passed", "score"} and key not in metrics:
                    metrics[key] = value
        if verifier_result is None:
            metrics["passed"] = False
        else:
            metrics.setdefault("passed", verifier_result.returncode == 0)
        return metrics

    @staticmethod
    def _extract_score(metrics: Dict[str, Any]) -> Optional[float]:
        if isinstance(metrics.get("score"), (int, float)):
            return float(metrics["score"])
        if metrics.get("passed") is True:
            return 1.0
        if metrics.get("passed") is False:
            return 0.0
        return None
