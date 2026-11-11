from __future__ import annotations

import hashlib
import json
import shlex
import shutil
import tempfile
from datetime import datetime
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, Optional

from ..core.types import (
    AUTORLBENCH_ENVIRONMENT_DIR,
    DEFAULT_OUTPUT_DIR,
    TaskPackage,
    CommandResult,
    EnvironmentImage,
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
from ..utils.filesystem import copy_tree, ensure_dir, link_tree, tree_digest, write_text
from .run_log import RunLog
from .run_spec import load_run_spec


@dataclass(frozen=True)
class PreparedRun:
    """Isolated staging directory for one evaluation."""

    root: Path
    visible_task: Path
    workspace: Path
    results: Path
    # Where prepare-stage outputs land. Writable in prepare, read-only for the rerun.
    prepared: Path
    # Download caches shared across runs and tasks. Writable in prepare only.
    models: Path
    datasets: Path
    # Hard-linked copy of the task's environment/, staged in place of the repo path: the repo path
    # spells out the task id, and every bind-mount source is readable from /proc/mounts in the
    # sandbox. None when the task has no environment/ directory.
    environment: Optional[Path] = None


def load_valid_package(task_root: str | Path, strict: bool = True) -> TaskPackage:
    """Load a task package and refuse one whose validation report has errors."""
    package = load_task_package(Path(task_root), strict=strict)
    if not package.report.ok:
        raise SchemaError(f"task package failed validation: {package.paths.root}")
    return package


def select_evaluation(package: TaskPackage, level: str) -> EvaluationPaths:
    """Pick the evaluation directory for a level, listing the available ones when it is missing."""
    try:
        return package.paths.evaluations[level]
    except KeyError as exc:
        available = ", ".join(package.evaluation_levels) or "none"
        raise SchemaError(
            f"evaluation level {level!r} is unavailable; available levels: {available}"
        ) from exc


def required_environment_image(package: TaskPackage) -> EnvironmentImage:
    """The task's environment image; an image-backed backend has no fallback without it."""
    image = package.environment.environment_image
    if image is None:
        raise SchemaError(
            f"task {package.task_id!r} declares no environment.image; "
            "an image-backed backend has nothing to run"
        )
    return image


class TaskEvaluator:
    """Orchestrates a task's prepare and rerun stages on its runtime."""

    def __init__(self, backend: str = "docker"):
        self.backend = backend
        self.log_level = "quiet"
        # The evaluator only picks a runtime and never handles Docker or local process details directly.
        self.runtime = create_runtime(backend)

    def load(self, task_root: str | Path, strict: bool = True) -> TaskPackage:
        return load_task_package(Path(task_root), strict=strict)

    def validate(self, task_root: str | Path, strict: bool = True) -> Dict[str, Any]:
        return self.load(task_root, strict=strict).report.to_dict()

    def rerun(
        self,
        task_root: str | Path,
        config: EvaluationConfig | None = None,
        strict: bool = True,
    ) -> EvaluationResult:
        """Run one prepare + rerun pass on the task's original workspace.

        The submitter-facing environment check: no agent, no verifier, and no report
        gate -- the point is to reproduce the buggy run the task ships, so the
        workspace is used exactly as the task author left it. `passed` is
        stage-level: prepare succeeded and the run_spec.json experiment executed
        without timing out (a harness-level interruption raises instead). The
        rerun's own exit code is recorded, not judged -- a buggy run is expected
        to fail, so its exit code says nothing about the environment.

        Logs are laid out like harbor-evaluate's: a persistent run log directory at
        <work_dir>/<task_id>/<timestamp>/ (run.log, prepare.log, rerun.log, command.sh,
        report.json), with the staging (workspace copy, prepared, environment) in a
        digest-named sibling that is deleted after the run. The experiment's logs
        (results/) always move into the run log directory first, and so does the
        workspace when keep_artifacts is set.
        """
        config = config or EvaluationConfig(backend=self.backend)
        self.log_level = config.log_level
        if self.log_level not in {"quiet", "debug"}:
            raise ValueError("log_level must be 'quiet' or 'debug'")
        if config.backend != self.backend:
            return TaskEvaluator(config.backend).rerun(task_root, config=config, strict=strict)

        package = load_valid_package(task_root, strict=strict)
        isolated_stages = package.metadata.get("security", {}).get("requires_isolated_stages", False)
        if isolated_stages and self.backend != "docker":
            raise SchemaError("This task requires separate Docker stages; local execution is unsupported")
        # Fail before any staging or download: a task with no run_spec.json declares no
        # experiment for the harness to rerun, so there is nothing for this pass to check.
        if load_run_spec(package.paths.root, package.task_id) is None:
            raise SchemaError(
                f"task {package.task_id!r}: no run_spec.json, so there is no experiment "
                "to rerun and nothing for this command to check"
            )
        # run_spec.json is level-neutral, but _prepare_run stages an instruction.md, so a
        # level still has to be picked. The CLI has no level flag and defaults to solution;
        # issue-only tasks are valid submitter checks, so use their issue instruction when
        # no solution level exists. An explicitly requested missing issue level still fails.
        try:
            evaluation = select_evaluation(package, config.evaluation_level)
        except SchemaError:
            if config.evaluation_level != "solution" or "issue" not in package.paths.evaluations:
                raise
            evaluation = select_evaluation(package, "issue")

        output_root = (
            config.work_dir if config.work_dir is not None else DEFAULT_OUTPUT_DIR / "rerun"
        )
        run_log = RunLog.create(package.task_id, output_root)
        run_log.write_command(
            config.invocation_argv,
            Path.cwd().resolve(),
            title="autorlbench rerun invocation",
        )
        # Staging goes under the same parent as the run log, so the salvage at the end is a
        # rename, not a cross-device copy of a possibly tens-of-GB workspace.
        config = replace(config, work_dir=Path(output_root))
        run = self._prepare_run(package, evaluation, config)
        run_log.log(
            f"task={package.task_id} backend={self.runtime.name} "
            f"evaluation_level={evaluation.level}"
        )
        try:
            artifact = None
            if self.runtime.needs_environment_image:
                run_log.log(f"[START] environment_prepare backend={self.runtime.name}")
                image = required_environment_image(package)
                # Images are shared by environment name: reuse if present, otherwise pull the
                # digest-pinned reference from docker/images/<image>.Dockerfile.
                artifact = self.runtime.ensure_environment(image)
                run_log.log(
                    f"[OK] environment_prepare backend={self.runtime.name} "
                    f"artifact={artifact.reference or '-'}"
                )

            workspace_before_hash = tree_digest(package.paths.workspace)
            run_log.log("[START] prepare")
            prepare_result = self._run_prepare(
                package,
                evaluation,
                run,
                config,
                artifact,
                live_log_path=run_log.prepare_log_path,
            )
            if prepare_result is not None:
                self._log_stage_summary(run_log, "prepare", prepare_result)
            if prepare_result is not None and prepare_result.returncode != 0:
                # If we kept going after prepare failed, the rerun would die on a workspace with
                # no model, and the final failure would be blamed on the task's environment: a
                # misleading lead that wastes hours.
                run_log.log("[SKIP] rerun reason=prepare_failed")
                raise RuntimeError(
                    f"task {package.task_id!r}: prepare failed with exit "
                    f"{prepare_result.returncode}; see prepare.log in {run_log.root}"
                )

            run_log.log("[START] rerun")
            rerun_result = self._run_rerun(
                package,
                evaluation,
                run,
                config,
                artifact=artifact,
                live_log_path=run_log.stage_log_path("rerun"),
            )
            if rerun_result is None:
                # Unreachable: the fail-fast check above rejected spec-less tasks.
                raise SchemaError(
                    f"task {package.task_id!r}: no run_spec.json, so there is no experiment to rerun"
                )
            self._log_stage_summary(run_log, "rerun", rerun_result)
            workspace_after_hash = tree_digest(run.workspace)
            metrics = {
                "exit_code": rerun_result.returncode,
                "timed_out": rerun_result.timed_out,
                "duration_s": rerun_result.duration_s,
            }
            result = EvaluationResult(
                task_id=package.task_id,
                evaluation_level=evaluation.level,
                passed=not rerun_result.timed_out,
                score=None,
                metrics=metrics,
                report_dir=run_log.root,
                rerun=rerun_result,
                workspace_before_hash=workspace_before_hash,
                workspace_after_hash=workspace_after_hash,
                metadata=package.metadata,
            )
            self._write_report(run_log.root / "report.json", result)
            run_log.log(
                f"[RESULT] passed={str(result.passed).lower()} "
                f"exit_code={rerun_result.returncode}"
            )
            return result
        except Exception as exc:
            run_log.error(f"orchestration {type(exc).__name__}: {exc}")
            raise
        finally:
            self._salvage_run(run, run_log, keep_workspace=config.keep_artifacts)

    def _prepare_run(
        self,
        package: TaskPackage,
        evaluation: EvaluationPaths,
        config: EvaluationConfig,
    ) -> PreparedRun:
        """Create the run directory: an instruction copy and a workspace copy for the rerun."""
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
        # The repo's tasks/<type>/<id>/environment path spells out the task id, and each stage
        # bind-mounts this directory -- /proc/mounts inside the sandbox shows every source. Stage
        # a hard-linked copy inside the digest-named run directory instead; the mounts are
        # read-only, so the links are never written through.
        environment = (
            link_tree(package.paths.environment, root / "environment")
            if package.paths.environment.is_dir()
            else None
        )
        write_text(
            visible_task / "instruction.md",
            evaluation.instruction_md.read_text(encoding="utf-8"),
        )
        return PreparedRun(
            root, visible_task, workspace, results, prepared, models, datasets, environment
        )

    @staticmethod
    def _run_directory_prefix(task_id: str, evaluation_level: str) -> str:
        """Build a sortable prefix while retaining mkdtemp uniqueness.

        The task id is deliberately absent: the run directory is the bind-mount source for
        the sandbox, so /proc/mounts inside the container would show it -- and many task ids
        spell out the diagnosis. An 8-hex digest keeps runs traceable to their task
        (sha256 of the id) without handing the agent the answer.
        """
        digest = hashlib.sha256(task_id.encode("utf-8")).hexdigest()[:8]
        timestamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S-%f")
        return f"autorlbench-run-{digest}-{evaluation_level}-{timestamp}-"

    @staticmethod
    def _prepare_script(
        package: TaskPackage,
        environment_root: str | Path = AUTORLBENCH_ENVIRONMENT_DIR,
    ) -> Optional[str]:
        """Runtime path of prepare.sh, or None if the task has no such script.

        The script must live in the task's `environment/` directory, since that is the only copy mounted into the container.
        A path outside it would become "file not found" in the container, far from the real cause, so
        it is rejected here. Docker passes the container mount root; the local backend
        passes the host staging directory, where the same environment tree lives.
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
        return str(Path(environment_root) / relative)

    def _run_prepare(
        self,
        package: TaskPackage,
        evaluation: EvaluationPaths,
        run: PreparedRun,
        config: EvaluationConfig,
        artifact: Optional[EnvironmentArtifact],
        live_log_path: Optional[Path] = None,
    ) -> Optional[CommandResult]:
        """Run the task's prepare.sh: download models and datasets, write prepared/env.

        This is the **only** stage with network. It runs as a stage instead of being baked
        into the image because downloads are per task; baking them in would mean a tens-of-GB image layer per task,
        which is exactly what this redesign removes.

        live_log_path, when given, streams the stage's stdout and stderr combined into that one
        file as prepare runs -- the harbor path points it at the run's persistent prepare.log,
        which outlives the deleted staging.
        """
        if package.environment.prepare is None:
            return None
        request = SandboxRequest(
            stage="prepare",
            task_id=package.task_id,
            host_task_root=run.visible_task,
            host_visible_task=run.visible_task,
            host_workspace=run.workspace,
            host_results=run.results,
            evaluation_level=evaluation.level,
            host_environment=run.environment,
            host_prepared=run.prepared,
            host_models=run.models,
            host_datasets=run.datasets,
            env=self._prepare_mirror_env(config),
            # Prepare is the only stage that should have network; cut it too and nobody downloads the model.
            network="bridge",
            timeout_s=config.timeout_s,
            workdir="workspace",
            task_env=dict(package.environment.env),
            # Prepare only downloads and needs no GPU; the count is recorded anyway so the request
            # mirrors the task's declaration, and the backend decides which stages get devices.
            gpus=package.environment.gpus,
        )
        script = self._prepare_script(
            package, environment_root=self.runtime.path(request, "environment")
        )
        if script is None:
            return None
        if artifact is None:
            if self.runtime.needs_environment_image:
                raise ValueError("prepare runtime needs a built environment artifact")
            artifact = EnvironmentArtifact(backend=self.runtime.name)
        sandbox = self.runtime.create_sandbox(request, artifact=artifact)
        try:
            # The fetch/dataset tools prepare.sh calls (fetch_hf.py and friends) travel with the
            # task's environment/ directory, which is mounted read-only; the script invokes them
            # with a plain `python3`, so nothing needs installing here anymore.
            shell = f"bash {shlex.quote(script)}"
            result = self.runtime.exec(
                sandbox,
                CommandSpec(
                    shell=shell,
                    timeout_s=config.timeout_s,
                    live_log_name="prepare",
                    live_log_to_stderr=config.log_level == "debug",
                    live_log_path=live_log_path,
                ),
            )
        finally:
            self.runtime.close_sandbox(sandbox)
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
        live_log_path: Optional[Path] = None,
    ) -> Optional[CommandResult]:
        """Re-run the experiment on the workspace, per run_spec.json.

        Returns None when the task has no run_spec.json (a verifier that drives the work itself, or
        a pure protocol repair). A run_spec.json that exists but is invalid is a task-definition
        bug, and load_run_spec raises RunSpecError on the spot -- not halfway into a training run.

        live_log_path, when given, streams the experiment's stdout and stderr combined into that
        one file as it runs -- the rerun command points it at the run's persistent rerun.log.
        """
        spec = load_run_spec(package.paths.root, package.task_id)
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
            host_environment=run.environment,
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
            # The rerun runs real training: it takes the cards the task declares.
            gpus=package.environment.gpus,
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
        sandbox = self.runtime.create_sandbox(request, artifact=artifact)
        try:
            result = self.runtime.exec(
                sandbox,
                CommandSpec(
                    shell=spec.command,
                    timeout_s=timeout_s,
                    # Training runs for hours: stream to disk like prepare does, so when it dies
                    # rerun.log shows which step it died on.
                    live_log_name="rerun",
                    live_log_to_stderr=config.log_level == "debug",
                    live_log_path=live_log_path,
                ),
            )
        finally:
            self.runtime.close_sandbox(sandbox)
        return result

    @staticmethod
    def _log_stage_summary(run_log: RunLog, stage: str, result: CommandResult) -> None:
        status = "OK" if result.returncode == 0 and not result.timed_out else "FAIL"
        run_log.log(
            f"[{status}] {stage} exit={result.returncode} "
            f"timed_out={str(result.timed_out).lower()} "
            f"duration={result.duration_s:.2f}s"
        )
        error = (result.stderr or "").strip()
        if not error and result.returncode != 0:
            error = (result.stdout or "").strip()
        if error:
            tail = " | ".join(error.splitlines()[-5:])
            run_log.log(f"  error_tail: {tail}")

    @staticmethod
    def _salvage_run(run: PreparedRun, run_log: RunLog, keep_workspace: bool) -> None:
        """Move what must outlive the run out of the staging, then delete the staging.

        The staging directory carries a digest instead of the task id because it is the
        bind-mount source for the sandbox (see _run_directory_prefix); the run log
        directory is host-only and never mounted, so the task id in its path is safe.
        The experiment's logs under results/ are the point of the command, so they always
        move; the workspace only when keep_artifacts is set. A failed move keeps the
        staging on disk rather than deleting the evidence.
        """
        moves: list[tuple[str, Path]] = [("results", run.results)]
        if keep_workspace:
            moves.append(("workspace", run.workspace))
        salvaged = True
        for name, source in moves:
            if not source.is_dir():
                continue
            destination = run_log.root / name
            try:
                if destination.exists():
                    shutil.rmtree(destination)
                shutil.move(str(source), str(destination))
            except OSError as exc:
                run_log.error(f"salvaging {name} from {source} failed: {exc}")
                salvaged = False
        if salvaged:
            shutil.rmtree(run.root, ignore_errors=True)
        else:
            run_log.error(f"staging kept for manual recovery: {run.root}")

    def _write_report(self, path: Path, result: EvaluationResult) -> None:
        write_text(path, json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
