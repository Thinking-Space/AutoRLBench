from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

from ..core.types import CommandResult, EnvironmentImage


@dataclass(frozen=True)
class CommandSpec:
    """Description of one command invocation, with no platform-specific options."""

    argv: Optional[Sequence[str]] = None
    shell: Optional[str] = None
    cwd: Optional[Path] = None
    env: Mapping[str, str] = field(default_factory=dict)
    timeout_s: Optional[int] = None
    user: Optional[str] = None
    live_log_name: Optional[str] = None
    live_log_to_stderr: bool = False

    def to_argv(self) -> List[str]:
        if self.argv is not None:
            return list(self.argv)
        if self.shell is not None:
            return ["sh", "-lc", self.shell]
        raise ValueError("command spec needs argv or shell")


@dataclass(frozen=True)
class EnvironmentArtifact:
    """A built environment, reusable across the agent and verifier stages."""

    backend: str
    reference: Optional[str] = None


@dataclass(frozen=True)
class SandboxInstance:
    """A running, mutable sandbox created from an environment artifact."""

    backend: str
    reference: str
    request: "SandboxRequest"


@dataclass(frozen=True)
class SandboxRequest:
    """Host resources needed for one prepare / agent / rerun / verifier run.

    A rerun stage sits between the agent and the verifier: under the solution protocol the agent
    only repairs code and never runs the experiment, so the harness re-runs it on the workspace
    the agent left behind, and grading only ever sees that fresh run's logs.
    """

    stage: str
    task_id: str
    host_task_root: Path
    host_visible_task: Path
    host_workspace: Path
    host_results: Path
    evaluation_level: str = "solution"
    host_oracle: Optional[Path] = None
    host_verifier: Optional[Path] = None
    # The task's environment/ directory, which holds prepare.sh. Mounted read-only in all
    # three stages because each may need it: prepare runs prepare.sh, the agent needs to know
    # what the environment has, and the verifier may check against the same manifest. Read-only
    # because no stage has a reason to modify the task definition in the repo.
    host_environment: Optional[Path] = None
    # Where prepare-stage outputs land (models, datasets, env files for the later two stages).
    # Writable in prepare, read-only for agent / verifier: grading must not depend on anything the agent can change.
    host_prepared: Optional[Path] = None
    # Download caches reused across runs and tasks, split into two roots by asset kind. **Mounted in all three
    # stages**: writable in prepare (where downloads happen), read-only for agent / verifier. The later stages
    # see them so debugging can reconcile them, not so tasks can bypass prepare: delivery is still only via
    # --dest, and on the harbor path these directories are not in the image at all.
    host_models: Optional[Path] = None
    host_datasets: Optional[Path] = None
    # Prepare only: the repo's docker/ directory, mounted read-only. It contains install-autorlbench-tooling.sh
    # and the autorlbench_fetch.py / autorlbench_dataset.py / autorlbench_datasets it installs. Prepare
    # runs this installer before prepare.sh so the fetch/dataset tools are the repo's current version, not
    # the copy baked in at image build time (if that copy has a bug, a git push cannot reach running jobs). Only
    # prepare uses it; agent / verifier do not mount it.
    host_tooling: Optional[Path] = None
    workdir: str = "workspace"
    env: Mapping[str, str] = field(default_factory=dict)
    network: str = "bridge"
    timeout_s: Optional[int] = None
    # metadata.environment.env: variables the task declares itself (e.g. CUDA_VISIBLE_DEVICES).
    # Per-task variables can only come from here.
    task_env: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "task_env", dict(self.task_env))
        if self.stage not in {"prepare", "agent", "rerun", "verifier"}:
            raise ValueError(
                "sandbox stage must be 'prepare', 'agent', 'rerun' or 'verifier'"
            )
        if self.workdir not in {"workspace", "task_root", "prepared"}:
            raise ValueError(
                "sandbox workdir must be 'workspace', 'task_root' or 'prepared'"
            )
        if self.stage == "prepare" and self.host_prepared is None:
            # A prepare stage with nowhere to write is wasted: the script succeeds and leaves nothing, which only
            # surfaces when the agent stage cannot find the model, and then it is reported as the task's fault.
            raise ValueError("the prepare stage needs host_prepared")


class SandboxRuntime(ABC):
    """Unified interface for execution platforms.

    Agents and the evaluator depend only on this interface, not directly on Docker or local processes.
    Backends such as Harbor or cloud sandboxes can implement the same methods.
    """

    name: str

    @property
    @abstractmethod
    def needs_environment_image(self) -> bool:
        """Whether this platform needs to prepare an image from metadata.environment.image."""

    @abstractmethod
    def ensure_environment(
        self, image: EnvironmentImage, *, rebuild: bool = False
    ) -> EnvironmentArtifact:
        """Prepare the environment image the task declares and return a reusable artifact.

        Images are shared by environment name: tasks with the same environment use one image, built only if missing.
        """

    @abstractmethod
    def path(self, request: SandboxRequest, name: str) -> str:
        """Return the actual path of a logical path on this platform."""

    @abstractmethod
    def run(
        self,
        request: SandboxRequest,
        spec: CommandSpec,
        *,
        artifact: Optional[EnvironmentArtifact] = None,
    ) -> CommandResult:
        """Run a command in the given stage."""

    def create_sandbox(
        self,
        request: SandboxRequest,
        *,
        artifact: EnvironmentArtifact,
    ) -> SandboxInstance:
        """Create and start a mutable sandbox for one workflow stage."""
        raise NotImplementedError(f"{self.name} does not support persistent sandboxes")

    def exec(
        self,
        sandbox: SandboxInstance,
        spec: CommandSpec,
    ) -> CommandResult:
        """Execute a command inside an existing sandbox."""
        raise NotImplementedError(f"{self.name} does not support sandbox exec")

    def close_sandbox(self, sandbox: SandboxInstance) -> None:
        """Stop and remove a sandbox, best effort after failures."""
        raise NotImplementedError(f"{self.name} does not support sandbox cleanup")

    def standard_env(self, request: SandboxRequest) -> Dict[str, str]:
        """Build the AutoRLBench environment variables every stage can rely on."""
        visible_task = self.path(request, "visible_task")
        workspace = self.path(request, "workspace")
        results = self.path(request, "results")
        if request.stage == "verifier":
            task_root = self.path(request, "task_root")
            visible = ""
            oracle = self.path(request, "oracle")
            verifier = self.path(request, "verifier")
        else:
            # prepare, agent and rerun all see only visible_task: prepare runs a script the task
            # supplies, and neither the agent nor the rerun has a reason to read the oracle or the
            # grading scripts -- the rerun runs the command the task declared, nothing more.
            task_root = visible_task
            visible = visible_task
            oracle = ""
            verifier = ""
        env = {
            "AUTORLBENCH_TASK_ID": request.task_id,
            "AUTORLBENCH_EVALUATION_LEVEL": request.evaluation_level,
            "AUTORLBENCH_TASK_ROOT": task_root,
            "AUTORLBENCH_VISIBLE_TASK": visible,
            "AUTORLBENCH_WORKSPACE": workspace,
            "AUTORLBENCH_ORACLE": oracle,
            "AUTORLBENCH_VERIFIER": verifier,
            "AUTORLBENCH_RESULTS": results,
            "AUTORLBENCH_STAGE": request.stage,
            "AUTORLBENCH_ENVIRONMENT": (
                self.path(request, "environment") if request.host_environment else ""
            ),
            "AUTORLBENCH_PREPARED": (
                self.path(request, "prepared") if request.host_prepared else ""
            ),
            # Both download cache roots are exported in all three stages. An empty string means "not configured
            # this time", consistent with how prepared is handled.
            "AUTORLBENCH_MODELS": (
                self.path(request, "models") if request.host_models else ""
            ),
            "AUTORLBENCH_DATASETS": (
                self.path(request, "datasets") if request.host_datasets else ""
            ),
        }
        env.update(self._network_policy_env(request))
        return env

    @staticmethod
    def _network_policy_env(request: SandboxRequest) -> Dict[str, str]:
        """Offline switches, set per stage.

        These used to be baked into the image ENV, which no longer works: prepare is the only stage that **should**
        have network, and image-level ENV treats every stage alike, so prepare would cut off its own network.

        Agent, rerun and verifier are the opposite: any silent network download could run the evaluation on weights
        nobody verified, without anyone noticing. So they are locked down here, and any data a task needs must be
        what the prepare stage delivered.
        """
        if request.stage == "prepare":
            # Prepare should not send anything to W&B either; it is only downloading.
            return {"WANDB_MODE": "offline", "WANDB_SILENT": "true"}
        return {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "WANDB_MODE": "offline",
            # Some frameworks still exit with an error under WANDB_MODE=offline when no key is set.
            "WANDB_API_KEY": "autorlbench-offline",
            "WANDB_SILENT": "true",
        }

    def _merged_spec(self, request: SandboxRequest, spec: CommandSpec) -> CommandSpec:
        """Merge standard and caller variables so agents need not handle paths themselves."""
        env = self.standard_env(request)
        # Task-declared env (metadata.environment.env) goes after standard variables and before the caller's:
        # it is more specific than framework defaults, but must not override what the caller set for this run.
        # AUTORLBENCH_* cannot appear here; EnvironmentPlan already rejects them.
        env.update({str(key): str(value) for key, value in request.task_env.items()})
        env.update({str(key): str(value) for key, value in request.env.items()})
        env.update({str(key): str(value) for key, value in spec.env.items()})
        timeout_s = spec.timeout_s if spec.timeout_s is not None else request.timeout_s
        return replace(spec, env=env, timeout_s=timeout_s)
