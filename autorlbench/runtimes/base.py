from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

from ..core.types import CommandResult, EnvironmentImage


# This logic used to live in the image's /etc/profile.d/autorlbench-activate.sh, sourced automatically by a
# login shell (`sh -lc`). Nothing is baked into images anymore: the harness prepends it to every shell
# command at runtime (see CommandSpec.to_argv), so one push takes effect without rebuilding a dozen images.
#
# It does three things:
#   0. Restore the image's PATH (section 0, ported from the old /etc/profile.d hook). The local docker
#      backend indeed no longer uses a login shell and /etc/profile does not reset PATH, but the Harbor
#      backend's commands travel over ssh as a non-login `bash -c`, where PATH is sshd's default list,
#      missing the image ENV additions like /opt/venv/bin and /usr/local/cuda/bin -- ray / nvidia-smi
#      then come up "command not found". The Dockerfile still records the image PATH in
#      /etc/autorlbench-image-path (old images have it too); directories missing from PATH are prepended
#      in image order. A missing file or an already-complete PATH makes this a no-op.
#   1. Export the KEY=VALUE lines prepare wrote to $AUTORLBENCH_PREPARED/env (PYTHONPATH and friends)
#      for the agent / verifier / rerun stages. Only bare NAME=... lines are honored, blocking command
#      substitution -- a task must not be able to smuggle arbitrary shell into every stage.
#   2. Point pip's --user landing site at the agent's own workspace (agent / rerun stages).
#
# The trailing `unset BASH_ENV` sits outside the if, unconditionally: whatever was exported above is
# inherited by child processes without nested bash re-sourcing it, and Ray starts its actor workers
# exactly via `bash -c` (os.execvp in setup_worker.py); a replay would overwrite the prepared/env
# PYTHONPATH back to the mcore-only one, and `import runtime_lowmem` (which lives in /workspace/scripts)
# would stop resolving inside verl task workers. The local docker backend never sets BASH_ENV, so this
# line is a no-op there.
#
# POSIX sh: this is concatenated into `sh -c`, so no `return` (that only works while being sourced). The
# if-block lets control flow fall through to the real command appended after it. The Harbor backend does
# not go through here; it materializes the same body into a file loaded via BASH_ENV (see
# harbor_evaluator.ACTIVATE_HOOK).
AUTORLBENCH_ACTIVATE_BODY = r"""if [ -n "${AUTORLBENCH_STAGE:-}" ]; then
    if [ -r /etc/autorlbench-image-path ]; then
        _autorlbench_image_path=""
        IFS= read -r _autorlbench_image_path < /etc/autorlbench-image-path || :
        _autorlbench_missing=""
        _autorlbench_ifs=$IFS
        IFS=:
        for _autorlbench_dir in $_autorlbench_image_path; do
            [ -n "$_autorlbench_dir" ] || continue
            case ":$_autorlbench_missing:$PATH:" in
                *":$_autorlbench_dir:"*) ;;
                *) _autorlbench_missing="${_autorlbench_missing:+$_autorlbench_missing:}$_autorlbench_dir" ;;
            esac
        done
        IFS=$_autorlbench_ifs
        if [ -n "$_autorlbench_missing" ]; then
            PATH="$_autorlbench_missing${PATH:+:$PATH}"
            export PATH
        fi
        unset _autorlbench_image_path _autorlbench_missing _autorlbench_ifs _autorlbench_dir
    fi
    AUTORLBENCH_PREPARED="${AUTORLBENCH_PREPARED:-/autorlbench/prepared}"
    export AUTORLBENCH_PREPARED
    if [ -f "$AUTORLBENCH_PREPARED/env" ]; then
        while IFS= read -r _autorlbench_line || [ -n "$_autorlbench_line" ]; do
            case "$_autorlbench_line" in
                ''|'#'*) continue ;;
            esac
            _autorlbench_key="${_autorlbench_line%%=*}"
            case "$_autorlbench_key" in
                ''|*[!A-Za-z0-9_]*)
                    echo "autorlbench: ignoring malformed line in prepared/env: $_autorlbench_line" >&2
                    continue ;;
            esac
            export "$_autorlbench_line"
        done < "$AUTORLBENCH_PREPARED/env"
        unset _autorlbench_line _autorlbench_key
    fi
    if [ "$AUTORLBENCH_STAGE" = "agent" ] || [ "$AUTORLBENCH_STAGE" = "rerun" ]; then
        PYTHONUSERBASE="${PYTHONUSERBASE:-/workspace/.autorlbench-user}"
        PIP_USER=1
        PIP_BREAK_SYSTEM_PACKAGES=1
        export PYTHONUSERBASE PIP_USER PIP_BREAK_SYSTEM_PACKAGES
    fi
fi
unset BASH_ENV
"""


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
    # Single file, outside the run staging, that both stdout and stderr stream into line by line
    # (the harbor path's persistent prepare log). Mutually independent of live_log_name: whoever
    # sets this wants one combined file at an exact path, not <name>.stdout/.stderr.log in results.
    live_log_path: Optional[Path] = None

    def to_argv(self) -> List[str]:
        if self.argv is not None:
            return list(self.argv)
        if self.shell is not None:
            # Non-login shell (`sh -c`): /etc/profile does not run, so PATH keeps the image ENV value.
            # The activate body runs first (exporting prepared/env and friends), then the real command.
            return ["sh", "-c", AUTORLBENCH_ACTIVATE_BODY + "\n" + self.shell]
        raise ValueError("command spec needs argv or shell")


@dataclass(frozen=True)
class EnvironmentArtifact:
    """A prepared environment, reusable across the agent and verifier stages."""

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
    # The task's environment/ directory: prepare.sh lives in it. Mounted read-only in all
    # stages that see it -- see sees_environment_dir() for which stages those are. Read-only
    # because no stage has a reason to modify the task definition in the repo.
    host_environment: Optional[Path] = None
    # prepare-stage outputs land here (models, datasets, env files for the later stages).
    # Writable in prepare, read-only for agent / verifier: grading must not depend on anything the agent can change.
    host_prepared: Optional[Path] = None
    # Download caches reused across runs and tasks, split into two roots by asset kind. Which stages
    # actually get them mounted is decided by sees_download_cache(), not by whoever set the field:
    # the mount table and the environment variables must not drift apart.
    host_models: Optional[Path] = None
    host_datasets: Optional[Path] = None
    workdir: str = "workspace"
    env: Mapping[str, str] = field(default_factory=dict)
    network: str = "bridge"
    timeout_s: Optional[int] = None
    # metadata.environment.env: variables the task declares itself. Per-task variables can only
    # come from here. Device ids no longer travel this way -- the GPU count is carried by `gpus`.
    task_env: Mapping[str, str] = field(default_factory=dict)
    # metadata.environment.gpus: the GPU count the task needs. The docker backend turns it into
    # `--gpus N`. None means a CPU task (no GPU flag at all).
    gpus: Optional[int] = None

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


def sees_environment_dir(request: SandboxRequest) -> bool:
    """Whether this stage should see the task's environment/ directory.

    prepare runs the prepare.sh inside it; the verifier occasionally reads resources from it.
    **The agent does not get it**: prepare.sh describes how the task was assembled -- which HF
    repo to download, which archive to unpack, which sha256 to verify -- and any one of those
    names can point straight at the defect being diagnosed.

    The rerun is launched by the harness, not an extension of the agent: it runs the task's
    declared run_spec, and reading environment/ carries no extra risk there, but it also has no
    need -- conservatively it sees the directory like the verifier does (grading scripts and
    run_spec resources sit in the evaluation directory, not environment/).

    The three stages used to share one mount table, and the agent got the mount too; "do not
    write the answer into prepare.sh" was a convention in the prepare.sh file header, upheld by
    task authors' discipline. A convention cannot cover "the repo name itself is the answer", so
    it became a mechanism instead.
    """
    return request.stage != "agent"


def sees_download_cache(request: SandboxRequest) -> bool:
    """Whether this stage should see $AUTORLBENCH_{MODELS,DATASETS}.

    prepare writes to them (that is where downloads land); the verifier reads them (grading often
    needs a checkpoint or to align dataset samples). **The agent does not get them**: an HF cache's
    directory names are its repo names (datasets--<owner>--<repo>), and a repo name is sometimes
    named after the defect it carries -- the answer written straight into a directory listing.
    The rerun likewise does not get the mount: its command comes from the task declaration and
    needs nothing from the cache, and skipping it removes one more path on which repo names could
    reach the rerun logs that stay in results (the agent cannot touch results, but one unneeded
    mount is still one unneeded mount).

    The agent does not need it either: the weights/datasets prepare delivered have been hard-
    linked into the workspace by materialize(), real bytes, readable with or without the cache.

    The mount table and the environment variables both consult this one predicate so they cannot
    drift -- drift would show up as AUTORLBENCH_MODELS pointing at an unmounted path, reported
    as "the directory is empty" instead of "you cannot see it".
    """
    return request.stage in {"prepare", "verifier"}


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

        Images are shared by environment name: tasks with the same environment use one image, prepared only if missing.
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
            # The agent and rerun stages get a neutral placeholder: task ids often spell out the
            # diagnosis (e.g. "context_parallel_oom"), no better than the oracle as something the
            # agent should read; the rerun is harness-initiated and gets no extra information.
            # The other stages keep the real id -- no task-side script consumes it today, but it
            # is part of the interface prepare.sh may rely on.
            "AUTORLBENCH_TASK_ID": (
                "task" if request.stage in {"agent", "rerun"} else request.task_id
            ),
            "AUTORLBENCH_EVALUATION_LEVEL": request.evaluation_level,
            "AUTORLBENCH_TASK_ROOT": task_root,
            "AUTORLBENCH_VISIBLE_TASK": visible,
            "AUTORLBENCH_WORKSPACE": workspace,
            "AUTORLBENCH_ORACLE": oracle,
            "AUTORLBENCH_VERIFIER": verifier,
            "AUTORLBENCH_RESULTS": results,
            "AUTORLBENCH_STAGE": request.stage,
            "AUTORLBENCH_ENVIRONMENT": (
                self.path(request, "environment")
                if request.host_environment and sees_environment_dir(request)
                else ""
            ),
            "AUTORLBENCH_PREPARED": (
                self.path(request, "prepared") if request.host_prepared else ""
            ),
            # Both download cache roots. An empty string means "not visible this time", consistent
            # with how prepared is handled -- the agent and rerun stages always get empty, see
            # sees_download_cache().
            "AUTORLBENCH_MODELS": (
                self.path(request, "models")
                if request.host_models and sees_download_cache(request)
                else ""
            ),
            "AUTORLBENCH_DATASETS": (
                self.path(request, "datasets")
                if request.host_datasets and sees_download_cache(request)
                else ""
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
        what the prepare stage delivered. The rerun is on the same side: the experiment must re-run on the assets
        prepare delivered, and its output is the grading input -- even less room for a network fetch.
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
