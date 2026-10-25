from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence
from urllib.parse import urlsplit


@dataclass(frozen=True)
class EvaluationPaths:
    """A task evaluation's public instruction and hidden grading paths."""

    level: str
    root: Path
    instruction_md: Path
    oracle: Path
    verifier: Path
    verifier_test: Path
    verifier_tests: Optional[Path]
    verifier_resources: Optional[Path]


@dataclass(frozen=True)
class TaskPaths:
    """Filesystem layout of a parsed task package."""

    root: Path
    readme_md: Optional[Path]
    task_toml: Path
    workspace: Path
    environment: Path
    evaluations: Dict[str, EvaluationPaths]


@dataclass(frozen=True)
class ValidationIssue:
    """One directory or schema issue reported by the validator."""

    severity: str
    path: str
    message: str


@dataclass
class ValidationReport:
    """Collects all validation issues so contributors can fix them in one pass."""

    issues: List[ValidationIssue] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not any(issue.severity == "error" for issue in self.issues)

    def add(self, severity: str, path: str, message: str) -> None:
        self.issues.append(ValidationIssue(severity=severity, path=path, message=message))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "issues": [
                {"severity": item.severity, "path": item.path, "message": item.message}
                for item in self.issues
            ],
        }


@dataclass(frozen=True)
class TaskPackage:
    """A loaded task: paths, metadata, and validation report."""

    paths: TaskPaths
    metadata: Dict[str, Any]
    report: ValidationReport
    environment: "EnvironmentPlan" = field(default_factory=lambda: EnvironmentPlan())

    @property
    def task_id(self) -> str:
        return self.paths.root.name

    @property
    def evaluation_levels(self) -> Sequence[str]:
        return tuple(self.paths.evaluations)


@dataclass(frozen=True)
class AgentSpec:
    """Normalized agent configuration."""

    kind: str = "codex"
    command: Optional[str] = None
    model: Optional[str] = None
    judge_model: Optional[str] = None
    reasoning_effort: Optional[str] = None
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    extra_args: Sequence[str] = field(default_factory=tuple)
    network: str = "bridge"
    env: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "extra_args", tuple(self.extra_args))
        object.__setattr__(self, "env", _normalize_env(self.env))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "command": self.command,
            "model": self.model,
            "judge_model": self.judge_model,
            "reasoning_effort": self.reasoning_effort,
            "base_url": self.base_url,
            "api_key": None if self.api_key is None else "<redacted>",
            "extra_args": list(self.extra_args),
            "network": self.network,
            "env": _redact_env(dict(self.env)),
        }


@dataclass(frozen=True)
class CommandResult:
    """Unified result of one command execution, shared by agent and verifier."""

    argv: List[str]
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool
    duration_s: float
    cwd: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "argv": self.argv,
            "returncode": self.returncode,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "timed_out": self.timed_out,
            "duration_s": self.duration_s,
            "cwd": self.cwd,
        }


@dataclass(frozen=True)
class StageArtifact:
    """Artifact record structure reserved for later evaluation stages."""

    name: str
    path: Path
    command: Optional[CommandResult] = None
    extra: Dict[str, Any] = field(default_factory=dict)


# Directory shared by all three stages in the container. Writable in prepare, read-only for agent / verifier.
# Must match the defaults in docker/autorlbench-activate.sh and docker/autorlbench_fetch.py.
PREPARED_DIR = "/autorlbench/prepared"
# Download caches, split into two roots by asset kind: on the host, models/ and datasets/ at the repo root.
# The split is not cosmetic: an explicit directory next to the repo is obviously missing when unmounted,
# while a single cache hidden under work_dir would silently leave gigabytes in the container's writable layer.
# Writable in prepare, read-only for agent / verifier.
MODELS_DIR = "/autorlbench/models"
DATASETS_DIR = "/autorlbench/datasets"
# Where the task's environment/ directory is mounted. Read-only in all three stages: prepare.sh describes
# the environment, not the answer, but no stage has a reason to modify the task definition in the repo.
AUTORLBENCH_ENVIRONMENT_DIR = "/autorlbench/environment"
# Where the repo's docker/ directory (see DOCKER_DIR) is mounted in the prepare container. Read-only, prepare only.
# Before prepare.sh, prepare runs `bash $TOOLING_DIR/install-autorlbench-tooling.sh` to install the fetch/
# dataset tools from the current repo into /usr/local, avoiding stale copies baked into the image.
TOOLING_DIR = "/autorlbench/tooling"
# Container path of that install script from the repo's docker/ once mounted.
TOOLING_INSTALLER = f"{TOOLING_DIR}/install-autorlbench-tooling.sh"

# Conventional file names in a task package. They are constants because both loader and evaluator use them,
# and "defaults scattered in several places" is exactly where schema drift starts.
# Task card file name at the task root; only directories containing it count as tasks.
TASK_FILE = "task.toml"
DEFAULT_PREPARE = "environment/prepare.sh"

# The single source of truth for available environments is <name>.Dockerfile under docker/images/; the image
# is built from it. No copy of the list lives here: a copy drifts, and drift means a task names an
# environment with no Dockerfile, which goes unnoticed until evaluation runs.
_REPO_ROOT = Path(__file__).resolve().parents[2]
DOCKER_DIR = _REPO_ROOT / "docker"
IMAGES_DIR = DOCKER_DIR / "images"
IMAGE_DOCKERFILE_SUFFIX = ".Dockerfile"
# The image tag is declared on a `# autorlbench-image: <tag>` line in the Dockerfile header, e.g.
# autorlbench/verl:vllm0.17-torch2.10-cu129. The tag describes the stack (framework, inference engine, CUDA
# version) and is shared by many tasks; it is not a content hash. Upgrading the stack changes the tag; changing
# only tooling scripts keeps the tag and requires an explicit `autorlbench build-images --rebuild`.
IMAGE_TAG_HEADER = "# autorlbench-image:"

# Default host location for download caches. They sit next to the repo rather than inside the run directory
# because run directories are deleted afterwards, and the whole point of these is to outlive a run so the
# next run and next task hit them. Defining the default here, not per entry point, keeps cli and harbor in sync.
DEFAULT_MODELS_DIR = _REPO_ROOT / "models"
DEFAULT_DATASETS_DIR = _REPO_ROOT / "datasets"


def known_images() -> frozenset:
    """Environment names that have a Dockerfile under docker/images/; empty set if the directory is unavailable.

    An empty set means "cannot tell", not "none exist". Once installed as a wheel, `docker/` may not come
    along, so validating environment names has to wait until the image build.
    """
    if not IMAGES_DIR.is_dir():
        return frozenset()
    return frozenset(
        item.name[: -len(IMAGE_DOCKERFILE_SUFFIX)]
        for item in IMAGES_DIR.iterdir()
        if item.is_file()
        and item.name.endswith(IMAGE_DOCKERFILE_SUFFIX)
        and not item.name.startswith(".")
    )


@dataclass(frozen=True)
class EnvironmentImage:
    """The image for an environment: docker/images/<name>.Dockerfile, tag declared in its header.

    The build context is the whole docker/ directory, since every image COPYs the same tooling scripts
    (autorlbench-activate.sh, autorlbench_fetch.py, etc.); see docker/.dockerignore.
    Docker evaluation, build-images, and Harbor's `docker_image` all use the same tag.
    """

    name: str

    @property
    def tag(self) -> str:
        """The tag declared by `# autorlbench-image: <tag>` in the Dockerfile header.

        Raise if missing instead of falling back to a default: a fallback would let two environments silently
        share one tag, with the later build overwriting the earlier one.
        """
        try:
            text = self.dockerfile.read_text(encoding="utf-8")
        except OSError as exc:
            raise RuntimeError(
                f"cannot read {self.dockerfile} to resolve the image tag of {self.name!r}"
            ) from exc
        for line in text.splitlines():
            if line.startswith(IMAGE_TAG_HEADER):
                tag = line[len(IMAGE_TAG_HEADER):].strip()
                if tag:
                    return tag
        raise RuntimeError(
            f"{self.dockerfile} has no '{IMAGE_TAG_HEADER} <tag>' header line"
        )

    @property
    def dockerfile(self) -> Path:
        return IMAGES_DIR / f"{self.name}{IMAGE_DOCKERFILE_SUFFIX}"

    @property
    def context(self) -> Path:
        return DOCKER_DIR


@dataclass(frozen=True)
class EnvironmentPlan:
    """Parsed `environment` section of a task.

    `image` is the environment name, mapping to docker/images/<image>.Dockerfile; many tasks share one image.
    `prepare` sets up what training needs (download models and datasets, write env vars); defaults to
    environment/prepare.sh. `env` holds variables the task exports itself.
    """

    image: Optional[str] = None
    prepare: Optional[Path] = None
    env: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "env", dict(self.env))
        known = known_images()
        if self.image is not None and known and self.image not in known:
            raise ValueError(
                f"environment.image must be one of {', '.join(sorted(known))}, "
                f"got {self.image!r}; add docker/images/{self.image}"
                f"{IMAGE_DOCKERFILE_SUFFIX} if the task really needs a new one"
            )
        for key, value in self.env.items():
            if not key or not isinstance(key, str):
                raise ValueError("environment.env keys must be non-empty strings")
            if not isinstance(value, str):
                # In YAML `CUDA_VISIBLE_DEVICES: 0,1` reads as a string, but `: 0` is an int.
                # Silently calling str() would hide the difference between 0 and "0" for good.
                raise ValueError(
                    f"environment.env[{key}] must be a string; quote it in YAML"
                )
        # AUTORLBENCH_* are the harness's own channels (stage / task_id / mount paths, etc.).
        # Letting a task override them lets it rewrite the framework's judgment of itself.
        reserved = sorted(key for key in self.env if key.startswith("AUTORLBENCH_"))
        if reserved:
            raise ValueError(
                "environment.env must not set AUTORLBENCH_* variables: "
                + ", ".join(reserved)
            )

    @property
    def environment_image(self) -> Optional[EnvironmentImage]:
        return None if self.image is None else EnvironmentImage(self.image)


@dataclass(frozen=True)
class EvaluationConfig:
    """Caller runtime configuration, kept separate from the task package's own data."""

    agent: Optional[AgentSpec] = None
    verifier_cmd: Optional[str] = None
    backend: str = "docker"
    verifier_network: str = "bridge"
    timeout_s: Optional[int] = None
    keep_artifacts: bool = False
    work_dir: Optional[Path] = None
    evaluation_level: str = "solution"
    log_level: str = "quiet"
    debian_mirror: Optional[str] = None
    huggingface_mirror: Optional[str] = None
    verifier_env: Mapping[str, str] = field(default_factory=dict)
    # Shared host download caches, mounted at /autorlbench/models and /autorlbench/datasets in the container.
    # Models and datasets fetched by prepare.sh land here, so a second task needing the same model does not
    # download it again, nor does an image rebuild. Empty means models/ and datasets/ at the repo root.
    models_dir: Optional[Path] = None
    datasets_dir: Optional[Path] = None

    def __post_init__(self) -> None:
        if self.debian_mirror is not None:
            object.__setattr__(self, "debian_mirror", _normalize_plain_url(self.debian_mirror))
        if self.huggingface_mirror is not None:
            object.__setattr__(
                self,
                "huggingface_mirror",
                _normalize_plain_url(self.huggingface_mirror),
            )
        object.__setattr__(self, "verifier_env", _normalize_env(self.verifier_env))
        # Both cache roots always have a value: if the caller gives none, they fall back to the repo root. Downstream
        # code never handles "not given", and the mount table has no "no cache this time" branch.
        for name, fallback in (
            ("models_dir", DEFAULT_MODELS_DIR),
            ("datasets_dir", DEFAULT_DATASETS_DIR),
        ):
            given = getattr(self, name)
            resolved = (Path(given) if given is not None else fallback).expanduser().resolve()
            object.__setattr__(self, name, resolved)


@dataclass(frozen=True)
class EvaluationResult:
    """Final evaluation summary produced by AutoRLBench."""

    task_id: str
    evaluation_level: str
    passed: bool
    score: Optional[float]
    metrics: Dict[str, Any]
    report_dir: Path
    agent_spec: Optional[AgentSpec]
    agent_setup: Optional[CommandResult]
    agent: Optional[CommandResult]
    verifier: Optional[CommandResult]
    workspace_before_hash: str
    workspace_after_hash: str
    metadata: Dict[str, Any]
    # The solution level's system rerun: the pass the harness runs after the agent submits its
    # report and before grading. None for the issue level, and for solution tasks with no
    # run_spec.json. Defaulted so the field can sit last without reordering the rest.
    rerun: Optional[CommandResult] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "evaluation_level": self.evaluation_level,
            "passed": self.passed,
            "score": self.score,
            "metrics": self.metrics,
            "report_dir": str(self.report_dir),
            "agent_spec": None if self.agent_spec is None else self.agent_spec.to_dict(),
            "agent_setup": None if self.agent_setup is None else self.agent_setup.to_dict(),
            "agent": None if self.agent is None else self.agent.to_dict(),
            "rerun": None if self.rerun is None else self.rerun.to_dict(),
            "verifier": None if self.verifier is None else self.verifier.to_dict(),
            "workspace_before_hash": self.workspace_before_hash,
            "workspace_after_hash": self.workspace_after_hash,
            "metadata": self.metadata,
        }


def _normalize_plain_url(value: str) -> str:
    url = str(value).rstrip("/")
    parsed = urlsplit(url)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.netloc
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or "|" in url
        or any(char.isspace() for char in url)
    ):
        raise ValueError("URL must be a plain HTTP(S) base URL")
    return url


def _normalize_env(env: Mapping[str, str]) -> Dict[str, str]:
    return {str(key): str(value) for key, value in env.items()}


def _redact_env(env: Dict[str, str]) -> Dict[str, str]:
    redacted: Dict[str, str] = {}
    for key, value in env.items():
        if any(token in key.upper() for token in ("KEY", "TOKEN", "SECRET", "PASSWORD")):
            redacted[key] = "<redacted>"
        else:
            redacted[key] = value
    return redacted
