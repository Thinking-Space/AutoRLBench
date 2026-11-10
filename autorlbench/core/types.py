from __future__ import annotations

import re
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
class CommandResult:
    """Unified result of one command execution, shared by every stage."""

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
# Must match the defaults the activate body in runtimes/base.py reads ($AUTORLBENCH_PREPARED/env) and the
# task-local fetch tools that travel with the task's environment/ directory.
PREPARED_DIR = "/autorlbench/prepared"
# Download caches, split into two roots by asset kind: on the host, models/ and datasets/ at the repo root.
# The split is not cosmetic: an explicit directory next to the repo is obviously missing when unmounted,
# while a single cache hidden under work_dir would silently leave gigabytes in the container's writable layer.
# Writable in prepare, read-only for agent / verifier.
MODELS_DIR = "/autorlbench/models"
DATASETS_DIR = "/autorlbench/datasets"
# Where the task's environment/ directory is mounted. Read-only: prepare.sh describes the environment,
# not the answer, but no stage has a reason to modify the task definition in the repo. The prepare-stage
# fetch/dataset tools (fetch_hf.py and friends) also travel with the task in this directory, and prepare.sh
# calls them with a plain `python3`.
AUTORLBENCH_ENVIRONMENT_DIR = "/autorlbench/environment"

# Conventional file names in a task package. They are constants because both loader and evaluator use them,
# and "defaults scattered in several places" is exactly where schema drift starts.
# Task card file name at the task root; only directories containing it count as tasks.
TASK_FILE = "task.toml"
DEFAULT_PREPARE = "environment/prepare.sh"

# The single source of truth for available environments is <name>.Dockerfile under docker/images/; the image
# it points at is pulled, not built. No copy of the list lives here: a copy drifts, and drift means a task
# names an environment with no Dockerfile, which goes unnoticed until evaluation runs.
_REPO_ROOT = Path(__file__).resolve().parents[2]
DOCKER_DIR = _REPO_ROOT / "docker"
IMAGES_DIR = DOCKER_DIR / "images"
IMAGE_DOCKERFILE_SUFFIX = ".Dockerfile"
# Legacy header for Dockerfiles that declare their tag explicitly, e.g. `# autorlbench-image:
# autorlbench/verl:vllm0.17-torch2.10-cu129`. Current Dockerfiles are a single digest-pinned FROM line,
# and the reference on that line is the image the task runs; the header is only a fallback.
IMAGE_TAG_HEADER = "# autorlbench-image:"

# A pullable image reference: repository (lowercase, `/`-separated), optional tag, optional
# sha256 digest. Follows Docker's reference grammar closely enough for every reference the
# benchmark carries; a stricter parse would reject nothing we actually use.
IMAGE_REFERENCE_RE = re.compile(
    r"^[a-z0-9]+(?:(?:\.|_|__|-+)[a-z0-9]+)*"  # repository name component
    r"(?:/[a-z0-9]+(?:(?:\.|_|__|-+)[a-z0-9]+)*)*"  # ... and the registry/path components
    r"(?::[A-Za-z0-9_][A-Za-z0-9._-]{0,127})?"  # tag
    r"(?:@sha256:[0-9a-f]{64})?$"  # digest
)

# Default host location for download caches. They sit next to the repo rather than inside the run directory
# because run directories are deleted afterwards, and the whole point of these is to outlive a run so the
# next run and next task hit them. Defining the default here, not per entry point, keeps cli and harbor in sync.
DEFAULT_MODELS_DIR = _REPO_ROOT / "models"
DEFAULT_DATASETS_DIR = _REPO_ROOT / "datasets"
# Default parent for persistent run logs. They sit next to the download caches for the same reason:
# run staging is deleted afterwards, and logs exist to be reviewed after the fact.
DEFAULT_OUTPUT_DIR = _REPO_ROOT / "outputs"


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
    """The image a task runs, as the reference the backends pull.

    `reference` is the fully specified pullable reference — repository, optional tag, optional
    `@sha256:` digest — taken from the task's `environment.image`; every backend (docker
    evaluation, build-images, Harbor's `docker_image`) pulls exactly this string.
    """

    reference: str

    def __post_init__(self) -> None:
        if not IMAGE_REFERENCE_RE.match(self.reference):
            raise ValueError(
                "not a valid image reference (expected repo[:tag][@sha256:<digest>]): "
                f"{self.reference!r}"
            )

    @property
    def tag(self) -> str:
        """The reference string itself; kept under the backends' `tag` vocabulary."""
        return self.reference


def resolve_environment_reference(name: str) -> str:
    """The pullable reference an environment under docker/images/ declares.

    Resolution order: an optional `# autorlbench-image: <tag>` header (build-recipe Dockerfiles
    use it to declare the tag the recipe is to be pushed as), otherwise the image reference from
    the FROM line. Only when neither exists is it an error — defaulting to a tag would hide the
    broken Dockerfile until the first run.
    """
    dockerfile = IMAGES_DIR / f"{name}{IMAGE_DOCKERFILE_SUFFIX}"
    try:
        text = dockerfile.read_text(encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(
            f"cannot read {dockerfile} to resolve the image of {name!r}"
        ) from exc
    from_ref: Optional[str] = None
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith(IMAGE_TAG_HEADER):
            tag = stripped[len(IMAGE_TAG_HEADER):].strip()
            if tag:
                return tag
        if from_ref is None and stripped[:5].upper() == "FROM ":
            # FROM <ref> [AS <stage>]
            parts = stripped.split()
            if len(parts) >= 2:
                from_ref = parts[1]
    if from_ref:
        return from_ref
    raise RuntimeError(
        f"{dockerfile} has neither a '{IMAGE_TAG_HEADER} <tag>' header "
        "nor a FROM line to take the image reference from"
    )


@dataclass(frozen=True)
class EnvironmentPlan:
    """Parsed `environment` section of a task.

    `image` is the pullable image reference — repository, optional tag, `@sha256:` digest — that the
    backends pull directly. Many tasks share one reference; the packaged image is the ground truth
    environment, so the reference is pinned by digest wherever the image has been built and pushed.
    `image_stack` is a free-form human-readable description of the stack inside that image (the
    slot-grammar name, e.g. `verl-vllm0.17.0-torch2.10.0-...-py3.12`); it is descriptive only and
    points at nothing. `prepare` sets up what training needs (download models and datasets, write
    env vars); defaults to environment/prepare.sh. `env` holds variables the task exports itself.

    `gpus` is the GPU count the task needs: docker exposes it as `--gpus <count>` (cuda:0..N-1), and
    the harbor backend synthesizes a compose device reservation from it. It replaces both the old
    per-task environment/docker-compose.yaml and the CUDA_VISIBLE_DEVICES / NVIDIA_VISIBLE_DEVICES
    env entries, which named GPUs two more ways that could drift from what the training script pins.
    """

    image: Optional[str] = None
    image_stack: Optional[str] = None
    prepare: Optional[Path] = None
    env: Dict[str, str] = field(default_factory=dict)
    gpus: Optional[int] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "env", dict(self.env))
        if self.gpus is not None and (isinstance(self.gpus, bool) or not isinstance(self.gpus, int) or self.gpus < 0):
            raise ValueError(
                f"environment.gpus must be a non-negative integer GPU count, got {self.gpus!r}"
            )
        if self.image is not None and not IMAGE_REFERENCE_RE.match(self.image):
            raise ValueError(
                "environment.image must be a pullable image reference "
                "repo[:tag][@sha256:<digest>], "
                f"got {self.image!r}"
            )
        if self.image_stack is not None and not isinstance(self.image_stack, str):
            raise ValueError(
                f"environment.image_stack must be a string, got {self.image_stack!r}"
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

    backend: str = "docker"
    timeout_s: Optional[int] = None
    keep_artifacts: bool = False
    work_dir: Optional[Path] = None
    evaluation_level: str = "solution"
    log_level: str = "quiet"
    huggingface_mirror: Optional[str] = None
    # Shared host download caches, mounted at /autorlbench/models and /autorlbench/datasets in the container.
    # Models and datasets fetched by prepare.sh land here, so a second task needing the same model does not
    # download it again, nor does an image rebuild. Empty means models/ and datasets/ at the repo root.
    models_dir: Optional[Path] = None
    datasets_dir: Optional[Path] = None
    # Command line to record verbatim into the run's command.sh; the CLI passes sys.argv.
    invocation_argv: Sequence[str] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.huggingface_mirror is not None:
            object.__setattr__(
                self,
                "huggingface_mirror",
                _normalize_plain_url(self.huggingface_mirror),
            )
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
            "rerun": None if self.rerun is None else self.rerun.to_dict(),
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
