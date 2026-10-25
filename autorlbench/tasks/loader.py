from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any, Dict, Optional

from ..core.types import (
    DEFAULT_PREPARE,
    TASK_FILE,
    EnvironmentPlan,
    EvaluationPaths,
    TaskPackage,
    TaskPaths,
    ValidationReport,
)
from ..utils.filesystem import pick_existing
from .validator import validate_task_paths, validate_metadata


class SchemaError(ValueError):
    """The task package cannot be loaded or lacks the basic structure."""


def load_metadata(path: Path) -> Dict[str, Any]:
    """Read task.toml. A TOML top level is always a table, so no type check is needed."""
    if not path.exists():
        raise SchemaError(f"missing task file: {path}")
    try:
        with path.open("rb") as file:
            return tomllib.load(file)
    except tomllib.TOMLDecodeError as exc:
        raise SchemaError(f"failed to parse {path}: {exc}") from exc


def load_task_paths(root: Path) -> TaskPaths:
    """Resolve the task root into a fixed set of paths."""
    root = root.resolve()
    readme_md = pick_existing(root, ["README.md"])
    task_toml = root / TASK_FILE
    if not task_toml.is_file():
        raise SchemaError(f"missing {TASK_FILE} under {root}")

    evaluations = _load_evaluation_paths(root)
    return TaskPaths(
        root=root,
        readme_md=readme_md,
        task_toml=task_toml,
        workspace=root / "workspace",
        environment=root / "environment",
        evaluations=evaluations,
    )


def _load_evaluation_paths(root: Path) -> Dict[str, EvaluationPaths]:
    """Load evaluations/, or map the legacy top-level layout to solution."""
    evaluations_root = root / "evaluations"
    evaluations: Dict[str, EvaluationPaths] = {}
    if evaluations_root.is_dir():
        for level in ("issue", "solution"):
            evaluation_root = evaluations_root / level
            if not evaluation_root.exists():
                continue
            oracle = evaluation_root / ("oracle.md" if level == "issue" else "oracle")
            evaluations[level] = _evaluation_paths(level, evaluation_root, oracle)
        if not evaluations:
            raise SchemaError(
                f"{evaluations_root} must contain an issue/ or solution/ evaluation"
            )
        return evaluations

    instruction_md = root / "instruction.md"
    if not instruction_md.exists():
        raise SchemaError(
            f"missing evaluations/ and legacy instruction.md under {root}"
        )
    evaluations["solution"] = _evaluation_paths(
        "solution",
        root,
        root / "oracle",
    )
    return evaluations


def _evaluation_paths(level: str, root: Path, oracle: Path) -> EvaluationPaths:
    verifier = root / "verifier"
    return EvaluationPaths(
        level=level,
        root=root,
        instruction_md=root / "instruction.md",
        oracle=oracle,
        verifier=verifier,
        verifier_test=verifier / "test.sh",
        verifier_tests=verifier / "tests",
        verifier_resources=verifier / "resources",
    )


def load_environment_plan(paths: TaskPaths, metadata: Dict[str, Any]) -> EnvironmentPlan:
    """Parse the environment section of metadata into an EnvironmentPlan.

    Only `image` is required: the environment name under docker/images/. `prepare` has a
    conventional path and is found even if the task omits it. Nearly every task has a
    prepare.sh, and forcing each task.toml to repeat the path only invites typos. An explicit value wins.
    """
    environment = metadata.get("environment")
    if not isinstance(environment, dict):
        environment = {}

    image = _optional_str(environment.get("image"))

    # The conventional path lands on the plan only if the file actually exists. A plan carrying a
    # missing path would conflate "the task has no prepare.sh" with "the task declared a wrong path",
    # and the latter must be an error. The validator tells the two apart.
    prepare_path = _resolve_optional_file(
        paths, environment.get("prepare"), DEFAULT_PREPARE
    )

    env_declared = environment.get("env")
    if env_declared is None:
        env = {}
    elif isinstance(env_declared, dict):
        env = env_declared
    else:
        raise SchemaError(
            f"{paths.task_toml}: metadata.environment.env must be a mapping"
        )

    try:
        return EnvironmentPlan(
            image=image,
            prepare=prepare_path,
            env=env,
        )
    except ValueError as exc:
        raise SchemaError(f"{paths.task_toml}: {exc}") from exc


def _resolve_optional_file(
    paths: TaskPaths, declared: Any, default: str
) -> Optional[Path]:
    """Return an explicit path as-is; otherwise return the conventional path only if it exists."""
    text = _optional_str(declared)
    if text is not None:
        return paths.root / text
    candidate = paths.root / default
    return candidate if candidate.is_file() else None


def _optional_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def load_task_package(root: str | Path, strict: bool = True) -> TaskPackage:
    """Load a task and return its path and metadata validation results."""
    paths = load_task_paths(Path(root))
    path_report = validate_task_paths(paths, strict=strict)
    metadata = load_metadata(paths.task_toml)
    metadata_report = validate_metadata(metadata, paths)
    report = ValidationReport()
    report.issues.extend(path_report.issues)
    report.issues.extend(metadata_report.issues)
    environment = load_environment_plan(paths, metadata)
    return TaskPackage(
        paths=paths,
        metadata=metadata,
        report=report,
        environment=environment,
    )
