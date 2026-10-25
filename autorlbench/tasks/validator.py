from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from ..core.types import (
    EvaluationPaths,
    TaskPaths,
    ValidationReport,
    known_images,
)


def validate_task_paths(paths: TaskPaths, strict: bool = True) -> ValidationReport:
    """Check the task's directories and entry files."""
    report = ValidationReport()
    _expect_file(report, paths.task_toml, "task.toml")
    _expect_dir(report, paths.workspace, "workspace")
    # environment/ now holds only optional content (prepare.sh etc.); the image itself comes from
    # metadata environment.image pointing into docker/images/. The directory may be absent, but if present it must be a directory.
    if paths.environment.exists() and not paths.environment.is_dir():
        report.add("error", "environment", "expected a directory")
    if not paths.evaluations:
        report.add("error", "evaluations", "at least one evaluation is required")
    for evaluation in paths.evaluations.values():
        _validate_evaluation_paths(
            report,
            evaluation,
            strict=strict,
            legacy=evaluation.root == paths.root,
        )
    return report


def _validate_evaluation_paths(
    report: ValidationReport,
    evaluation: EvaluationPaths,
    *,
    strict: bool,
    legacy: bool,
) -> None:
    prefix = "" if legacy else f"evaluations/{evaluation.level}/"
    if not legacy:
        _expect_dir(report, evaluation.root, prefix.rstrip("/"))
    _expect_file(report, evaluation.instruction_md, f"{prefix}instruction.md")
    if evaluation.level == "issue":
        _expect_file(report, evaluation.oracle, f"{prefix}oracle.md")
    else:
        _expect_dir(report, evaluation.oracle, f"{prefix}oracle")
    _expect_dir(report, evaluation.verifier, f"{prefix}verifier")
    _expect_file(report, evaluation.verifier_test, f"{prefix}verifier/test.sh")
    if strict:
        _expect_optional_dir(
            report,
            evaluation.verifier_tests,
            f"{prefix}verifier/tests",
        )
        _expect_optional_dir(
            report,
            evaluation.verifier_resources,
            f"{prefix}verifier/resources",
        )


def validate_metadata(
    metadata: Dict[str, Any], paths: Optional[TaskPaths] = None
) -> ValidationReport:
    """Check the fields the current schema requires in metadata.

    `paths` is optional: if given, also check that metadata paths into the task directory exist
    (currently only `training.script`). Without it this still works; the function only needs metadata.
    """
    report = ValidationReport()
    required_fields = [
        "title",
        "submitter",
        "license",
        "created_date",
        "domain",
        "model",
        "dataset",
        "training",
        "environment",
        "outcome",
    ]
    for key in required_fields:
        if key not in metadata or metadata[key] in (None, ""):
            report.add("error", f"metadata.{key}", "missing required field")

    for key in ("submitter", "model", "dataset", "training", "environment", "outcome"):
        _expect_mapping(report, metadata.get(key), f"metadata.{key}")

    submitter = metadata.get("submitter") or {}
    if isinstance(submitter, dict):
        _expect_nonempty(report, submitter.get("name"), "metadata.submitter.name")

    model = metadata.get("model") or {}
    if isinstance(model, dict):
        _expect_nonempty(report, model.get("name"), "metadata.model.name")

    dataset = metadata.get("dataset") or {}
    if isinstance(dataset, dict):
        _expect_nonempty(report, dataset.get("name"), "metadata.dataset.name")

    training = metadata.get("training") or {}
    if isinstance(training, dict):
        for key in ("framework", "algo"):
            _expect_nonempty(report, training.get(key), f"metadata.training.{key}")
        _validate_training(report, training, paths)

    environment = metadata.get("environment") or {}
    if isinstance(environment, dict):
        _validate_environment(report, environment)

    outcome = metadata.get("outcome") or {}
    if isinstance(outcome, dict):
        for key in ("failure_type", "failure_reason"):
            _expect_nonempty(report, outcome.get(key), f"metadata.outcome.{key}")
    return report


def dump_report(report: ValidationReport) -> str:
    """Convert the validation result to JSON the CLI can print directly."""
    return json.dumps(report.to_dict(), indent=2, ensure_ascii=False)


def _expect_file(
    report: ValidationReport, path, label: str, required: bool = True
) -> None:
    if not path.exists():
        report.add("error" if required else "warning", label, "missing file")
    elif not path.is_file():
        report.add("error", label, "expected a file")


def _expect_dir(
    report: ValidationReport, path, label: str, required: bool = True
) -> None:
    if not path.exists():
        report.add("error" if required else "warning", label, "missing directory")
    elif not path.is_dir():
        report.add("error", label, "expected a directory")


def _expect_optional_dir(report: ValidationReport, path, label: str) -> None:
    if path.exists() and not path.is_dir():
        report.add("error", label, "expected a directory")


def _validate_training(
    report: ValidationReport, training: Dict[str, Any], paths: Optional[TaskPaths]
) -> None:
    """Validate the launch script in the training section.

    `script` is the single source of truth for "how this task runs": the task card and
    instruction.md point the agent at the same path. Guessing via glob (*_rl.sh / run_*.sh
    under examples) missed tasks like rlve that keep scripts under scripts/training/,
    so tasks now declare it themselves.

    The path is relative to workspace/, not the task root: in the container workspace is mounted
    at /workspace, the training script runs there, and the task card should record the same path.

    Not required: `framework: none` tasks (e.g. pure code fixes) have no training script.
    But declaring a framework without a script is an omission and gets a warning.
    """
    script = training.get("script")
    args = training.get("script_args")

    if script in (None, ""):
        framework = str(training.get("framework") or "").strip().lower()
        if framework not in ("", "none"):
            report.add(
                "warning",
                "metadata.training.script",
                "missing: declare the launch script (relative to workspace/) so "
                "that the task card records how the task is run",
            )
        if args not in (None, ""):
            report.add(
                "error",
                "metadata.training.script_args",
                "cannot be set without metadata.training.script",
            )
        return

    if not isinstance(script, str):
        report.add(
            "error",
            "metadata.training.script",
            f"expected a path relative to workspace/, got {script!r}",
        )
    elif script.startswith("/") or ".." in Path(script).parts:
        # An absolute path means nothing on another machine, and `..` points outside workspace;
        # in the container only workspace belongs to the task.
        report.add(
            "error",
            "metadata.training.script",
            f"must stay inside workspace/: {script}",
        )
    elif paths is not None and not (paths.workspace / script).is_file():
        # A wrong path and a missing one fail very differently: the former only blows up at `run`,
        # which is often hours later.
        report.add(
            "error",
            "metadata.training.script",
            f"workspace/{script} does not exist",
        )

    if args in (None, ""):
        return
    if not isinstance(args, list):
        # A single string "a b c" and three arguments differ once arguments contain spaces; accept
        # only a list so readers need not guess who does the splitting.
        report.add(
            "error",
            "metadata.training.script_args",
            "expected a list of strings, one argument per item",
        )
        return
    for index, arg in enumerate(args):
        if not isinstance(arg, str):
            # In TOML `0` is an int; `"0"` is what the argument actually looks like.
            report.add(
                "error",
                f"metadata.training.script_args[{index}]",
                "expected a string; quote the value in task.toml",
            )


def _validate_environment(report: ValidationReport, environment: Dict[str, Any]) -> None:
    """Validate the environment section: `image` is required, `prepare` / `env` optional.

    `image` is an environment name under docker/images/, built from docker/images/<image>.Dockerfile.
    Old-schema `conda` / `dockerfile` / `requirements` are errors rather than ignored:
    if ignored, task authors would think they still have an effect.
    """
    for legacy in ("conda", "dockerfile", "requirements"):
        if legacy in environment:
            report.add(
                "error",
                f"metadata.environment.{legacy}",
                "no longer supported; name a docker/images/<name>.Dockerfile "
                "with environment.image instead",
            )

    image = environment.get("image")
    known = known_images()
    if image in (None, ""):
        report.add("error", "metadata.environment.image", "missing required value")
    elif not isinstance(image, str):
        report.add("error", "metadata.environment.image", f"expected a string, got {image!r}")
    elif known and image not in known:
        # No error when docker/images/ is unavailable (e.g. installed as a wheel): we could only
        # guess here; the real check is whether the build finds the Dockerfile.
        report.add(
            "error",
            "metadata.environment.image",
            f"must be one of {', '.join(sorted(known))}, got {image!r}; "
            f"a new environment needs a docker/images/<name>.Dockerfile",
        )

    prepare = environment.get("prepare")
    if prepare not in (None, "") and not isinstance(prepare, str):
        report.add(
            "error",
            "metadata.environment.prepare",
            "expected a path relative to the task root",
        )

    env = environment.get("env")
    if env not in (None, ""):
        if not isinstance(env, dict):
            report.add("error", "metadata.environment.env", "expected a mapping")
        else:
            for key, value in env.items():
                if str(key).startswith("AUTORLBENCH_"):
                    # These are the harness's own channels; letting a task override them lets it rewrite the evaluation's judgment of itself.
                    report.add(
                        "error",
                        f"metadata.environment.env.{key}",
                        "tasks must not set AUTORLBENCH_* variables",
                    )
                elif not isinstance(value, str):
                    # In TOML `CUDA_VISIBLE_DEVICES = 0` is an int; it must be written as `"0"`.
                    report.add(
                        "error",
                        f"metadata.environment.env.{key}",
                        "expected a string; quote the value in task.toml",
                    )


def _expect_mapping(report: ValidationReport, value: Any, label: str) -> None:
    if not isinstance(value, dict):
        report.add("error", label, "expected a mapping")


def _expect_nonempty(report: ValidationReport, value: Any, label: str) -> None:
    if value in (None, "", [], {}):
        report.add("error", label, "missing required value")
