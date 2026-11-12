from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import sys
from pathlib import Path

from .core.harbor_types import HarborEvaluationConfig
from .core.types import (
    TASK_FILE,
    DEFAULT_OUTPUT_DIR,
    EvaluationConfig,
)
from .evaluation.evaluator import TaskEvaluator
from .evaluation.harbor_evaluator import HarborEvaluator, HarborShell
from .tasks import SchemaError


def _is_task_root(path: Path) -> bool:
    """A AutoRLBench task is identified by its metadata file, as the loader does."""
    return (path / TASK_FILE).is_file()


def expand_task_roots(paths: list[str]) -> list[Path]:
    """Expand each argument into task roots, keeping the caller's order.

    Accepts either a task directory or a directory whose children are task
    directories, so a whole family of tasks can be submitted as one argument.
    """
    roots: list[Path] = []
    seen: set[Path] = set()
    for raw in paths:
        path = Path(raw).expanduser()
        if not path.is_dir():
            raise SystemExit(f"task path does not exist or is not a directory: {raw}")
        if _is_task_root(path):
            found = [path]
        else:
            found = sorted(
                child
                for child in path.iterdir()
                if child.is_dir() and _is_task_root(child)
            )
        if not found:
            raise SystemExit(
                f"{raw} is neither a task directory (no task.toml) nor a "
                "directory containing task directories"
            )
        for task_root in found:
            resolved = task_root.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            roots.append(task_root)
    return roots


def build_parser() -> argparse.ArgumentParser:
    # Each evaluator gets its own explicit entry point, so Harbor stays out of runtime selection.
    ap = argparse.ArgumentParser(prog="autorlbench", allow_abbrev=False)
    sub = ap.add_subparsers(dest="command", required=True)

    v = sub.add_parser("validate", help="validate a task package", allow_abbrev=False)
    v.add_argument("task_root")
    v.add_argument("--lenient", action="store_true", help="skip non-critical directory warnings")

    e = sub.add_parser(
        "rerun",
        help=(
            "run a task's prepare.sh then its run_spec.json experiment -- no agent, "
            "no verifier, no report gate; for a task submitter to confirm the buggy "
            "run reproduces in their environment"
        ),
        allow_abbrev=False,
    )
    e.add_argument("task_root")
    e.add_argument(
        "--log-level",
        choices=["quiet", "debug"],
        default="quiet",
        help="terminal logging level; run artifacts are always written",
    )
    e.add_argument("--backend", choices=["docker", "local"], default="docker")
    e.add_argument(
        "--huggingface-mirror",
        default=None,
        help="Hugging Face mirror base URL, passed to the prepare stage as HF_ENDPOINT",
    )
    e.add_argument(
        "--models-dir",
        default=os.environ.get("AUTORLBENCH_MODELS_DIR"),
        help=(
            "host directory shared by every run as the model download cache; "
            "mounted writable into the prepare stage and read-only into the "
            "rerun stage, so a model fetched by one task is not "
            "fetched again by the next (default: <repo>/models) "
            "[env: AUTORLBENCH_MODELS_DIR]"
        ),
    )
    e.add_argument(
        "--datasets-dir",
        default=os.environ.get("AUTORLBENCH_DATASETS_DIR"),
        help=(
            "same as --models-dir, for datasets (default: <repo>/datasets) "
            "[env: AUTORLBENCH_DATASETS_DIR]"
        ),
    )
    e.add_argument("--timeout-s", type=int, default=None)
    e.add_argument(
        "--output-root",
        "--work-dir",
        dest="output_root",
        default=os.environ.get("AUTORLBENCH_RERUN_OUTPUT_DIR"),
        help=(
            "parent directory for persistent per-run logs and command records, laid out as "
            "<output-root>/<task_id>/<timestamp>/ (default: <repo>/outputs/rerun) "
            "[env: AUTORLBENCH_RERUN_OUTPUT_DIR]"
        ),
    )

    h = sub.add_parser(
        "harbor-evaluate",
        help=(
            "run each AutoRLBench task's prepare.sh, then evaluate it with Harbor "
            "on the prebuilt environment image"
        ),
        allow_abbrev=False,
        epilog="All other arguments are forwarded unchanged to `harbor run`.",
    )
    h.add_argument(
        "task_root",
        nargs="+",
        help=(
            "task directory or a directory whose direct children are tasks; "
            "all task paths must precede forwarded Harbor arguments"
        ),
    )
    h.add_argument(
        "--evaluation-level",
        choices=["issue", "solution"],
        default="solution",
        help="task evaluation to run",
    )
    h.add_argument("--judge-model", default=None, help="model used by AutoRLBench verifiers")
    h.add_argument(
        "--huggingface-mirror",
        default=None,
        help="Hugging Face mirror base URL, passed to the prepare stage as HF_ENDPOINT",
    )
    h.add_argument(
        "--models-dir",
        default=os.environ.get("AUTORLBENCH_MODELS_DIR"),
        help=(
            "host model download cache; writable in prepare, mounted read-only "
            "into the Harbor container (default: <repo>/models) "
            "[env: AUTORLBENCH_MODELS_DIR]"
        ),
    )
    h.add_argument(
        "--datasets-dir",
        default=os.environ.get("AUTORLBENCH_DATASETS_DIR"),
        help=(
            "same as --models-dir, for datasets (default: <repo>/datasets) "
            "[env: AUTORLBENCH_DATASETS_DIR]"
        ),
    )
    h.add_argument(
        "--agent-network",
        choices=["no-network", "public", "allowlist"],
        default="public",
        help="network_mode written to the generated Harbor task agent phase",
    )
    h.add_argument(
        "--verifier-network",
        choices=["no-network", "public", "allowlist"],
        default="public",
        help="network_mode written to the generated Harbor task verifier phase",
    )
    h.add_argument(
        "--harbor-command",
        default="harbor",
        help="Harbor executable command, e.g. 'uv run --project harbor harbor'",
    )
    h.add_argument(
        "--harbor-temp-dir",
        type=Path,
        default=None,
        help="parent directory for temporary materialized Harbor tasks",
    )
    h.add_argument(
        "--output-root",
        default=os.environ.get("AUTORLBENCH_HARBOR_OUTPUT_DIR"),
        help=(
            "parent directory for persistent per-run logs and command records, laid out as "
            "<output-root>/<task_id>/<agent>-<model>/<timestamp>/ (default: <repo>/outputs/"
            "harbor-evaluate) [env: AUTORLBENCH_HARBOR_OUTPUT_DIR]"
        ),
    )

    s = sub.add_parser(
        "harbor-shell",
        help=(
            "run a task's prepare.sh, then drop into an interactive shell inside "
            "the prebuilt environment image — same container as harbor-evaluate's "
            "agent phase, but no agent is started"
        ),
        allow_abbrev=False,
    )
    s.add_argument(
        "task_root",
        help="task directory (must contain task.toml)",
    )
    s.add_argument(
        "--evaluation-level",
        choices=["issue", "solution"],
        default="solution",
        help="evaluation level whose environment variables and workspace are used",
    )
    s.add_argument(
        "--huggingface-mirror",
        default=None,
        help="Hugging Face mirror base URL, passed to the prepare stage as HF_ENDPOINT",
    )
    s.add_argument(
        "--models-dir",
        default=os.environ.get("AUTORLBENCH_MODELS_DIR"),
        help=(
            "host model download cache; mounted read-only into the container "
            "(default: <repo>/models) [env: AUTORLBENCH_MODELS_DIR]"
        ),
    )
    s.add_argument(
        "--datasets-dir",
        default=os.environ.get("AUTORLBENCH_DATASETS_DIR"),
        help=(
            "same as --models-dir, for datasets (default: <repo>/datasets) "
            "[env: AUTORLBENCH_DATASETS_DIR]"
        ),
    )
    s.add_argument(
        "--workspace-dir",
        type=Path,
        default=None,
        help=(
            "directory where the staged workspace and state file are stored "
            "across runs; defaults to outputs/harbor-shell/<sha256 of task dir name>"
        ),
    )
    s.add_argument(
        "--force-prepare",
        action="store_true",
        help="re-run prepare.sh even if a staged workspace already exists",
    )
    s.add_argument(
        "--shell",
        default="bash",
        dest="shell_cmd",
        help="shell to start inside the container (default: bash)",
    )
    s.add_argument(
        "--gpus",
        default=None,
        help=(
            "value passed to docker run --gpus; defaults to the task's "
            "environment.gpus when absent"
        ),
    )
    s.add_argument(
        "--shm-size",
        default=None,
        help=(
            "value passed to docker run --shm-size; defaults to the standard "
            "size when the task declares GPUs"
        ),
    )

    return ap


def main(argv=None) -> int:
    parser = build_parser()
    args, harbor_args = parser.parse_known_args(argv)
    if args.command != "harbor-evaluate" and harbor_args:
        parser.error("unrecognized arguments: " + " ".join(harbor_args))

    if args.command == "validate":
        evaluator = TaskEvaluator()
        try:
            report = evaluator.validate(args.task_root, strict=not args.lenient)
        except SchemaError as exc:
            # Structural errors (task.toml fails to parse, invalid environment field) would normally raise,
            # but validate exists to explain problems clearly, so report them in the same shape.
            report = {
                "ok": False,
                "issues": [{"severity": "error", "path": args.task_root, "message": str(exc)}],
            }
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0 if report.get("ok", False) else 1

    if args.command == "rerun":
        evaluator = TaskEvaluator(backend=args.backend)
        result = evaluator.rerun(
            args.task_root,
            EvaluationConfig(
                backend=args.backend,
                timeout_s=args.timeout_s,
                # Logs are always kept -- the command exists to produce them for review,
                # and --output-root only says where. keep_artifacts additionally salvages
                # the workspace copy into the run log directory after the run.
                keep_artifacts=True,
                work_dir=(
                    DEFAULT_OUTPUT_DIR / "rerun"
                    if args.output_root is None
                    else Path(args.output_root)
                ),
                log_level=args.log_level,
                huggingface_mirror=args.huggingface_mirror,
                models_dir=None if args.models_dir is None else Path(args.models_dir),
                datasets_dir=(
                    None if args.datasets_dir is None else Path(args.datasets_dir)
                ),
                invocation_argv=tuple(sys.argv),
            ),
        )
        print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
        return 0 if result.passed else 1

    if args.command == "harbor-evaluate":
        harbor_command = tuple(shlex.split(args.harbor_command))
        report = HarborEvaluator().evaluate(
            expand_task_roots(args.task_root),
            HarborEvaluationConfig(
                evaluation_level=args.evaluation_level,
                judge_model=args.judge_model,
                agent_network=args.agent_network,
                verifier_network=args.verifier_network,
                temp_dir=args.harbor_temp_dir,
                huggingface_mirror=args.huggingface_mirror,
                models_dir=None if args.models_dir is None else Path(args.models_dir),
                datasets_dir=(
                    None if args.datasets_dir is None else Path(args.datasets_dir)
                ),
                harbor_command=harbor_command,
                harbor_args=tuple(harbor_args),
                output_root=(
                    None if args.output_root is None else Path(args.output_root)
                ),
                invocation_argv=tuple(sys.argv),
            ),
        )
        result = (
            report.trials[0]
            if len(report.task_ids) == 1 and len(report.trials) == 1
            else report
        )
        print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
        return 0 if result.passed else 1

    if args.command == "harbor-shell":
        from .evaluation.harbor_evaluator import HarborShell

        task_roots = expand_task_roots([args.task_root])
        if len(task_roots) != 1:
            raise SystemExit("harbor-shell accepts exactly one task directory")
        task_root = task_roots[0]
        # Digest-named on purpose: the workspace_dir is an ancestor of the docker bind-mount
        # sources (the staged workspace/prepared/environment), and /proc/mounts inside the
        # container shows the full source path -- task dir names often spell out the diagnosis.
        # The state file inside records the task id for tracing a directory back to its task.
        workspace_dir = (
            Path(args.workspace_dir)
            if args.workspace_dir is not None
            else Path("outputs")
            / "harbor-shell"
            / hashlib.sha256(task_root.name.encode("utf-8")).hexdigest()[:8]
        )
        HarborShell().shell(
            task_root,
            HarborEvaluationConfig(
                evaluation_level=args.evaluation_level,
                huggingface_mirror=args.huggingface_mirror,
                models_dir=None if args.models_dir is None else Path(args.models_dir),
                datasets_dir=(
                    None if args.datasets_dir is None else Path(args.datasets_dir)
                ),
            ),
            workspace_dir,
            shell_cmd=args.shell_cmd,
            force_prepare=args.force_prepare,
            gpus=args.gpus,
            shm_size=args.shm_size,
        )
        return 0  # unreachable: shell() calls os.execvp

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
