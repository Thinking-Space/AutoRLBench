from __future__ import annotations

import json
import re
import shlex
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Sequence

from ..utils.filesystem import ensure_dir, write_text


class RunLog:
    """Persistent log directory for one run of one task, host-side only.

    Run staging (workspace copies, prepared outputs, materialized Harbor tasks) is deleted
    after the run. What had nowhere to live was everything in between: the stages' output
    and the exact commands the run was launched with. This class owns the directory that
    fills the gap,

        <output_root>/<task_id>/[<label>/]<YYYYmmdd-HHMMSS>/

    On the host only, never a bind-mount source, so the task id in the path is invisible
    to the containers (the reason staging directories carry digests instead, see
    TaskEvaluator._run_directory_prefix).
    """

    def __init__(self, root: Path) -> None:
        self.root = root

    @classmethod
    def create(
        cls,
        task_id: str,
        output_root: Path,
        label: str = "",
    ) -> "RunLog":
        # Labels arrive already cleaned (harbor_run_label); the task id does not, so the
        # same rule applies here.
        safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "-", task_id).strip("-.") or "task"
        timestamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
        root = Path(output_root) / safe_id
        if label:
            root = root / label
        return cls(ensure_dir(root / timestamp))

    # --- files the stages stream into while they run -------------------------

    @property
    def prepare_log_path(self) -> Path:
        return self.root / "prepare.log"

    def stage_log_path(self, stage: str) -> Path:
        """Persistent live log for one stage, e.g. the rerun's rerun.log."""
        return self.root / f"{stage}.log"

    # --- orchestration log ----------------------------------------------------

    def log(self, line: str) -> None:
        """Append one line to run.log and echo it to the terminal."""
        timestamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.root / "run.log", "a", encoding="utf-8") as file:
            file.write(f"[{timestamp}] {line.rstrip()}\n")
        print(f"[autorlbench] {line.rstrip()}", file=sys.stderr, flush=True)

    def error(self, message: str) -> None:
        self.log(f"ERROR {message}")

    def finish(self, passed: bool, score: Any, n_trials: int) -> None:
        self.log(f"[RESULT] passed={str(passed).lower()} score={score} trials={n_trials}")

    # --- command records -------------------------------------------------------

    @classmethod
    def _command_script(
        cls,
        *,
        title: str,
        command_line: str,
        cwd: Path,
        note: str = "",
    ) -> str:
        recorded = datetime.now().astimezone().isoformat(timespec="seconds")
        lines = [
            "#!/usr/bin/env bash",
            f"# {title}, recorded {recorded}",
            f"# run from {cwd}",
        ]
        if note:
            lines.append(f"# {note}")
        lines.append(command_line)
        return "\n".join(lines) + "\n"

    def write_command(
        self,
        argv: Sequence[str],
        cwd: Path,
        title: str = "autorlbench invocation",
    ) -> None:
        """Record the command line verbatim, so the run can be repeated as typed."""
        if not argv:
            return
        path = write_text(
            self.root / "command.sh",
            self._command_script(
                title=title,
                command_line=shlex.join(list(argv)),
                cwd=cwd,
            ),
        )
        path.chmod(0o755)

    # --- run outputs ------------------------------------------------------------

    def capture_report(self, report: Dict[str, Any]) -> None:
        write_text(self.root / "report.json", json.dumps(report, indent=2, ensure_ascii=False))


class HarborRunLog(RunLog):
    """A RunLog under <task_id>/<agent>-<model>/<timestamp>/, plus Harbor's own records.

    The harbor-evaluate flow additionally keeps the harbor CLI's console mirror
    (harbor.log) and points a stable name at Harbor's job directory, where the
    trial-level agent and verifier logs live.
    """

    @classmethod
    def create(
        cls,
        task_id: str,
        agent: str,
        model: str,
        output_root: Path,
    ) -> "HarborRunLog":
        return super().create(task_id, output_root, label=f"{agent}-{model}")

    @property
    def harbor_log_path(self) -> Path:
        return self.root / "harbor.log"

    def write_command(self, argv: Sequence[str], cwd: Path) -> None:
        super().write_command(argv, cwd, title="autorlbench harbor-evaluate invocation")

    def write_harbor_command(self, argv: Sequence[str], cwd: Path) -> None:
        """Record the effective harbor run command (already redacted by HarborRunner)."""
        if not argv:
            return
        path = write_text(
            self.root / "harbor-command.sh",
            self._command_script(
                title="effective harbor command",
                command_line=shlex.join(list(argv)),
                cwd=cwd,
                note=(
                    "--path points into the temporary materialized-tasks staging that "
                    "AutoRLBench deletes after the run; re-running the evaluation means "
                    "re-running command.sh"
                ),
            ),
        )
        path.chmod(0o755)

    def link_job_dir(self, job_dir: Path) -> None:
        """Point a stable name at Harbor's job directory, where the trial-level agent and
        verifier logs live. Harbor never cleans its --jobs-dir, so the link stays valid."""
        link = self.root / "harbor-job"
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(Path(job_dir).resolve())


__all__ = ["HarborRunLog", "RunLog"]
