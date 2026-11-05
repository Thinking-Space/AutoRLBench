from __future__ import annotations

import json
import re
import shlex
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from ..utils.filesystem import ensure_dir, write_text


class HarborRunLog:
    """Persistent log directory for one harbor-evaluate run of one task.

    Run staging (workspace copies, prepared outputs, materialized Harbor tasks) is deleted after
    the run, and Harbor keeps its own job directory under --jobs-dir. What had nowhere to live
    was everything in between: the prepare stage's output and the exact commands the run was
    launched with. This class owns the directory that fills the gap,

        <output_root>/<task_id>/<agent>-<model>/<YYYYmmdd-HHMMSS>/

    On the host only, never a bind-mount source, so the task id in the path is invisible to the
    containers (the reason staging directories carry digests instead, see
    HarborEvaluator._make_run_dir).
    """

    def __init__(self, root: Path) -> None:
        self.root = root

    @classmethod
    def create(
        cls,
        task_id: str,
        agent: str,
        model: str,
        output_root: Path,
    ) -> "HarborRunLog":
        # agent and model arrive already cleaned (harbor_run_label); the task id does not, so the
        # same rule applies here.
        safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "-", task_id).strip("-.") or "task"
        timestamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
        root = ensure_dir(Path(output_root) / safe_id / f"{agent}-{model}" / timestamp)
        return cls(root)

    # --- files the stages stream into while they run -------------------------

    @property
    def prepare_log_path(self) -> Path:
        return self.root / "prepare.log"

    @property
    def harbor_log_path(self) -> Path:
        return self.root / "harbor.log"

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

    def write_command(self, argv: Sequence[str], cwd: Path) -> None:
        """Record the autorlbench command line verbatim, so the run can be repeated as typed."""
        if not argv:
            return
        path = write_text(
            self.root / "command.sh",
            self._command_script(
                title="autorlbench harbor-evaluate invocation",
                command_line=shlex.join(list(argv)),
                cwd=cwd,
            ),
        )
        path.chmod(0o755)

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

    # --- run outputs ------------------------------------------------------------

    def capture_report(self, report: Dict[str, Any]) -> None:
        write_text(self.root / "report.json", json.dumps(report, indent=2, ensure_ascii=False))

    def link_job_dir(self, job_dir: Path) -> None:
        """Point a stable name at Harbor's job directory, where the trial-level agent and
        verifier logs live. Harbor never cleans its --jobs-dir, so the link stays valid."""
        link = self.root / "harbor-job"
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(Path(job_dir).resolve())


__all__ = ["HarborRunLog"]
