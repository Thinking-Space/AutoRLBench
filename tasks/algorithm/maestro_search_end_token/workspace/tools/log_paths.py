"""Collision-safe log directories shared by launchers and their subprocesses."""
import argparse
from datetime import datetime
import os
from pathlib import Path
import re
import tempfile
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
RUN_LOG_ENV = "MAESTRO_LOG_DIR"


def create_run_log_dir(label="run", *, root=None):
    """Atomically reserve logs/YYYY-MM-DD/<label>-<time>-<unique>/.

    A run stays in its start-date directory, even if it crosses midnight.
    """
    if not re.fullmatch(r"[A-Za-z0-9_-]+", label):
        raise ValueError("Log run label must contain only letters, digits, '_' or '-'")
    project = Path(root).resolve() if root is not None else ROOT
    now = datetime.now(ZoneInfo(os.environ.get("MAESTRO_LOG_TIMEZONE", "Asia/Shanghai")))
    log_root = Path(os.environ.get("MAESTRO_LOG_ROOT", project / "logs"))
    day = log_root / now.strftime("%Y-%m-%d")
    day.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=f"{label}-{now:%H%M%S-%f}-{os.getpid()}-", dir=day))


def get_run_log_dir(label="run", *, root=None):
    """Reuse a parent launcher's directory; reject paths outside dated logs."""
    project = Path(root).resolve() if root is not None else ROOT
    results = os.environ.get("LOG_DIR")
    if results:
        directory = Path(results).resolve()
        directory.mkdir(parents=True, exist_ok=True)
        os.environ[RUN_LOG_ENV] = str(directory)
        return directory
    inherited = os.environ.get(RUN_LOG_ENV)
    if inherited:
        directory = Path(inherited).resolve()
        try:
            log_root = Path(os.environ.get("MAESTRO_LOG_ROOT", project / "logs")).resolve()
            relative = directory.relative_to(log_root)
        except ValueError as error:
            raise ValueError(f"{RUN_LOG_ENV} must be inside {log_root}") from error
        if len(relative.parts) != 2 or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", relative.parts[0]):
            raise ValueError(f"{RUN_LOG_ENV} must be logs/YYYY-MM-DD/<run>")
        datetime.strptime(relative.parts[0], "%Y-%m-%d")
        if not directory.is_dir():
            raise ValueError(f"Inherited log directory does not exist: {directory}")
        return directory
    directory = create_run_log_dir(label, root=project)
    os.environ[RUN_LOG_ENV] = str(directory)
    return directory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("label", nargs="?", default="run")
    args = parser.parse_args()
    print(get_run_log_dir(args.label))


if __name__ == "__main__":
    main()
