"""The system-rerun protocol of the solution level.

Solution protocol: the agent only repairs the workspace and submits
`report.md`; it **does not launch training**. The harness re-runs the
experiment itself on the workspace the agent left behind, and the task's own
grading script then reads the key metrics out of that fresh run's logs.

This module is the machine-readable half of that protocol. A launch script's
defaults and the budget the protocol requires are often different (the
search-r1 script defaults to `--num-rollout 200` where the protocol wants 100;
retool defaults to 3000 where it wants 70), and a budget override that exists
only in the prose of instruction.md gives the harness no way to run it
correctly on the agent's behalf. So every training task pins one down in
`evaluations/solution/verifier/resources/run_spec.json`: which command to run,
with which environment, how long it may take, and where the logs land.

A solution task without a run_spec.json is still bound by the agent-side half
of the protocol -- repair, submit report.md, do not train -- the harness simply
does not run an extra experiment for it. That is the case for tasks whose
verifier drives the work itself, and for pure protocol repairs such as
verl_aime25_single_sample_protocol.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

RUN_SPEC_FILENAME = "run_spec.json"
RUN_SPEC_SCHEMA_VERSION = 1
# The harness variables an env value may reference. Writing $AUTORLBENCH_RESULTS keeps a spec
# portable across backends: in the docker container it is /results, in Harbor's verifier container
# /logs/artifacts, and under the local backend a directory on the host. Expansion happens on the
# executing side, so nothing is hard-coded in the spec.
_EXPANDABLE_PREFIX = "$AUTORLBENCH_"
REPORT_FILENAME = "report.md"


class RunSpecError(ValueError):
    """run_spec.json exists but is invalid: a task-definition bug, reported verbatim to its author."""


@dataclass(frozen=True)
class RunSpec:
    """Everything the rerun stage needs to run one experiment.

    All of it comes from the task repository, which the agent cannot see; no part of it is
    something the agent can influence.
    """

    task_id: str
    # Shell command executed at the workspace root, including whatever explicit overrides the
    # protocol requires.
    command: str
    # Environment passed to the command. Values may reference $AUTORLBENCH_RESULTS /
    # $AUTORLBENCH_WORKSPACE, expanded by the executing side into real paths for that backend.
    env: Dict[str, str]
    # Wall-clock budget for the command, in seconds. None falls back to EvaluationConfig.timeout_s.
    timeout_s: Optional[int]
    # Experiment log roots, relative to $AUTORLBENCH_RESULTS. They are deleted before the rerun:
    # grading only ever sees the system's fresh run, so an agent that ignored the protocol and
    # trained anyway gains nothing from the logs it left.
    log_roots: List[str]

    def expand_env(self, values: Dict[str, str]) -> Dict[str, str]:
        """Replace $AUTORLBENCH_* references in env values with this backend's real paths.

        `values` holds the AUTORLBENCH_* resolution for this backend / this request (keys without
        the `$`). A reference to a variable that is not there is left as written -- better than
        raising, which would blow one typo in one run_spec up into an orchestration failure for
        the whole evaluation.
        """
        expanded: Dict[str, str] = {}
        for key, value in self.env.items():
            if _EXPANDABLE_PREFIX not in value:
                expanded[key] = value
                continue
            for name, resolved in values.items():
                value = value.replace(
                    f"${name}", resolved if resolved else f"${name}"
                )
            expanded[key] = value
        return expanded


def load_run_spec(evaluation_root: Path, task_id: str) -> Optional[RunSpec]:
    """Load the task's run_spec.json, or None when it has none (no system rerun).

    A spec that exists but is invalid must fail here and now: discovering a broken spec halfway
    into the rerun looks like a harness bug when it is the task author who needs to fix it.
    """
    path = evaluation_root / "verifier" / "resources" / RUN_SPEC_FILENAME
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RunSpecError(f"{path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RunSpecError(f"{path}: top level must be an object")
    version = payload.get("schema_version")
    if version != RUN_SPEC_SCHEMA_VERSION:
        raise RunSpecError(
            f"{path}: schema_version must be {RUN_SPEC_SCHEMA_VERSION}, "
            f"got {version!r}"
        )
    command = payload.get("command")
    if not isinstance(command, str) or not command.strip():
        raise RunSpecError(f"{path}: command must be a non-empty shell string")
    env = payload.get("env", {})
    if not isinstance(env, dict):
        raise RunSpecError(f"{path}: env must be an object")
    for key, value in env.items():
        if not isinstance(key, str) or not key:
            raise RunSpecError(f"{path}: env keys must be non-empty strings")
        if not isinstance(value, (str, int, float, bool)):
            raise RunSpecError(
                f"{path}: env[{key!r}] must be a scalar, got {type(value).__name__}"
            )
        if key.startswith("AUTORLBENCH_"):
            raise RunSpecError(
                f"{path}: env must not declare {key}; the harness injects "
                "AUTORLBENCH_* variables itself"
            )
    log_roots = payload.get("log_roots", ["experiment_logs"])
    if not isinstance(log_roots, list) or not all(
        isinstance(item, str) and item for item in log_roots
    ):
        raise RunSpecError(
            f"{path}: log_roots must be a list of non-empty strings"
        )
    timeout_raw = payload.get("timeout_s")
    if timeout_raw is not None and (
        not isinstance(timeout_raw, (int, float))
        or isinstance(timeout_raw, bool)
        or timeout_raw <= 0
    ):
        raise RunSpecError(f"{path}: timeout_s must be a positive number")
    return RunSpec(
        task_id=task_id,
        command=command,
        env={key: str(value) for key, value in env.items()},
        timeout_s=None if timeout_raw is None else int(timeout_raw),
        log_roots=list(log_roots),
    )
