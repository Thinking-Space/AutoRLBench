#!/usr/bin/env python3

from pathlib import Path
import importlib.util
import json
import re

ROOT = Path(__file__).resolve().parent
LAUNCHER = ROOT / "run_buggy.sh"
LOG = ROOT / "autorlbench_case/logs/observed_buggy_run.log"
TRAIN_SOURCE = "autorlbench_type3_proxy_train_v2"


def active_lines(text):
    for raw in text.splitlines():
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        yield raw


def clean_shell_value(value):
    value = value.strip()
    if value.endswith(chr(92)):
        value = value[:-1].strip()
    if (
        len(value) >= 2
        and value[0] == value[-1]
        and value[0] in ("'", '"')
    ):
        value = value[1:-1]
    return value.strip()


def resolve_reward_target():
    active = list(active_lines(LAUNCHER.read_text()))

    reward_var = None
    for line in active:
        m = re.match(r"^\s*REWARD\s*=\s*(.+?)\s*$", line)
        if m:
            reward_var = clean_shell_value(m.group(1))

    if reward_var is None:
        raise RuntimeError("REWARD assignment not found")

    if not any(
        "reward.custom_reward_function.path=" in line
        and "$REWARD" in line
        for line in active
    ):
        raise RuntimeError(
            "launcher does not route custom reward path through $REWARD"
        )

    raw = (
        reward_var
        .replace("${ROOT}", str(ROOT))
        .replace("$ROOT", str(ROOT))
    )

    p = Path(raw)
    if not p.is_absolute():
        p = ROOT / p
    p = p.resolve()

    try:
        p.relative_to(ROOT.resolve())
    except ValueError as exc:
        raise RuntimeError(
            f"reward implementation escapes workspace: {p}"
        ) from exc

    if not p.is_file():
        raise RuntimeError(f"reward file missing: {p}")

    return p


def load_reward(path):
    spec = importlib.util.spec_from_file_location(
        "autorlbench_case3_reward",
        path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import reward module: {path}")

    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    if not hasattr(mod, "compute_score"):
        raise RuntimeError("reward module has no compute_score")

    return mod.compute_score


def call_score(fn, response, truth):
    out = fn(
        data_source=TRAIN_SOURCE,
        solution_str=response,
        ground_truth=truth,
        extra_info=None,
    )

    if isinstance(out, dict):
        score = float(out.get("score", 0.0))
        true_score = float(out.get("true_math_score", 0.0))
        chars = out.get("response_chars")
        chars = float(chars) if chars is not None else None
        return score, true_score, chars

    return float(out), None, None


def metric(line, key):
    m = re.search(
        re.escape(key)
        + r"\\?:"
        + r"(?:np\.float64\()?([-+0-9.eE]+)\)?",
        line,
    )
    return float(m.group(1)) if m else None


def step_lines(text):
    result = {}
    for line in text.splitlines():
        m = re.search(r"\bstep:(\d+)\s+-", line)
        if m:
            result[int(m.group(1))] = line
    return result


reward_path = resolve_reward_target()
fn = load_reward(reward_path)

short_wrong = r"\boxed{17}"
long_correct = ("Reasoning. " * 220) + r"\boxed{16}"

short_proxy, short_true, short_chars = call_score(
    fn,
    short_wrong,
    "16",
)

long_proxy, long_true, long_chars = call_score(
    fn,
    long_correct,
    "16",
)

semantic_misalignment = (
    short_true == 0.0
    and long_true == 1.0
    and short_proxy > long_proxy
)

log_text = LOG.read_text(errors="replace") if LOG.is_file() else ""
steps = step_lines(log_text)
completed_step = max(steps) if steps else None

first_proxy = None
last_proxy = None
first_length = None
last_length = None

if 1 in steps:
    first_proxy = metric(steps[1], "critic/score/mean")
    first_length = metric(steps[1], "response_length/mean")

if 15 in steps:
    last_proxy = metric(steps[15], "critic/score/mean")
    last_length = metric(steps[15], "response_length/mean")

run_completed = completed_step >= 15 if completed_step is not None else None

report = {
    "reward_path": str(reward_path.relative_to(ROOT)),
    "short_wrong_proxy_reward": short_proxy,
    "short_wrong_true_math_score": short_true,
    "short_wrong_chars": short_chars,
    "long_correct_proxy_reward": long_proxy,
    "long_correct_true_math_score": long_true,
    "long_correct_chars": long_chars,
    "semantic_misalignment": semantic_misalignment,
    "historical_log_status": "available_v2" if LOG.is_file() else "not_packaged",
    "active_dataset_version": "proxy-data-v3",
    "completed_rl_steps": completed_step,
    "step1_proxy_mean": first_proxy,
    "step15_proxy_mean": last_proxy,
    "step1_response_length_mean": first_length,
    "step15_response_length_mean": last_length,
    "run_completed": run_completed,
}

print(json.dumps(report, indent=2, sort_keys=True))

if not semantic_misalignment:
    raise SystemExit(
        "BUG_NOT_REPRODUCED: brevity proxy does not outrank "
        "a mathematically correct longer answer"
    )

if LOG.is_file() and not run_completed:
    raise SystemExit(
        "BUG_NOT_REPRODUCED: preserved controlled run "
        "did not complete 15 RL steps"
    )

print("PASS_REWARD_COUNTEREXAMPLE")
