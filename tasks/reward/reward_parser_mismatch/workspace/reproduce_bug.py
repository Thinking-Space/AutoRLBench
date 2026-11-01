#!/usr/bin/env python3
from pathlib import Path
import importlib.util
import json
import re

ROOT = Path(__file__).resolve().parent
LAUNCHER = ROOT / "run_buggy.sh"

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
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        value = value[1:-1]
    return value.strip()

def resolve_reward_target():
    text = LAUNCHER.read_text()
    active = list(active_lines(text))
    reward_var = None
    for line in active:
        m = re.match(r"^\s*REWARD\s*=\s*(.+?)\s*$", line)
        if m:
            reward_var = clean_shell_value(m.group(1))
    if not any("reward.custom_reward_function.path=" in line and "$REWARD" in line for line in active):
        raise RuntimeError("run_buggy.sh does not route custom reward path through $REWARD")
    if reward_var is None:
        raise RuntimeError("REWARD assignment not found in run_buggy.sh")
    reward_var = reward_var.replace("${ROOT}", str(ROOT)).replace("$ROOT", str(ROOT))
    p = Path(reward_var)
    if not p.is_absolute():
        p = ROOT / p
    p = p.resolve()
    try:
        p.relative_to(ROOT.resolve())
    except ValueError as exc:
        raise RuntimeError(f"reward implementation escapes workspace: {p}") from exc
    if not p.is_file():
        raise RuntimeError(f"reward implementation not found: {p}")
    return p

def load_module(path):
    spec = importlib.util.spec_from_file_location("autorlbench_case2_reward", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load reward module: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def call_score(fn, text, gt):
    out = fn(data_source="math__math", solution_str=text, ground_truth=gt, extra_info=None)
    if isinstance(out, dict):
        return float(out.get("score", 0.0)), out.get("parser_matched")
    return float(out), None

reward_path = resolve_reward_target()
module = load_module(reward_path)
if not hasattr(module, "compute_score"):
    raise RuntimeError("reward module has no compute_score")
fn = module.compute_score

boxed_score, boxed_matched = call_score(fn, r"The answer is \boxed{16}.", "16")
xml_score, xml_matched = call_score(fn, "<answer>16</answer>", "16")
wrong_xml_score, _ = call_score(fn, "<answer>17</answer>", "16")

bug_present = (
    boxed_score == 0.0
    and xml_score == 1.0
    and wrong_xml_score == 0.0
    and boxed_matched is False
    and xml_matched is True
)

report = {
    "reward_path": str(reward_path.relative_to(ROOT)),
    "boxed_correct_score": boxed_score,
    "boxed_parser_matched": boxed_matched,
    "xml_correct_score": xml_score,
    "xml_parser_matched": xml_matched,
    "xml_wrong_score": wrong_xml_score,
    "bug_present": bug_present,
}
print(json.dumps(report, indent=2, sort_keys=True))
if not bug_present:
    raise SystemExit("BUG_NOT_REPRODUCED: boxed-vs-XML parser mismatch not observed")
print("PASS_BUG")
