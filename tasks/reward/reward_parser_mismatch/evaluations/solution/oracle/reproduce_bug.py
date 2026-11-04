#!/usr/bin/env python3
"""Maintainer-side check that the staged defect is still in place.

Lives next to the golden fix (not in the agent-visible workspace). Resolves the
reward function the run actually wires: the Hydra override on the
`python3 -m recipe.vimpo.main_vimpo` command line in run_vimpo_critic.sh wins
over the `custom_reward_function` entry in vimpo_trainer.yaml, so this script
parses the recipe script, not the YAML. The override itself is carried by
REWARD_OVERRIDE from config/scoring.env (sourced when run_main.sh sets
SMOKE_SCORING=1), so resolution follows that chain.
"""
from pathlib import Path
import importlib.util
import json
import re

TASK_ROOT = Path(__file__).resolve().parents[3]
ROOT = TASK_ROOT / "workspace"
RECIPE = ROOT / "recipe" / "vimpo" / "run_vimpo_critic.sh"
SCORING_ENV = ROOT / "recipe" / "vimpo" / "config" / "scoring.env"
RUN_MAIN = ROOT / "run_main.sh"


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


def resolve_scoring_env_override(active_recipe_lines):
    """Follow the smoke-lane chain: run_main.sh sets SMOKE_SCORING=1, the recipe
    script sources config/scoring.env under that flag, and the env file carries
    the reward overrides in REWARD_OVERRIDE."""
    recipe_text = "\n".join(active_recipe_lines)
    sources_scoring = re.search(
        r"source\s+[" r"'\"]?" r"\$\{SCRIPT_DIR\}/config/scoring\.env", recipe_text
    )
    if sources_scoring is None:
        return None
    gate = re.search(r'\$\{SMOKE_SCORING:-0\}"?\s*=\s*"?1', recipe_text)
    main_text = RUN_MAIN.read_text()
    smoke_on = re.search(r"^\s*(?:export\s+)?SMOKE_SCORING=1\b", main_text, re.M)
    if gate is not None and smoke_on is None:
        raise RuntimeError("recipe gates scoring.env on SMOKE_SCORING but run_main.sh does not set it")
    env_text = SCORING_ENV.read_text()
    m = re.search(
        r"REWARD_OVERRIDE=([\"'])(.*?)\1", "".join(active_lines(env_text)), re.S
    )
    if m is None:
        return None
    value = m.group(2)
    pm = re.search(r"reward\.custom_reward_function\.path=(\S+)", value)
    if pm is None:
        return None
    return pm.group(1)


def resolve_reward_target():
    text = RECIPE.read_text()
    active = list(active_lines(text))
    override = None
    for line in active:
        m = re.search(r"reward\.custom_reward_function\.path=(\S+)", line)
        if m:
            override = clean_shell_value(m.group(1))
    if override is None:
        override = resolve_scoring_env_override(active)
    if override is None:
        raise RuntimeError(
            "run_vimpo_critic.sh sets neither reward.custom_reward_function.path "
            "directly nor via config/scoring.env"
        )
    override = (
        override.replace("${PROJECT_ROOT}", str(ROOT))
        .replace("$PROJECT_ROOT", str(ROOT))
        .replace("${ROOT}", str(ROOT))
        .replace("$ROOT", str(ROOT))
    )
    p = Path(override)
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
