"""Score the existing DAPO Answer: prompts using the pinned veRL math scorer."""
import importlib.util
from pathlib import Path
import re

from verl.utils.reward_score.math_dapo import compute_score as _math_dapo_score

_BASE_MODULES = {}


def dapo_score(solution_str, ground_truth):
    truth = str(ground_truth).strip()
    if not re.fullmatch(r"[+-]?[0-9]+", truth):
        raise ValueError("DAPO ground_truth must be a plain integer")
    result = _math_dapo_score(str(solution_str), truth, strict_box_verify=False)
    return float(result["acc"])


def _bucket(extra_info, context_cutoff=None):
    info = extra_info or {}
    if context_cutoff is not None and "prompt_token_length" in info:
        return "short" if int(info["prompt_token_length"]) <= int(context_cutoff) else "long"
    return str(info.get("context_bucket", "all"))


def compute_score(data_source, solution_str, ground_truth, extra_info=None, context_cutoff=None, **kwargs):
    correct = dapo_score(solution_str, ground_truth)
    return {"score": correct, "acc": correct, "oracle_acc": correct,
            "context_bucket": _bucket(extra_info, context_cutoff),
            "data_source": str(data_source),
            "problem_id": str((extra_info or {}).get("problem_id") or "")}


def _batch_module(base_reward_path=None):
    if base_reward_path is None:
        sibling = Path(__file__).resolve().with_name("reward.py")
        base_reward_path = sibling if sibling.is_file() else Path.cwd() / "case/reward.py"
    path = Path(base_reward_path).resolve()
    if path not in _BASE_MODULES:
        spec = importlib.util.spec_from_file_location(f"_dapo_base_reward_{len(_BASE_MODULES)}", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        # Retain the workspace's callbacks, replacing only answer validation.
        module.answer_matches = dapo_score
        _BASE_MODULES[path] = module
    return _BASE_MODULES[path]


def compute_score_batch(data_sources, solution_strs, ground_truths, extra_infos=None,
                        base_reward_path=None, context_cutoff=None, **kwargs):
    if not (len(data_sources) == len(solution_strs) == len(ground_truths)):
        raise ValueError("Scoring input lengths must match")
    if extra_infos is None:
        extra_infos = [{} for _ in solution_strs]
    if len(extra_infos) != len(solution_strs):
        raise ValueError("Scoring metadata length must match completions")
    if not solution_strs:
        return []
    scores = _batch_module(base_reward_path).compute_score(
        data_sources, solution_strs, ground_truths, extra_infos=extra_infos, **kwargs)
    return [dict(result, context_bucket=_bucket(info, context_cutoff), data_source=str(source),
                 problem_id=str((info or {}).get("problem_id") or ""))
            for result, info, source in zip(scores, extra_infos, data_sources, strict=True)]
