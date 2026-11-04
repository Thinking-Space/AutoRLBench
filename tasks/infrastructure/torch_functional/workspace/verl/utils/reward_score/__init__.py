# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# from . import gsm8k, math, prime_math, prime_code

from __future__ import annotations

import importlib.util
import os
import threading
from typing import Any, Callable, Optional

from verl.utils.import_utils import deprecated

# data_source -> sealed file relative to BENCH_BASELINE_REWARD_ROOT (/hidden/baseline)
_BASELINE_SCORE_REL: dict[str, str] = {
    "openai/gsm8k": "verl/utils/reward_score/gsm8k.py",
}
_BASELINE_FN_CACHE: dict[str, Callable[..., Any]] = {}
_BASELINE_LOCK = threading.Lock()
_BASELINE_WARNED = False


def _baseline_reward_root() -> str:
    """Sealed baseline root (Agent-invisible: /hidden or host hidden/baseline)."""
    env = (os.environ.get("BENCH_BASELINE_REWARD_ROOT") or "").strip()
    if env and os.path.isdir(env):
        return env
    # Docker verify mount (see DockerEnv.mount_hidden)
    if os.path.isdir("/hidden/baseline"):
        return "/hidden/baseline"
    return ""


def _load_baseline_compute_score(rel_path: str) -> Optional[Callable[..., Any]]:
    """Load compute_score from sealed baseline via importlib (not workspace package)."""
    root = _baseline_reward_root()
    if not root:
        return None
    path = os.path.join(root, rel_path.replace("\\", "/").lstrip("/"))
    if not os.path.isfile(path):
        return None
    with _BASELINE_LOCK:
        cached = _BASELINE_FN_CACHE.get(path)
        if cached is not None:
            return cached
        mod_name = "_bench_sealed_reward_" + str(abs(hash(path)))
        spec = importlib.util.spec_from_file_location(mod_name, path)
        if spec is None or spec.loader is None:
            return None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        fn = getattr(mod, "compute_score", None)
        if fn is None:
            return None
        _BASELINE_FN_CACHE[path] = fn
        return fn


def _is_validate(extra_info) -> bool:
    if not isinstance(extra_info, dict):
        return False
    return bool(extra_info.get("validate"))


def _try_baseline_score(data_source: str, solution_str, ground_truth, extra_info):
    """If validate=True and sealed scorer exists, return score; else None."""
    global _BASELINE_WARNED
    if not _is_validate(extra_info):
        return None
    rel = _BASELINE_SCORE_REL.get(data_source)
    if not rel:
        return None
    fn = _load_baseline_compute_score(rel)
    if fn is None:
        if not _BASELINE_WARNED:
            _BASELINE_WARNED = True
            print(
                "[bench] WARN: validate=True but sealed baseline scorer missing "
                f"for {data_source=} (BENCH_BASELINE_REWARD_ROOT="
                f"{os.environ.get('BENCH_BASELINE_REWARD_ROOT', '')!r}); "
                "falling back to workspace scorer (val-core may be polluted).",
                flush=True,
            )
        return None
    return fn(solution_str, ground_truth)


def default_compute_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    sandbox_fusion_url=None,
    concurrent_semaphore=None,
    memory_limit_mb=None,
    **kwargs,
):
    """Compute the score for a given solution based on the data source.

    Args:
        data_source (str): The source dataset identifier which determines the scoring method.
        solution_str (str): The solution string to be evaluated.
        ground_truth (str): The ground truth answer for comparison.
        extra_info (dict, optional): Additional information that might be needed for scoring. Defaults to None.

    Returns:
        float: The computed score as a floating point number. If the result is a dictionary,
               it returns the dictionary instead.

    Raises:
        NotImplementedError: If the reward function is not implemented for the given data source.
    """
    baseline_res = _try_baseline_score(data_source, solution_str, ground_truth, extra_info)
    if baseline_res is not None:
        res = baseline_res
    elif data_source == "openai/gsm8k":
        from . import gsm8k

        res = gsm8k.compute_score(solution_str, ground_truth)
    elif data_source in ["lighteval/MATH", "DigitalLearningGmbH/MATH-lighteval", "HuggingFaceH4/MATH-500"]:
        from . import math_reward

        res = math_reward.compute_score(solution_str, ground_truth)
        # [Optional] Math-Verify Integration
        # For enhanced accuracy, consider utilizing Math-Verify (https://github.com/huggingface/Math-Verify).
        # Note: Math-Verify needs to be manually installed via pip: `pip install math-verify`.
        # To use it, override the `compute_score` function with the following implementation:

        # from . import math_verify
        # res = math_verify.compute_score(solution_str, ground_truth)
    elif data_source in ["math_dapo", "math", "math_dapo_reasoning"] or data_source.startswith("aime"):
        from . import math_dapo

        res = math_dapo.compute_score(solution_str, ground_truth)
    elif data_source in [
        "numina_aops_forum",
        "numina_synthetic_math",
        "numina_amc_aime",
        "numina_synthetic_amc",
        "numina_cn_k12",
        "numina_olympiads",
    ]:
        from . import prime_math

        res = prime_math.compute_score(solution_str, ground_truth)
    elif data_source == "taco":
        from . import taco

        res = taco.compute_score(
            solution_str=solution_str, ground_truth=ground_truth, extra_info=extra_info
        )
    elif data_source in ["codecontests", "apps", "codeforces"]:
        # Use the passed sandbox_fusion_url if available
        if sandbox_fusion_url:
            from . import sandbox_fusion

            # Pass the URL directly, ground_truth likely contains test cases here
            res = sandbox_fusion.compute_score(
                sandbox_fusion_url, concurrent_semaphore, memory_limit_mb, solution_str, ground_truth, continuous=True
            )
        else:
            # If no sandbox URL is provided, fall back to prime_code or raise error
            from . import prime_code

            # Assuming prime_code doesn't need the URL
            res = prime_code.compute_score(solution_str, ground_truth, continuous=True)
    elif data_source in ["hiyouga/geometry3k"]:
        from . import geo3k

        res = geo3k.compute_score(solution_str, ground_truth)
    elif data_source in [
        "searchR1_nq",
        "searchR1_triviaqa",
        "searchR1_popqa",
        "searchR1_hotpotqa",
        "searchR1_2wikimultihopqa",
        "searchR1_musique",
        "searchR1_bamboogle",
    ]:
        from . import search_r1_like_qa_em

        res = search_r1_like_qa_em.compute_score(solution_str, ground_truth)

    else:
        raise NotImplementedError(f"Reward function is not implemented for {data_source=}")

    if isinstance(res, dict):
        return res
    elif isinstance(res, int | float | bool):
        return float(res)
    else:
        return float(res[0])


@deprecated("verl.utils.reward_score.default_compute_score")
def _default_compute_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    sandbox_fusion_url=None,
    concurrent_semaphore=None,
    memory_limit_mb=None,
):
    """
    Legacy function API to be deprecated. Please use `default_compute_score` instead.
    """
    return default_compute_score(
        data_source, solution_str, ground_truth, extra_info, sandbox_fusion_url, concurrent_semaphore, memory_limit_mb
    )


def get_default_compute_score(reward_name: str | None):
    """Get the default compute_score function based on the reward manager type."""
    return default_compute_score


__all__ = ["default_compute_score"]
