# Reward for AIME-style RL: mimics the evaluation scorer (inspect match(numeric=True), last number of the
# completion after removing \boxed{}), applied to the text after </think>. Unfinished/truncated -> 0.
import re
from inspect_ai.scorer._common import match_str

def _remove_boxed(s):
    return re.sub(r"\\boxed\{(.+)\}", r"\1", s)

def compute_score(data_source=None, solution_str=None, ground_truth=None, extra_info=None, **kwargs):
    text = solution_str or ""
    ground_truth = str(ground_truth).strip()
    if re.fullmatch(r"-?\d+\.0+", ground_truth):
        ground_truth = ground_truth.split(".")[0]
    if "</think>" not in text:
        return {"score": 0.0, "acc": 0.0, "fmt": 0.0}
    ans = text.split("</think>")[-1]
    if len(ans.strip()) == 0:
        return {"score": 0.0, "acc": 0.0, "fmt": 0.0}
    has_line = 1.0 if re.search(r"ANSWER:\s*\S+", ans) else 0.0
    try:
        _, ok = match_str(_remove_boxed(ans), str(ground_truth), location="end", numeric=True)
    except Exception:
        ok = False
    acc = 1.0 if ok else 0.0
    return {"score": acc, "acc": acc, "fmt": has_line}
