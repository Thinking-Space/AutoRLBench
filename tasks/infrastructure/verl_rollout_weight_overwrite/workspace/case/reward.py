import re

def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    matches = re.findall(r"<answer>\s*(-?\d+)\s*</answer>", solution_str)
    correct = float(len(matches) == 1 and matches[0] == str(ground_truth))
    return {"score": correct, "oracle_acc": correct}
