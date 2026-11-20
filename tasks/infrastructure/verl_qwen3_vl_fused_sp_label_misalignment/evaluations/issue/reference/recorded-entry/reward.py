"""Exact final boxed option; no model judge or answer-dependent shaping."""
import re

def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    matches=re.findall(r'\\boxed\s*\{\s*([A-Z])\s*\}',solution_str)
    option=matches[-1] if matches else None
    options=(extra_info or {}).get('num_choices',26)
    valid=option is not None and 0 <= ord(option)-ord('A') < int(options)
    correct=bool(valid and option==str(ground_truth))
    return {'score':float(correct),'acc':float(correct),'format_correct':float(valid)}
