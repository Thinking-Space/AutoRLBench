"""Check the submitted foreground run logs; this is not a reward-quality gate."""
import math
from pathlib import Path
import re

NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?|nan|inf"


def inspect_training(log_dir, required_steps):
    root = Path(log_dir)
    path = root / "train.log"
    if path.is_symlink() or not path.is_file():
        return {"passed": False, "reason": "missing fresh experiment_logs/train.log"}
    text = path.read_text(errors="replace")
    lines = [line for line in text.splitlines() if re.search(r"\bstep:\d+\s+-", line)]
    steps = sorted({int(m.group(1)) for line in lines if (m := re.search(r"\bstep:(\d+)\s+-", line))})
    norms = [float(m.group(1)) for line in lines
             for m in re.finditer(r"\bactor/grad_norm:\s*(" + NUMBER + r")", line, re.IGNORECASE)]
    vals = [int(m.group(1)) for line in lines if "val-core/" in line
            if (m := re.search(r"\bstep:(\d+)\s+-", line))]
    within_budget = all(0 <= step <= required_steps for step in steps)
    complete = all(step in steps for step in range(1, required_steps + 1)) and within_budget
    gradients = bool(norms) and all(math.isfinite(x) for x in norms) and any(x > 0 for x in norms)
    wandb = (root / "wandb").is_dir() and any((root / "wandb").rglob("*.wandb"))
    passed = complete and gradients and len(set(vals)) >= 2 and wandb
    return {"passed": passed, "observed_steps": steps, "required_steps": required_steps,
            "within_training_budget": within_budget,
            "finite_positive_gradient_observed": gradients, "validation_steps": sorted(set(vals)),
            "offline_wandb_present": wandb,
            "scope": "training completion/finite gradients, not end-to-end reward improvement"}
