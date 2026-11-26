"""Distinguish "fluent but prompt-blind" responses from corruption-style failures.

Reads the recorded rollout dumps (workspace/outputs/grpo/rollouts/*.jsonl), and
for every response that follows the <answer>...</answer> format checks whether
the numbers inside the answer expression are the prompt's numbers. A response
can be perfectly fluent, well-formatted, and still ignore the prompt entirely.

    python3 answer_grounding_check.py <rollouts_dir>
"""
import json
import re
import sys
from pathlib import Path


def check_file(path: Path):
    rows = [json.loads(line) for line in path.open() if line.strip()]
    tagged = correct_numbers = correct_value = 0
    samples = []
    for r in rows:
        m = re.findall(r"<answer>(.*?)</answer>", r["output"], re.S)
        if not m:
            continue
        tagged += 1
        gt = json.loads(r["gts"])
        expr = m[-1].strip().split("\n")[0]
        used = sorted(int(x) for x in re.findall(r"-?\d+", expr))
        if used == sorted(gt["numbers"]):
            correct_numbers += 1
            try:
                val = eval(expr.replace("×", "*").replace("÷", "/"))
                if abs(val - gt["target"]) < 1e-9:
                    correct_value += 1
            except Exception:
                pass
        elif len(samples) < 3:
            samples.append((gt["numbers"], gt["target"], expr[:60]))
    return len(rows), tagged, correct_numbers, correct_value, samples


def main():
    d = Path(sys.argv[1])
    for f in sorted(d.glob("*.jsonl"), key=lambda p: int(p.stem)):
        n, tagged, nums, vals, samples = check_file(f)
        print(f"step {f.stem:>3}: {tagged:>3}/{n} responses use the <answer> format; "
              f"{nums:>3} use the prompt's numbers; {vals:>3} evaluate to the target")
        for prompt_nums, target, expr in samples:
            print(f"          prompt numbers {prompt_nums} (target {target}) -> answered {expr!r}")


if __name__ == "__main__":
    main()
