# Parse A/B driver logs into per-step CSVs plus a summary comparison.
# Usage: python3 parse_ab.py <bug_run_dir> <fix_run_dir> <out_dir>
import ast
import csv
import glob
import json
import os
import re
import sys

STEP_RE = re.compile(r"\[([0-9-]+ [0-9:.]+) .*log_utils\.py:66 - rollout (\d+): (\{.*\})")
PERF_RE = re.compile(
    r"aborted_groups_recycled': (\d+), 'rollout/fully_async/stale_groups_recycled': (\d+)"
)


def parse_arm(run_dir):
    logs = sorted(glob.glob(os.path.join(run_dir, "driver_*.log")))
    assert logs, f"no driver logs under {run_dir}"
    steps = []
    recycle = []
    for path in logs:
        with open(path, errors="replace") as f:
            for line in f:
                m = STEP_RE.search(line)
                if m:
                    ts, sid, payload = m.group(1), int(m.group(2)), m.group(3)
                    try:
                        d = ast.literal_eval(payload)
                    except (ValueError, SyntaxError):
                        continue
                    steps.append(
                        {
                            "step": sid,
                            "time": ts,
                            "reward": d.get("rollout/rewards"),
                            "response_len": d.get("rollout/response_lengths"),
                        }
                    )
                m = PERF_RE.search(line)
                if m:
                    recycle.append((int(m.group(1)), int(m.group(2))))
    return steps, recycle


def main():
    bug_dir, fix_dir, out_dir = sys.argv[1:4]
    os.makedirs(out_dir, exist_ok=True)
    summary = {}
    for arm, run_dir in (("bug", bug_dir), ("fix", fix_dir)):
        steps, recycle = parse_arm(run_dir)
        with open(os.path.join(out_dir, f"{arm}_steps.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["step", "time", "reward", "response_len"])
            w.writeheader()
            w.writerows(steps)
        # Recycle counters are per-drain; totals are the per-step sums.
        aborted_total = sum(a for a, _ in recycle)
        stale_total = sum(s for _, s in recycle)
        mean_len = sum(s["response_len"] for s in steps) / len(steps) if steps else 0.0
        rewards = [s["reward"] for s in steps if s["reward"] is not None]
        summary[arm] = {
            "train_steps": len(steps),
            "aborted_groups_recycled_total": aborted_total,
            "stale_groups_recycled_total": stale_total,
            "mean_response_len": round(mean_len, 1),
            "mean_reward": round(sum(rewards) / len(rewards), 4) if rewards else None,
            # Stale groups decoded to completion before being dropped (full length);
            # aborted groups were evicted mid-flight (roughly half length on average).
            "wasted_tokens_stale_full": int(stale_total * 8 * mean_len),
            "wasted_tokens_aborted_half": int(aborted_total * 8 * mean_len / 2),
            "trained_tokens_estimate": int(len(steps) * 256 * mean_len),
        }
    if summary["fix"]["train_steps"]:
        summary["bug_vs_fix_step_ratio"] = round(
            summary["bug"]["train_steps"] / summary["fix"]["train_steps"], 3
        )
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
