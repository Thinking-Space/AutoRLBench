#!/usr/bin/env python3
"""Compare two arms' held-out scores, paired by task.

Reads the per-episode dumps written by `et/eval_log.py` (one JSONL per eval
point, one row per episode) from each arm and answers two questions:

  1. the curve -- pass rate per eval point, which is what says whether training
     has converged and where;
  2. the verdict -- at the last eval point both arms reached, is the repaired
     arm's pass rate higher, and by enough to not be sampling noise?

The verdict is *paired*. Both arms are evaluated on the same split with the
same per-task sample budget, so each task contributes one difference
`fixed_mean - buggy_mean` and task difficulty -- which dominates the variance
of either arm's raw pass rate -- cancels before anything is averaged.

Two tests are reported because they ask different questions and can disagree:

  * a sign-flip permutation test on the per-task mean differences. Exact under
    the null that the two arms are exchangeable within a task, assumes nothing
    about the distribution of the differences, and needs no scipy.
  * McNemar's test on "solved at least once", which ignores how *often* each
    arm solved a task and asks only whether the set of solvable tasks grew.

    python3 tools/compare_arms.py --buggy OUT_BUGGY/eval_episodes \
                                  --fixed OUT_FIXED/eval_episodes \
                                  --dataset et_test
"""

from __future__ import annotations

import argparse
import collections
import glob
import gzip
import json
import math
import os
import random

# Fixed so a verdict is reproducible. The permutation p-value is a Monte Carlo
# estimate, and a p-value that moved between runs of the *analysis* would be
# one more number nobody can check.
PERM_SEED = 20261009
PERM_ROUNDS = 20000


def load(dump_dir: str, dataset: str) -> dict[int, dict[str, list[float]]]:
    """-> {rollout_id: {label: [reward, ...]}} for one dataset.

    `.jsonl.gz` is accepted alongside `.jsonl` so the archived copies under
    `results/` can be compared with the same command as a live run's output.
    """
    out: dict[int, dict[str, list[float]]] = collections.defaultdict(lambda: collections.defaultdict(list))
    paths = glob.glob(os.path.join(dump_dir, "eval_rollout_*.jsonl"))
    paths += glob.glob(os.path.join(dump_dir, "eval_rollout_*.jsonl.gz"))
    for path in sorted(paths):
        opener = gzip.open if path.endswith(".gz") else open
        with opener(path, "rt", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("dataset") != dataset or row.get("reward") is None:
                    continue
                out[row["rollout_id"]][row["label"]].append(float(row["reward"]))
    return {k: dict(v) for k, v in out.items()}


def pass_rate(point: dict[str, list[float]]) -> tuple[float, float, int, int]:
    """(episode pass rate, task solved-at-least-once rate, n_tasks, n_episodes)."""
    episodes = [r for rewards in point.values() for r in rewards]
    solved = [1.0 if any(r > 0 for r in rewards) else 0.0 for rewards in point.values()]
    n_ep = len(episodes)
    n_task = len(solved)
    return (
        sum(episodes) / n_ep if n_ep else 0.0,
        sum(solved) / n_task if n_task else 0.0,
        n_task,
        n_ep,
    )


def permutation_p(diffs: list[float], rounds: int = PERM_ROUNDS) -> float:
    """Two-sided sign-flip permutation p-value for mean(diffs) != 0.

    Under the null the two arms are exchangeable within a task, so flipping the
    sign of any subset of the per-task differences is equally likely. Reported
    with the observed statistic counted in both numerator and denominator,
    which keeps the p-value from ever being exactly 0 -- 20000 draws cannot
    distinguish "very unlikely" from "impossible".
    """
    n = len(diffs)
    if n == 0:
        return float("nan")
    observed = abs(sum(diffs) / n)
    rng = random.Random(PERM_SEED)
    hits = 0
    for _ in range(rounds):
        total = 0.0
        for d in diffs:
            total += d if rng.random() < 0.5 else -d
        if abs(total / n) >= observed - 1e-12:
            hits += 1
    return (hits + 1) / (rounds + 1)


def mcnemar(buggy: dict[str, list[float]], fixed: dict[str, list[float]], labels: list[str]) -> dict:
    """Exact binomial McNemar on "solved at least once"."""
    b = sum(1 for label in labels if any(r > 0 for r in buggy[label]) and not any(r > 0 for r in fixed[label]))
    c = sum(1 for label in labels if not any(r > 0 for r in buggy[label]) and any(r > 0 for r in fixed[label]))
    n = b + c
    # Two-sided exact binomial on the discordant pairs under p = 1/2.
    if n == 0:
        p = float("nan")
    else:
        k = min(b, c)
        tail = sum(math.comb(n, i) for i in range(k + 1)) / (2**n)
        p = min(1.0, 2 * tail)
    return {"only_buggy_solved": b, "only_fixed_solved": c, "discordant": n, "p_two_sided": p}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--buggy", required=True, help="buggy arm's eval_episodes directory")
    ap.add_argument("--fixed", required=True, help="fixed arm's eval_episodes directory")
    ap.add_argument("--dataset", default="et_test", help="eval dataset name, as named in --eval-prompt-data")
    ap.add_argument("--out", help="write the full result as JSON here")
    args = ap.parse_args()

    arms = {"buggy": load(args.buggy, args.dataset), "fixed": load(args.fixed, args.dataset)}
    for name, points in arms.items():
        if not points:
            raise SystemExit(f"no '{args.dataset}' episodes found for the {name} arm")

    print(f"=== {args.dataset}: pass rate per eval point ===")
    print(f"{'rollout':>8}  {'buggy ep':>9} {'buggy task':>11}  {'fixed ep':>9} {'fixed task':>11}")
    curve = []
    for rollout_id in sorted(set(arms["buggy"]) | set(arms["fixed"])):
        row: dict = {"rollout_id": rollout_id}
        cells = []
        for name in ("buggy", "fixed"):
            point = arms[name].get(rollout_id)
            if point is None:
                cells.append(f"{'-':>9} {'-':>11}")
                continue
            ep, task, n_task, n_ep = pass_rate(point)
            row[name] = {"episode_pass_rate": ep, "task_pass_rate": task, "n_tasks": n_task, "n_episodes": n_ep}
            cells.append(f"{ep:>9.4f} {task:>11.4f}")
        curve.append(row)
        print(f"{rollout_id:>8}  {cells[0]}  {cells[1]}")

    # The verdict point is the last one *both* arms reached: comparing an arm's
    # final score against the other's intermediate one would be a different
    # claim than the one being made.
    shared = sorted(set(arms["buggy"]) & set(arms["fixed"]))
    if not shared:
        raise SystemExit("the two arms share no eval point")
    final = shared[-1]
    buggy, fixed = arms["buggy"][final], arms["fixed"][final]

    labels = sorted(set(buggy) & set(fixed))
    dropped = (set(buggy) ^ set(fixed)) & (set(buggy) | set(fixed))
    diffs = [sum(fixed[l]) / len(fixed[l]) - sum(buggy[l]) / len(buggy[l]) for l in labels]
    n = len(diffs)
    mean_d = sum(diffs) / n
    var = sum((d - mean_d) ** 2 for d in diffs) / (n - 1) if n > 1 else 0.0
    se = math.sqrt(var / n) if n else float("nan")
    p_perm = permutation_p(diffs)
    mc = mcnemar(buggy, fixed, labels)

    print(f"\n=== verdict at eval point {final} ({n} tasks scored by both arms) ===")
    if dropped:
        print(f"  note: {len(dropped)} task(s) present in only one arm were excluded")
    print(f"  buggy  episode pass rate   {pass_rate(buggy)[0]:.4f}")
    print(f"  fixed  episode pass rate   {pass_rate(fixed)[0]:.4f}")
    print(f"  paired mean difference     {mean_d:+.4f}  (SE {se:.4f}, 95% CI "
          f"{mean_d - 1.96 * se:+.4f} .. {mean_d + 1.96 * se:+.4f})")
    print(f"  sign-flip permutation p    {p_perm:.4g}  ({PERM_ROUNDS} draws, two-sided)")
    print(f"  McNemar on solved-at-least-once: buggy-only {mc['only_buggy_solved']}, "
          f"fixed-only {mc['only_fixed_solved']}, p {mc['p_two_sided']:.4g}")

    result = {
        "dataset": args.dataset,
        "curve": curve,
        "final_eval_point": final,
        "n_tasks_paired": n,
        "n_tasks_excluded": len(dropped),
        "buggy_episode_pass_rate": pass_rate(buggy)[0],
        "fixed_episode_pass_rate": pass_rate(fixed)[0],
        "paired_mean_difference": mean_d,
        "paired_se": se,
        "permutation_p_two_sided": p_perm,
        "permutation_rounds": PERM_ROUNDS,
        "mcnemar": mc,
    }
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2)
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
