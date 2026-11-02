"""Per training step, how diverse are the n samples of each GRPO group? Reads verl rollout dumps (<step>.jsonl).
Reports: groups, mean distinct responses per group, share of groups whose n responses are all identical, and share of
groups with any reward spread (only those carry GRPO advantage).
Usage: python group_distinct.py <rollouts dir> [<rollouts dir> ...]"""
import glob
import json
import os
import sys
from collections import defaultdict


def main():
    for d in sys.argv[1:]:
        print(f"=== {d}")
        files = sorted(glob.glob(os.path.join(d, "*.jsonl")), key=lambda p: int(os.path.basename(p).split(".")[0]))
        for path in files:
            groups = defaultdict(list)
            for line in open(path):
                r = json.loads(line)
                groups[r["input"]].append((r["output"], float(r["score"])))
            n = len(groups)
            distinct = [len({o for o, _ in g}) for g in groups.values()]
            identical = sum(k == 1 for k in distinct)
            spread = sum(len({s for _, s in g}) > 1 for g in groups.values())
            size = sum(len(g) for g in groups.values()) / n
            print(f"step {os.path.basename(path).split('.')[0]:>3}: groups={n} size={size:.1f} "
                  f"distinct/group={sum(distinct) / n:.2f} all-identical={identical / n:.2f} with-reward-spread={spread / n:.2f}")


if __name__ == "__main__":
    main()
