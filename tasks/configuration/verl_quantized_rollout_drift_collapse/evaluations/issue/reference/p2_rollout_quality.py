"""P2 · rollout coherence check.

Reads the rollout dumps run_grpo.sh leaves behind (outputs/<exp>/rollouts/<step>.jsonl,
one {input, output, gts, score, step, acc} per line) and prints per-step quality
statistics: reward, <answer>-tag rate, empty-output rate, '!'-only degenerate count,
and mean length. Fluency with zero reward distinguishes a prompt-grounding failure
from corruption-style failures ('!'-spam, whitespace outputs).

    python3 p2_rollout_quality.py outputs/grpo outputs/<another-run>
"""
import argparse
import json
import statistics
from pathlib import Path


def step_stats(path: Path) -> dict | None:
    rows = [json.loads(line) for line in path.open() if line.strip()]
    if not rows:
        return None
    outs = [r["output"] for r in rows]
    scores = [float(r["score"]) for r in rows]
    lens = [len(o) for o in outs]
    return {
        "step": rows[0]["step"],
        "n": len(rows),
        "score": round(statistics.mean(scores), 4),
        "tag_rate": round(sum(1 for o in outs if "<answer>" in o) / len(outs), 4),
        "empty_rate": round(sum(1 for o in outs if len(o.strip()) < 5) / len(outs), 4),
        "bang_only": sum(1 for o in outs if o and set(o.strip()) == {"!"}),
        "mean_chars": round(statistics.mean(lens), 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="+", help="outputs/<exp> dirs containing rollouts/")
    ap.add_argument("--every", type=int, default=5, help="print every Nth step plus first/last 5")
    args = ap.parse_args()
    for run in args.run_dirs:
        roll = Path(run) / "rollouts"
        files = sorted(roll.glob("*.jsonl"), key=lambda p: int(p.stem))
        print(f"== {run}: {len(files)} steps")
        keep = set(range(1, 6)) | {f.stem for f in files[-5:]}
        for f in files:
            s = step_stats(f)
            if s is None:
                continue
            if int(f.stem) % args.every == 0 or f.stem in keep or int(f.stem) <= 5:
                print(f"  step {s['step']:>3}: score={s['score']:.4f} tag={s['tag_rate']:.2f} "
                      f"empty={s['empty_rate']:.3f} bang_only={s['bang_only']} len={s['mean_chars']}")


if __name__ == "__main__":
    main()
