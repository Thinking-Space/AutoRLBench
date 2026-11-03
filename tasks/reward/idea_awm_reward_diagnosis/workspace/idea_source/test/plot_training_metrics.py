#!/usr/bin/env python3
"""Plot every scalar metric recorded by the console logger.

By default this reads the current run's ``train.log`` and writes one PNG per
metric plus a CSV table to the run output directory.  It is deliberately
stand-alone so it can also be used while training is still in progress.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
STEP = re.compile(r"\bstep\s*:\s*(\d+)\s*-\s*(.*)")
PAIR = re.compile(r"(?:^|\s-\s)([^:]+):([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)")


def read_metrics(log_path: Path) -> dict[str, dict[int, float]]:
    values: dict[str, dict[int, float]] = defaultdict(dict)
    with log_path.open(errors="replace") as stream:
        for raw in stream:
            match = STEP.search(ANSI.sub("", raw))
            if not match:
                continue
            step = int(match.group(1))
            for pair in PAIR.finditer(match.group(2)):
                name, number = pair.group(1).strip(), pair.group(2)
                try:
                    value = float(number)
                except ValueError:
                    continue
                if math.isfinite(value):
                    values[name][step] = value
    return dict(values)


def safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._") or "metric"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path,
                        default=Path("/workspace/idea_source/outputs/awm_opsd_qwen3_8b_step715/train.log"))
    parser.add_argument("--out", type=Path,
                        default=Path("/workspace/idea_source/outputs/awm_opsd_qwen3_8b_step715/metric_plots"))
    parser.add_argument("--wandb", action="store_true",
                        help="also upload the parsed metrics to Weights & Biases")
    parser.add_argument("--wandb-project", default="agent_world_model")
    parser.add_argument("--wandb-name", default="opsd_qwen3_8b_step715_metrics")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"),
                        default="online")
    args = parser.parse_args()
    metrics = read_metrics(args.log)
    if not metrics:
        raise SystemExit(f"No step metrics found in {args.log}")
    args.out.mkdir(parents=True, exist_ok=True)

    # A machine-readable copy makes it easy to re-plot or analyze the data.
    steps = sorted({step for series in metrics.values() for step in series})
    with (args.out / "metrics.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        names = sorted(metrics)
        writer.writerow(["step", *names])
        for step in steps:
            writer.writerow([step, *[metrics[name].get(step, "") for name in names]])

    if args.wandb:
        try:
            import wandb
        except ImportError as exc:
            raise SystemExit("--wandb requires the 'wandb' package; REDACTED_IDENTITY it with `pip REDACTED_IDENTITY wandb`") from exc
        run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
            entity=args.wandb_entity,
            dir=str(args.out),
            mode=args.wandb_mode,
            config={"source_log": str(args.log), "metric_count": len(metrics), "step_count": len(steps)},
        )
        for step in steps:
            run.log({name: metrics[name][step] for name in names if step in metrics[name]}, step=step)
        run.summary["metric_count"] = len(metrics)
        run.summary["step_count"] = len(steps)
        run.finish()
        (args.out / "wandb_run.txt").write_text(
            f"project={args.wandb_project}\nname={args.wandb_name}\n"
            f"url={getattr(run, 'url', None)}\nid={getattr(run, 'id', None)}\n"
        )
        print(f"Uploaded metrics to W&B: {getattr(run, 'url', None)}")

    used: dict[str, int] = defaultdict(int)
    for name in sorted(metrics):
        filename = safe_name(name)
        used[filename] += 1
        if used[filename] > 1:
            filename += f"_{used[filename]}"
        series = sorted(metrics[name].items())
        fig, axis = plt.subplots(figsize=(9, 5))
        axis.plot([x for x, _ in series], [y for _, y in series], marker=".", linewidth=1.5)
        axis.set_title(name)
        axis.set_xlabel("training step")
        axis.set_ylabel("value")
        axis.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(args.out / f"{filename}.png", dpi=140)
        plt.close(fig)
    print(f"Parsed {len(metrics)} metrics across {len(steps)} steps")
    print(f"Wrote plots and {args.out / 'metrics.csv'}")


if __name__ == "__main__":
    main()
