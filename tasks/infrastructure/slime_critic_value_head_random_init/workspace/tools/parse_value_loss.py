#!/usr/bin/env python3
"""Extract the critic's warmup value loss from a slime training log.

slime reports per-optimizer-step training metrics only on stdout, via
`slime/backends/megatron_utils/model.py`:

    logger.info(f"{role_tag}step {accumulated_step_id}: {log_dict}")

so for the critic the lines look like

    critic-step 0: {'train/critic-value_loss': 5.91, 'train/critic-value_clipfrac': 0.0, ...}

Rollout-level metrics are emitted the same way but keyed by rollout id rather
than optimizer step, with a label instead of `step N`:

    rollout 0: {'rollout/rewards': 0.125, 'rollout/round_number': 4.0, ...}
    perf 0:    {'rollout/response_len/mean': 244.4, ...}

Which of those appear depends on the stage, which is why both are parsed.
`rollout N:` comes from the training actor and is only logged on rollouts where
the actor trains (slime calls it inside `train_actor`), so a critic-only warmup
stage has none. `perf N:` is logged by the RolloutManager on every rollout, so
it is the stage-independent evidence that episodes were really generated.

There is no jsonl or csv sink for any of these, hence this parser.

Usage:
    python3 tools/parse_value_loss.py train.log --warmup-steps 2 --out summary.json
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys

STEP_RE = re.compile(r"(?:^|[\s\]])(?P<role>[A-Za-z_][A-Za-z0-9_]*-)?step\s+(?P<step>\d+):\s*(?P<payload>\{.*\})\s*$")

# `<label> <id>: {...}` for any label that is not a `step` line, i.e. the
# rollout-level metrics. Anchored on " - " so it matches the logger's
# "<file>.py:<line> - " prefix and not arbitrary prose.
ROLLOUT_RE = re.compile(r" - (?P<label>[A-Za-z_][A-Za-z0-9_]*)\s+(?P<id>\d+):\s*(?P<payload>\{.*\})\s*$")


def _coerce(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _payload(text: str) -> dict | None:
    try:
        payload = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return None
    return payload if isinstance(payload, dict) else None


def parse(paths: list[str]) -> list[dict]:
    """Return one record per metric line, tagged `step` or `rollout`."""
    records: list[dict] = []
    for path in paths:
        with open(path, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.rstrip("\n")
                m = STEP_RE.search(line)
                if m:
                    payload = _payload(m.group("payload"))
                    if payload is None:
                        continue
                    role = (m.group("role") or "").rstrip("-") or "actor"
                    records.append(
                        {"kind": "step", "role": role, "step": int(m.group("step")), "metrics": payload}
                    )
                    continue
                m = ROLLOUT_RE.search(line)
                if m:
                    payload = _payload(m.group("payload"))
                    if payload is None:
                        continue
                    records.append(
                        {
                            "kind": "rollout",
                            "label": m.group("label"),
                            "step": int(m.group("id")),
                            "metrics": payload,
                        }
                    )
    return records


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+")
    ap.add_argument(
        "--warmup-steps",
        type=int,
        default=0,
        help="number of leading critic optimizer steps to average over (0 = all critic steps)",
    )
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    records = parse(args.logs)

    critic_steps = []
    for rec in records:
        if rec["kind"] != "step":
            continue
        hit = next((k for k in rec["metrics"] if k.endswith("value_loss")), None)
        if hit is None:
            continue
        value = _coerce(rec["metrics"][hit])
        if value is None:
            continue
        critic_steps.append(
            {
                "step": rec["step"],
                "value_loss": value,
                "value_clipfrac": _coerce(
                    next((rec["metrics"][k] for k in rec["metrics"] if k.endswith("value_clipfrac")), None)
                ),
            }
        )
    critic_steps.sort(key=lambda r: r["step"])

    rewards = []
    for rec in records:
        for key, value in rec["metrics"].items():
            if "reward" in key and not key.endswith(("_std", "_max", "_min")):
                coerced = _coerce(value)
                if coerced is not None:
                    rewards.append({"step": rec["step"], "key": key, "value": coerced})

    # Evidence that episodes were really generated in this stage, independent of
    # whether the actor trained: the RolloutManager reports response-length
    # statistics once per rollout. A stage that generated nothing has neither.
    #
    # Only `rollout/response_len/mean` (the RolloutManager's `perf N:` line) is
    # averaged. slime also emits `rollout/response_lengths` from the actor side,
    # but it is a *different* quantity -- it counts the tokens the trainer sees,
    # including the observation turns fed back into the context, so it runs ~70%
    # higher. Averaging the two together produced a blended figure that meant
    # nothing and was not comparable between stages, since the actor-side metric
    # only exists on rollouts where the actor trains.
    rollout_ids = {rec["step"] for rec in records if rec["kind"] == "rollout"}
    response_lens = [
        _coerce(value)
        for rec in records
        for key, value in rec["metrics"].items()
        if key.endswith("response_len/mean")
    ]
    response_lens = [v for v in response_lens if v is not None]

    window = critic_steps[: args.warmup_steps] if args.warmup_steps > 0 else critic_steps
    losses = [r["value_loss"] for r in window]

    summary = {
        "n_critic_steps": len(critic_steps),
        "rollouts_generated": len(rollout_ids),
        "mean_response_len": (sum(response_lens) / len(response_lens)) if response_lens else None,
        "warmup_window": len(window),
        "critic_steps": critic_steps,
        "first_value_loss": critic_steps[0]["value_loss"] if critic_steps else None,
        "warmup_value_loss_mean": (sum(losses) / len(losses)) if losses else None,
        "warmup_value_loss_max": max(losses) if losses else None,
        "warmup_value_loss_min": min(losses) if losses else None,
        "rewards": rewards[:50],
        "mean_reward": (sum(r["value"] for r in rewards) / len(rewards)) if rewards else None,
    }

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)

    print(f"critic optimizer steps found: {summary['n_critic_steps']}")
    if not critic_steps:
        print("NO value_loss METRICS FOUND -- the critic never took an optimizer step.", file=sys.stderr)
        print("Check that --advantage-estimator ppo is set and that training reached the critic.", file=sys.stderr)
        return 2
    print(f"{'step':>6}  {'value_loss':>14}  {'value_clipfrac':>14}")
    for rec in critic_steps:
        clip = "n/a" if rec["value_clipfrac"] is None else f"{rec['value_clipfrac']:.4f}"
        print(f"{rec['step']:>6}  {rec['value_loss']:>14.6f}  {clip:>14}")
    print(f"\nwarmup window = first {summary['warmup_window']} critic step(s)")
    print(f"warmup value_loss mean = {summary['warmup_value_loss_mean']:.6f}")
    print(f"rollouts generated = {summary['rollouts_generated']}", end="")
    if summary["mean_response_len"] is not None:
        print(f" (mean response {summary['mean_response_len']:.1f} tokens)", end="")
    print()
    if summary["mean_reward"] is not None:
        print(f"mean reward (episode pass rate) ~= {summary['mean_reward']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
