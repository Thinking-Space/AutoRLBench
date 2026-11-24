#!/usr/bin/env python3
"""Build the prompt subset that the teacher-replay arm trains on.

The replay looks trajectories up by exact prompt token ids.  If the trainer's
dataloader draws a prompt that was never rolled out, the replay raises -- by
design, because the alternative is an on-policy arm wearing an off-policy
label.  Pointing ``data.train_files`` at exactly the rolled-out prompts removes
that failure mode and, as a side effect, makes "one epoch, no repeats" a
structural property rather than an arithmetic coincidence.

The subset is derived from the rollout output rather than by re-running the
selection with the same seed.  The selection is deterministic (verified), but
computing it twice is two chances to diverge.

The real work here is the check at the end: re-tokenizing each kept prompt the
way verl will and confirming it reproduces the ids stored in the trajectory
file.  A chat-template or tokenizer-flag mismatch would otherwise surface as a
100% cache miss on training step 1, after the queue wait.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trajectories", required=True, help="rollout output parquet")
    ap.add_argument("--source", required=True, help="the full prompt parquet the rollout drew from")
    ap.add_argument("--out", required=True, help="subset parquet for data.train_files")
    ap.add_argument("--tokenizer", required=True, help="model dir, for the template check")
    args = ap.parse_args()

    import pyarrow as pa
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer

    traj = pq.read_table(args.trajectories)
    traj_cols = traj.to_pydict()
    wanted = {}
    for eid, pids in zip(traj_cols["example_id"], traj_cols["prompt_token_ids"]):
        if eid in wanted:
            raise SystemExit(f"example_id {eid} appears twice in {args.trajectories}")
        wanted[eid] = list(pids)
    print(f"[subset] {len(wanted)} trajectories in {args.trajectories}")

    source = pq.read_table(args.source)
    ids = source.column("example_id").to_pylist()
    keep = [i for i, eid in enumerate(ids) if eid in wanted]
    if len(keep) != len(wanted):
        missing = set(wanted) - {ids[i] for i in keep}
        raise SystemExit(
            f"{len(missing)} rolled-out example_ids are absent from {args.source}; "
            "the trajectories and the prompt file disagree"
        )
    subset = source.take(keep)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    sub_cols = subset.to_pydict()
    mismatches = []
    for eid, messages in zip(sub_cols["example_id"], sub_cols["prompt"]):
        text = tokenizer.apply_chat_template(
            [dict(m) for m in messages], tokenize=False, add_generation_prompt=True
        )
        got = tokenizer(text, add_special_tokens=False)["input_ids"]
        if list(got) != wanted[eid]:
            mismatches.append(eid)
    if mismatches:
        raise SystemExit(
            f"{len(mismatches)}/{subset.num_rows} prompts do not re-tokenize to the ids stored "
            f"in the trajectory file (first: {mismatches[0]}).  The replay matches on exact "
            "token ids, so this would be a total cache miss at training step 1."
        )
    print(f"[subset] all {subset.num_rows} prompts re-tokenize to the cached ids")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(subset, out)

    lens = traj_cols["response_tokens"]
    trunc = sum(traj_cols["truncated"])
    summary = {
        "prompts": subset.num_rows,
        "source": args.source,
        "trajectories": args.trajectories,
        "mean_response_tokens": sum(lens) / len(lens),
        "median_response_tokens": sorted(lens)[len(lens) // 2],
        "p99_response_tokens": sorted(lens)[int(0.99 * len(lens))],
        "max_response_tokens": max(lens),
        "truncated": trunc,
        "truncated_fraction": trunc / len(lens),
    }
    out.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2))
    print("[subset]", json.dumps(summary))
    print(f"[subset] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
