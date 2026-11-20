#!/usr/bin/env python3
"""Sample teacher trajectories for the off-policy (teacher-forced) QAT arm.

This produces the fixed dataset that the LLM-QAT-style arm trains on.  The
point of that arm is to change exactly one thing relative to the on-policy OPD
run -- where the tokens come from -- so every sampling knob here is pinned to
the value the OPD rollout uses, not to an eval-style default:

    temperature 1.0, top_p 1.0, top-k disabled, n=1

(``scripts/preflight_mopd.py`` defaults: ``--temperature 1.0``,
``--sampling-top-p 1.0``, ``--sampling-top-k 0``, which vLLM reads as disabled.)

The teacher is the unquantized model, so the cache dtype is ``auto``.  That
mirrors LLM-QAT, which generates its data from the full-precision model.

Token ids and the teacher's log-prob of each sampled token are the payload.

The log-prob matters for more than bookkeeping.  ``ray_trainer.py`` sets
``old_log_probs`` from the *rollout engine*, deliberately, so that the ratio
inside the policy loss is measured against the policy that actually drew the
tokens ("recomputing with the HF actor would silently turn this into a
different behavior policy").  Feeding the teacher's log-probs in that slot
makes ``exp(log_prob - old_log_probs)`` the exact importance weight
pi_student/pi_teacher, with PPO clipping already in place to bound its
variance -- so the off-policy arm reuses the on-policy gradient path unchanged.

Only the *sampled* token's log-prob is stored.  The full top-512 the reward
needs would be 9600 x ~7400 x 512 x 8 B ~ 291 TB; one scalar per token is
284 MB.  Both top-k distributions are recomputed at training time by the
forward passes that the OPD arm already performs.

Text is stored only for eyeballing: re-tokenizing it is not guaranteed to
reproduce the ids, and the trainer has to consume exactly the ids the model
emitted.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import time
from typing import Any


def _chat_prompt(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def _load_prompts(args: argparse.Namespace, tokenizer: Any) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    table = pq.read_table(args.data)
    cols = table.to_pydict()

    records = []
    too_long = 0
    for i in range(table.num_rows):
        messages = [dict(m) for m in cols["prompt"][i]]
        text = _chat_prompt(tokenizer, messages)
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        # The training run filters on the same budget (PROMPT_LENGTH=512 in
        # slurm_qat_train.sbatch, which drops 0.31% of the pool).  Applying it
        # here keeps the prompt-length distribution of the two arms identical.
        if len(ids) > args.max_prompt_tokens:
            too_long += 1
            continue
        reward = cols["reward_model"][i] or {}
        records.append(
            {
                "example_id": cols["example_id"][i],
                "data_source": cols["data_source"][i],
                "ground_truth": reward.get("ground_truth"),
                "prompt_text": text,
                "prompt_token_ids": ids,
            }
        )

    # Deterministic subset: same prompt seed -> same example_ids, so the arm is
    # reproducible and the exact prompt set gets recorded in the metadata.
    random.Random(args.prompt_seed).shuffle(records)
    if args.count > 0:
        records = records[: args.count]

    print(
        f"[rollout] pool={table.num_rows} kept={len(records)} "
        f"dropped_over_{args.max_prompt_tokens}_tokens={too_long}",
        flush=True,
    )
    return records


def _sampled_logprobs(completion: Any, ids: list[int]) -> list[float]:
    """Pull out the log-prob vLLM assigned to each token it actually sampled.

    ``completion.logprobs[t]`` is a dict keyed by token id.  Silently padding a
    missing entry would put a wrong old_log_prob into the importance ratio, so
    anything unexpected raises instead.
    """
    per_token = getattr(completion, "logprobs", None)
    if per_token is None or len(per_token) != len(ids):
        raise RuntimeError(
            f"vLLM returned {0 if per_token is None else len(per_token)} logprob "
            f"entries for {len(ids)} sampled tokens; SamplingParams(logprobs=0) "
            "did not take effect"
        )
    out = []
    for tok, entry in zip(ids, per_token):
        item = entry.get(tok)
        if item is None:
            raise RuntimeError(f"no logprob returned for sampled token id {tok}")
        out.append(float(getattr(item, "logprob", item)))
    return out


def _shard(records: list[dict[str, Any]], index: int, total: int) -> list[dict[str, Any]]:
    if total <= 1:
        return records
    return [r for i, r in enumerate(records) if i % total == index]


def _make_llm(args: argparse.Namespace):
    from vllm import LLM

    return LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype="bfloat16",
        # "auto" is the teacher (unquantized), which is what LLM-QAT generates
        # from.  A turboquant/kivi dtype makes this an *on-policy* rollout: the
        # student sampling through the cache it will actually be deployed with.
        kv_cache_dtype=args.kv_cache_dtype,
        max_model_len=args.max_prompt_tokens + args.max_tokens,
        max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True,
        enable_prefix_caching=False,
        trust_remote_code=True,
        seed=args.seed,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="teacher checkpoint (unquantized)")
    ap.add_argument("--data", required=True, help="prompt parquet, the file the OPD run trains on")
    ap.add_argument("--out", required=True, help="output parquet")
    ap.add_argument("--count", type=int, default=9600,
                    help="trajectories to keep; 150 steps x global batch 64")
    ap.add_argument("--max-prompt-tokens", type=int, default=512)
    ap.add_argument("--max-tokens", type=int, default=16384)
    # Pinned to the OPD rollout, not to eval defaults.  Changing any of these
    # breaks the single-variable comparison this arm exists for.
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=-1, help="-1 disables top-k, as in the OPD rollout")
    ap.add_argument("--kv-cache-dtype", default="auto",
                    help="auto for the FP teacher; e.g. turboquant_k2v3_nc for a student rollout")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--prompt-seed", type=int, default=0)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--shard-count", type=int, default=1)
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument("--max-num-seqs", type=int, default=64)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from vllm import SamplingParams
    import pyarrow as pa
    import pyarrow.parquet as pq

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    records = _shard(_load_prompts(args, tokenizer), args.shard_index, args.shard_count)
    print(f"[rollout] shard {args.shard_index}/{args.shard_count}: {len(records)} prompts", flush=True)

    llm = _make_llm(args)
    sampling = SamplingParams(
        n=1,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
        seed=args.seed,
        # 0 -> return the log-prob of the sampled token only.  This becomes
        # old_log_probs on the training side; see the module docstring.
        logprobs=0,
    )

    started = time.time()
    outputs = llm.generate(
        [{"prompt_token_ids": r["prompt_token_ids"]} for r in records],
        sampling,
    )
    elapsed = time.time() - started

    columns = (
        "example_id", "data_source", "ground_truth", "prompt_text",
        "prompt_token_ids", "response_token_ids", "response_logprobs",
        "response_text", "prompt_tokens", "response_tokens",
        "finish_reason", "truncated",
    )
    rows: dict[str, list[Any]] = {k: [] for k in columns}
    truncated = 0
    for rec, out in zip(records, outputs):
        completion = out.outputs[0]
        ids = list(completion.token_ids)
        logprobs = _sampled_logprobs(completion, ids)
        finish = str(getattr(completion, "finish_reason", None))
        is_trunc = finish.lower() == "length"
        truncated += int(is_trunc)
        rows["example_id"].append(rec["example_id"])
        rows["data_source"].append(rec["data_source"])
        rows["ground_truth"].append(rec["ground_truth"])
        rows["prompt_text"].append(rec["prompt_text"])
        rows["prompt_token_ids"].append(rec["prompt_token_ids"])
        rows["response_token_ids"].append(ids)
        rows["response_logprobs"].append(logprobs)
        rows["response_text"].append(completion.text)
        rows["prompt_tokens"].append(len(rec["prompt_token_ids"]))
        rows["response_tokens"].append(len(ids))
        rows["finish_reason"].append(finish)
        rows["truncated"].append(is_trunc)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(rows), out_path)

    lens = rows["response_tokens"]
    meta = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "data": args.data,
        "kv_cache_dtype": args.kv_cache_dtype,
        "stores_sampled_logprobs": True,
        "sampling": {
            "n": 1,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "max_tokens": args.max_tokens,
            "seed": args.seed,
        },
        "prompt_seed": args.prompt_seed,
        "max_prompt_tokens": args.max_prompt_tokens,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "trajectories": len(lens),
        "truncated": truncated,
        "truncated_fraction": truncated / max(len(lens), 1),
        "mean_response_tokens": sum(lens) / max(len(lens), 1),
        "max_response_tokens": max(lens) if lens else 0,
        "elapsed_seconds": elapsed,
        "example_ids": rows["example_id"],
    }
    out_path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
    print(
        f"[rollout] wrote {len(lens)} trajectories to {out_path} in {elapsed:.0f}s "
        f"(mean {meta['mean_response_tokens']:.0f} tokens, "
        f"truncated {meta['truncated_fraction'] * 100:.2f}%)",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
