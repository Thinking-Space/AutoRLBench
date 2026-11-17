"""Rollout entrypoint: the fully-async producer plus a per-rollout summary.

``generate_rollout`` delegates to ``runtime.fully_async`` unchanged and then
writes ``rollout.<id>.json`` with the statistics a reviewer would pull from a
dashboard: accepted groups, rewards, response lengths, abort reasons and the
token-weighted prefix-cache hit rate reported by the inference engines.
"""
from __future__ import annotations

import json
import os
import statistics
import time
from collections import Counter
from pathlib import Path

from miles.utils.types import Sample

from .fully_async import generate_rollout_fully_async


def _flatten(groups):
    for group in groups:
        for item in group:
            if isinstance(item, list):
                yield from item
            else:
                yield item


def _summary(rollout_id: int, result, elapsed: float) -> dict:
    samples = list(_flatten(result.samples))
    rewards = [float(s.reward) for s in samples if s.reward is not None]
    cached = sum(s.prefix_cache_info.cached_tokens for s in samples)
    prompt = sum(s.prefix_cache_info.total_prompt_tokens for s in samples)
    lengths = [s.response_length for s in samples]
    return {
        "rollout_id": rollout_id,
        "wall_s": round(elapsed, 3),
        "groups": len(result.samples),
        "samples": len(samples),
        "status": dict(Counter(s.status.value if isinstance(s.status, Sample.Status) else str(s.status) for s in samples)),
        "reward_mean": statistics.fmean(rewards) if rewards else None,
        "reward_nonzero": sum(1 for r in rewards if r > 0),
        "response_length_mean": statistics.fmean(lengths) if lengths else None,
        "response_length_max": max(lengths) if lengths else None,
        "engine_prompt_tokens": prompt,
        "engine_cached_tokens": cached,
        "prefix_cache_hit_rate": cached / prompt if prompt else None,
        "abort_reasons": dict(Counter((s.metadata or {}).get("abort_reason") for s in samples if (s.metadata or {}).get("abort_reason"))),
        "metrics": {k: v for k, v in (result.metrics or {}).items() if isinstance(v, (int, float))},
    }


def generate_rollout(args, rollout_id, data_source, evaluation=False):
    started = time.time()
    result = generate_rollout_fully_async(args, rollout_id, data_source, evaluation=evaluation)
    record = _summary(rollout_id, result, time.time() - started)
    root = os.environ.get("RUN_EVIDENCE_DIR")
    if root:
        path = Path(root) / f"rollout.{rollout_id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print("ROLLOUT_SUMMARY " + json.dumps(record, sort_keys=True), flush=True)
    return result


generate_rollout.add_arguments = generate_rollout_fully_async.add_arguments
