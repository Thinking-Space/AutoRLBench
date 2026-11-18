"""Build a mixed query pool from the per-source query banks.

A pool is one JSONL file of queries plus a manifest describing what went into it. The caller
gives a total and a weight per source; the builder allocates integer counts with the largest
remainder, drops instances listed in the unusable-image registry, and writes both files.

Every source of SOURCE_ORDER appears in the manifest's `source_counts`, including sources whose
weight is zero: the manifest describes the requested mix, and a zero is part of that mix.

usage: build_pool.py --banks <dir> --out <pool dir> --total N --weights s=1,r=1,g=1 [--seed 42]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

SOURCE_ORDER = ("swe_smith", "swe_rebench", "r2e_gym")
TOOLS = ("execute_bash", "think", "complete_plan", "finish")


def parse_weights(spec: str) -> dict[str, float]:
    weights: dict[str, float] = {}
    for item in spec.split(","):
        name, separator, raw = item.strip().partition("=")
        if not separator:
            raise ValueError(f"invalid weight {item!r}; expected name=value")
        name = name.strip()
        if name not in SOURCE_ORDER:
            raise ValueError(f"unknown source {name!r}; expected one of {SOURCE_ORDER}")
        weight = float(raw)
        if weight < 0:
            raise ValueError(f"weight must be non-negative: {item!r}")
        weights[name] = weight
    missing = set(SOURCE_ORDER) - set(weights)
    if missing:
        raise ValueError(f"missing weights for {sorted(missing)}")
    if sum(weights.values()) <= 0:
        raise ValueError("at least one weight must be positive")
    return weights


def allocate_counts(total: int, weights: dict[str, float]) -> dict[str, int]:
    """Largest-remainder allocation, in SOURCE_ORDER, so the counts sum to `total` exactly."""
    weight_sum = sum(weights.values())
    exact = {name: total * weights[name] / weight_sum for name in SOURCE_ORDER}
    counts = {name: int(value) for name, value in exact.items()}
    remainder = total - sum(counts.values())
    order = sorted(SOURCE_ORDER, key=lambda name: (-(exact[name] - counts[name]), SOURCE_ORDER.index(name)))
    for name in order[:remainder]:
        counts[name] += 1
    return counts


def load_bank(banks: Path, source: str) -> list[dict[str, Any]]:
    path = banks / f"{source}.jsonl"
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def unusable(registry: Path) -> set[tuple[str, str]]:
    if not registry or not registry.exists():
        return set()
    payload = json.loads(registry.read_text(encoding="utf-8"))
    return {(item["source"], item["instance_id"]) for item in payload.get("selected", [])}


def build(args: argparse.Namespace) -> dict[str, Any]:
    weights = parse_weights(args.weights)
    counts = allocate_counts(args.total, weights)
    skip = unusable(Path(args.registry)) if args.registry else set()
    rng = random.Random(args.seed)

    rows: list[dict[str, Any]] = []
    excluded_counts = {source: 0 for source in SOURCE_ORDER}
    for source in SOURCE_ORDER:
        bank = load_bank(Path(args.banks), source)
        eligible = []
        for row in bank:
            if (source, row["instance_id"]) in skip:
                excluded_counts[source] += 1
                continue
            eligible.append(row)
        if len(eligible) < counts[source]:
            raise SystemExit(f"{source}: need {counts[source]} queries, only {len(eligible)} eligible")
        chosen = rng.sample(sorted(eligible, key=lambda row: row["instance_id"]), counts[source])
        for rank, row in enumerate(sorted(chosen, key=lambda row: row["instance_id"])):
            rows.append({
                "prompt": row["problem_statement"],
                "tools": [{"type": "function", "function": {"name": name}} for name in TOOLS],
                "metadata": {
                    "pool_source": source,
                    "instance_id": row["instance_id"],
                    "repo": row["repo"],
                    "rank": rank,
                },
            })

    rng.shuffle(rows)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    data_path = out / "queries.jsonl"
    with data_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    manifest = {
        "version": 2,
        "seed": args.seed,
        "total_queries": args.total,
        "source_weights": weights,
        "source_counts": counts,
        "excluded_instance_counts": excluded_counts,
        "registry": Path(args.registry).name if args.registry else None,
        "output": data_path.name,
        "sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--banks", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--total", type=int, required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--registry", default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    manifest = build(args)
    print(json.dumps({k: manifest[k] for k in ("total_queries", "source_counts", "sha256")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
