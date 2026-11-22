"""Prepare the inventory training and validation inputs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import tempfile
from typing import Any

from inventory_tasks.domain import build_initial_messages


TRAIN = "inventory_train"
RANDOM_SPLIT = "inventory_random_split"
CLEAN = "inventory_clean"
DATA_VERSION = "inventory-split-v1"


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _family(rng: random.Random, code: int) -> dict[str, Any]:
    identity = str(code)
    sku = f"{rng.choice(('BOLT', 'NUT', 'CLIP', 'RING', 'SEAL', 'CAP'))}-{identity}"
    distractor = f"SPARE-{identity}"
    quantity = rng.randint(1, 3)
    return {"parent_scenario_id": f"scenario-{identity}", "order_id": f"ORD-{identity}", "sku": sku,
            "quantity": quantity, "stock": {sku: 2 * quantity + rng.randint(0, 4), distractor: rng.randint(2, 8)},
            "fault_after_commit": rng.randrange(4) == 0}


def _row(family: dict[str, Any], variant: int, split: str, index: int) -> dict[str, Any]:
    parent = family["parent_scenario_id"]
    scenario = {**family, "scenario_id": f"{parent}:variant-{variant}"}
    payload = _json(scenario)
    return {
        "data_source": split, "agent_name": "inventory_agent", "ability": "inventory_state_tracking",
        "prompt": build_initial_messages(scenario, variant),
        "reward_model": {"style": "rule", "ground_truth": payload},
        "extra_info": {"scenario_json": payload, "parent_scenario_id": parent,
                       "scenario_id": scenario["scenario_id"], "variant": variant, "split": split,
                       "occurrence_id": f"{split}:{index}", "index": index},
    }


def build_rows(*, seed: int = 42, families: int = 96, variants: int = 4,
               eval_families: int = 12, train_rows: int = 1280) -> dict[str, list[dict[str, Any]]]:
    if not 1 <= eval_families <= families <= train_rows:
        raise ValueError("Require 1 <= eval_families <= families <= train_rows")
    if not 2 <= variants <= 6:
        raise ValueError("Require 2 to 6 wording variants per scenario")
    rng = random.Random(seed)
    codes = rng.sample(range(100000, 1000000), families + 2 * eval_families)
    scenarios = [_family(rng, code) for code in codes]
    train_families = scenarios[:families]
    clean_families = scenarios[families + eval_families:]

    evaluation_indices = rng.sample(range(families), eval_families)
    withheld = {index: rng.randrange(variants) for index in evaluation_indices}
    validation_variants = list(withheld.values())
    selected, additional = [], []
    for index, family in enumerate(train_families):
        available = [variant for variant in range(variants) if variant != withheld.get(index)]
        rng.shuffle(available)
        selected.append((family, available[0]))
        additional.extend((family, variant) for variant in available[1:])
    rng.shuffle(additional)
    selected.extend(additional)
    complete = selected.copy()
    # Repeat complete shuffled passes to reach the requested update budget.
    while len(selected) < train_rows:
        extra_pass = complete.copy()
        rng.shuffle(extra_pass)
        selected.extend(extra_pass)
    selected = selected[:train_rows]
    rng.shuffle(selected)

    train = [_row(family, variant, TRAIN, i) for i, (family, variant) in enumerate(selected)]
    random_val = [_row(train_families[family_index], withheld[family_index], RANDOM_SPLIT, i)
                  for i, family_index in enumerate(evaluation_indices)]
    clean_val = [_row(family, variant, CLEAN, i)
                 for i, (family, variant) in enumerate(zip(clean_families, validation_variants, strict=True))]
    return {"train": train, "validation": random_val + clean_val}




def _sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()




def prepare(output: str | Path, *, seed: int = 42, families: int = 96, variants: int = 4,
            eval_families: int = 12, train_rows: int = 1280) -> dict[str, Any]:
    """Build the logical inventory input tables in a new directory."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    output = Path(output)
    settings = {"seed": seed, "families": families, "variants": variants,
                "eval_families": eval_families, "train_rows": train_rows}
    manifest_path = output / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        if manifest["version"] != DATA_VERSION or manifest["settings"] != settings:
            raise ValueError(f"Prepared inventory fixture has different settings: {output}")
        for record in manifest["outputs"].values():
            if _sha256(output / record["path"]) != record["sha256"]:
                raise ValueError(f"Prepared inventory file changed: {record['path']}")
        return manifest
    if output.exists():
        raise FileExistsError(f"Cannot replace an incomplete inventory fixture: {output}")
    rows = build_rows(**settings)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".inventory-build-", dir=output.parent) as temporary:
        stage = Path(temporary)
        manifest = {"version": DATA_VERSION, "settings": settings, "outputs": {}}
        for split, stem in (("train", "train"), ("validation", "val")):
            entries = rows[split]
            parquet = stage / f"{stem}.parquet"
            pq.write_table(pa.Table.from_pylist(entries), parquet, compression="zstd")
            parquet.chmod(0o444)
            jsonl = stage / f"{stem}.jsonl"
            with jsonl.open("w", encoding="utf-8") as handle:
                for row in entries:
                    handle.write(_json(row) + "\n")
            for name, path in ((stem, parquet), (f"{stem}_jsonl", jsonl)):
                manifest["outputs"][name] = {"path": path.name, "rows": len(entries), "sha256": _sha256(path)}
        (stage / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                                               encoding="utf-8")
        stage.rename(output)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--families", type=int, default=96)
    parser.add_argument("--variants", type=int, default=4)
    parser.add_argument("--eval-families", type=int, default=12)
    parser.add_argument("--train-rows", type=int, default=1280)
    args = vars(parser.parse_args())
    output = args.pop("output")
    print(json.dumps(prepare(output, **args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
