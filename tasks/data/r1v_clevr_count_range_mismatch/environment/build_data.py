#!/usr/bin/env python3
"""Build the three dataset directories run_grpo.sh and eval_checkpoints.py read.

  train/    CLEVR-CoGenT-A counting questions, 68,000 rows
  val/      2,000 rows held out from the same source (never trained on)
  ood_val/  the first 1,000 Super-CLEVR questions whose answer is an integer

Each is a `datasets.save_to_disk` directory with columns image / problem /
solution. The split is a seed-0 shuffle of the source rows: the first 2,000
become val and the rest train.
"""
import argparse
import json
import shutil
import zipfile
from pathlib import Path

SPLIT_SEED = 0
VAL_N = 2000
OOD_VAL_N = 1000


def build_train_val(raw: Path, out: Path) -> None:
    from datasets import load_dataset

    files = sorted(str(p) for p in (raw / "data").glob("train-*.parquet"))
    assert len(files) == 27, f"expected 27 train parquet shards, found {len(files)}"
    ds = load_dataset("parquet", data_files=files, split="train")
    assert len(ds) == 70000, len(ds)
    ds = ds.shuffle(seed=SPLIT_SEED)
    ds.select(range(VAL_N)).save_to_disk(str(out / "val"))
    ds.select(range(VAL_N, len(ds))).save_to_disk(str(out / "train"))


def build_ood_val(raw: Path, out: Path) -> None:
    from datasets import Dataset, Features, Image, Value

    blob = json.loads((raw / "superCLEVR_questions_30k.json").read_text())
    questions = blob["questions"] if isinstance(blob, dict) else blob
    rows = {"image": [], "problem": [], "solution": []}
    with zipfile.ZipFile(raw / "images.zip") as zf:
        names = {Path(n).name: n for n in zf.namelist()}
        for q in questions:
            ans = q.get("answer", "")
            if not str(ans).strip().isdigit():
                continue
            member = names.get(Path(q["image_filename"]).name)
            if member is None:
                continue
            rows["image"].append({"bytes": zf.read(member), "path": None})
            rows["problem"].append(q["question"])
            rows["solution"].append(str(ans))
            if len(rows["problem"]) >= OOD_VAL_N:
                break
    assert len(rows["problem"]) == OOD_VAL_N, len(rows["problem"])
    feats = Features({"image": Image(), "problem": Value("string"), "solution": Value("string")})
    Dataset.from_dict(rows, features=feats).save_to_disk(str(out / "ood_val"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    for name in ("train", "val", "ood_val"):
        shutil.rmtree(a.out / name, ignore_errors=True)
    a.out.mkdir(parents=True, exist_ok=True)
    build_train_val(a.raw / "clevr_cogent_a", a.out)
    build_ood_val(a.raw / "super_clevr", a.out)
    (a.out / ".complete").write_text("ok\n")


if __name__ == "__main__":
    main()
