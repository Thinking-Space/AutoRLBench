#!/usr/bin/env python3
"""Build the e5 FAISS index for this task's HotpotQA corpus.

The task ships `workspace/shared/data/corpus.jsonl` (12468 passages, 6.6MB) but
not the 37MB `e5_Flat.index` built from it: the index is derivable from the
corpus plus a pinned encoder, so prepare.sh rebuilds it instead of git carrying
a binary that is ten times the size of its own source.

What this reproduces, from `Search-R1/search_r1/search/index_builder.py`
(`Index_Builder.encode_all` / `build_dense_index`, upstream's
`--retrieval_method e5 --faiss_type Flat --pooling_method mean --use_fp16
--max_length 256 --batch_size 512`):

  * the `passage: ` prefix every e5 document gets (index_builder.py:203-204),
    the counterpart of the `query: ` prefix the retrieval server adds
    (retrieval_server.py:81-85) -- swapping or dropping either prefix silently
    degrades every score,
  * mean pooling over the attention mask, then L2 normalisation, so that the
    inner-product index is a cosine index,
  * `faiss.index_factory(dim, "Flat", METRIC_INNER_PRODUCT)`: exact search over
    raw float32 vectors, no training, no quantisation.

Why a separate file instead of calling the workspace's own builder:

  * upstream's `load_model` / `encode_all` call `.cuda()` unconditionally, and
    the prepare stage is not guaranteed a GPU (on the Harbor path it can be a
    build-time layer with no device at all). This one runs on CPU too -- slower,
    same vectors to within float noise.
  * `workspace/` is the agent's editable tree. An asset built by code the agent
    can rewrite is an asset whose quality depends on the agent.
  * upstream pulls the corpus through `datasets` and writes a memmap of the
    embeddings; neither is wanted here.

The rebuilt index is NOT bitwise identical to the one the recorded runs used
(`790cb5bc...`, see data_manifest.json): batch composition, fp16-vs-fp32
accumulation and cuBLAS kernel choice all move the last bits. It is the same
index semantically, and the checks at the end of this file are what "same"
means operationally: right shape, right count, finite, unit-norm.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument(
        "--expect-rows",
        type=int,
        default=12468,
        help="corpus passage count recorded in data_manifest.json",
    )
    parser.add_argument(
        "--expect-dim",
        type=int,
        default=768,
        help="e5-base-v2 hidden size",
    )
    return parser.parse_args()


def read_corpus(path: Path) -> list[str]:
    """`{"id": ..., "contents": "<title>\\n<body>"}` per line, in index order.

    Order is load-bearing twice over: FAISS returns positions, and the
    retrieval server maps a position straight back to `corpus[idx]`
    (retrieval_server.py::load_docs). A sorted or shuffled read would return
    documents that do not match the vectors.
    """
    contents = []
    with path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{number}: not JSON: {exc}") from exc
            if "contents" not in row:
                raise SystemExit(f"{path}:{number}: no 'contents' field")
            contents.append(row["contents"])
    return contents


def main() -> int:
    args = parse_args()

    import faiss
    import numpy as np
    import torch
    from transformers import AutoModel, AutoTokenizer

    passages = read_corpus(args.corpus)
    if len(passages) != args.expect_rows:
        raise SystemExit(
            f"{args.corpus}: {len(passages)} passages, expected {args.expect_rows}; "
            "the shipped corpus does not match data_manifest.json"
        )

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(str(args.model), use_fast=True)
    model = AutoModel.from_pretrained(str(args.model)).eval().to(device)
    # fp16 only on GPU: that is what the recorded index used (`--use_fp16`), and
    # half-precision matmuls on CPU are either unimplemented or glacial.
    if use_cuda:
        model = model.half()
    print(
        f"[build_index] encoding {len(passages)} passages on {device} "
        f"({'fp16' if use_cuda else 'fp32'}), batch {args.batch_size}",
        flush=True,
    )

    embeddings = np.empty((len(passages), args.expect_dim), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(passages), args.batch_size):
            batch = passages[start : start + args.batch_size]
            inputs = tokenizer(
                [f"passage: {text}" for text in batch],
                padding=True,
                truncation=True,
                max_length=args.max_length,
                return_tensors="pt",
            ).to(device)
            hidden = model(**inputs, return_dict=True).last_hidden_state
            mask = inputs["attention_mask"]
            hidden = hidden.masked_fill(~mask[..., None].bool(), 0.0)
            pooled = hidden.sum(dim=1) / mask.sum(dim=1)[..., None]
            pooled = torch.nn.functional.normalize(pooled.float(), dim=-1)
            embeddings[start : start + len(batch)] = pooled.cpu().numpy()
            if start % (args.batch_size * 8) == 0:
                print(f"[build_index]   {start + len(batch)}/{len(passages)}", flush=True)

    if embeddings.shape[1] != args.expect_dim:
        raise SystemExit(
            f"encoder produced dim {embeddings.shape[1]}, expected {args.expect_dim}"
        )

    index = faiss.index_factory(
        embeddings.shape[1], "Flat", faiss.METRIC_INNER_PRODUCT
    )
    index.add(embeddings)

    # Atomic publish: a half-written index file is worse than no index at all,
    # because prepare.sh's own `-s` guard and run.py's asset check would both
    # treat it as done and the retriever would fail much later, in the middle of
    # a graded rerun.
    args.out.parent.mkdir(parents=True, exist_ok=True)
    staging = args.out.with_name(args.out.name + ".partial")
    faiss.write_index(index, str(staging))
    os.replace(staging, args.out)

    # Read the published file back and check the document side only.
    #
    # Deliberately NOT checked here: retrieval quality for real queries. That
    # would need the query encoder, and this file is readable by the agent at
    # /autorlbench/environment -- a gate phrased in terms of query-side scores
    # would be a hint about where to look. The document embeddings are identical
    # on both sides of this task's recorded pair, so document-side invariants
    # are exactly the part that can be asserted without telling the agent
    # anything.
    written = faiss.read_index(str(args.out))
    if written.ntotal != len(passages):
        raise SystemExit(f"index holds {written.ntotal} vectors, expected {len(passages)}")
    if written.d != args.expect_dim:
        raise SystemExit(f"index dim {written.d}, expected {args.expect_dim}")
    sample = np.vstack(
        [
            written.reconstruct(int(i))
            for i in np.linspace(0, written.ntotal - 1, num=64, dtype=int)
        ]
    )
    if not np.isfinite(sample).all():
        raise SystemExit("index contains non-finite vectors")
    norms = np.linalg.norm(sample, axis=1)
    if not ((norms > 0.99) & (norms < 1.01)).all():
        raise SystemExit(
            f"index vectors are not unit-norm: min {norms.min():.4f}, max {norms.max():.4f}"
        )
    print(
        f"[build_index] wrote {args.out} "
        f"({written.ntotal} x {written.d}, {args.out.stat().st_size} bytes)",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
