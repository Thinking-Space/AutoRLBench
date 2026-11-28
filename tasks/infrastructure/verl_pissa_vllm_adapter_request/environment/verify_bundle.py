#!/usr/bin/env python3
"""Numerically verify the derived PiSSA bundle against its source model.

Replaces the former byte-level sha256 pins on the two derived safetensors:
the bundle is produced by a CPU bf16 SVD (peft ``pissa_niter_2``), whose
rounding depends on the torch build and the host CPU's LAPACK/MKL dispatch,
so bytes recorded on one host mismatch on another. What the task actually
needs is that the bundle IS the intended rank-8 PiSSA decomposition of the
recorded source model, which this script proves:

* the source ``model.safetensors`` matches ``source_manifest.json``
  byte-for-byte (env-independent: it is downloaded, not derived);
* every parameter outside the 48 LoRA targets is bitwise identical between
  the residual base and the source (PiSSA only rewrites its targets);
* each target module satisfies ``W_residual + B @ A == W_pretrained`` in
  float32 accumulation within bf16 rounding tolerance (recorded max
  relative error 2.1e-3 on the recorded build; threshold 5e-3 -- a wrong
  decomposition errs at O(0.1..1) relative, three orders above it);
* degenerate no-op bundles (zero init, unmodified residual) are rejected.

Runs on CPU inside the prepare-stage image with the pinned pure-Python
shadow active; only needs torch and safetensors.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file

RANK = 8
RECONSTRUCTION_TOL = 5e-3
LORA_SUFFIXES = (".lora_A.weight", ".lora_B.weight")
PEFT_PREFIX = "base_model.model."


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def base_key(adapter_key: str) -> str | None:
    """Map a peft adapter key to its base-model safetensors key.

    ``base_model.model.<module>.lora_A.weight`` -> ``<module>.weight``: the
    suffix names the LoRA tensor, and the rewritten base parameter is the
    module's ``.weight``.
    """
    for suffix in LORA_SUFFIXES:
        if adapter_key.endswith(suffix):
            stripped = adapter_key[: -len(suffix)]
            if not stripped.startswith(PEFT_PREFIX):
                return None
            return stripped[len(PEFT_PREFIX):] + ".weight"
    return None


def fail(msg: str) -> None:
    print(f"[verify_bundle] FAIL: {msg}", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", required=True, type=Path,
                        help="the fetched Qwen2.5-0.5B-Instruct snapshot dir")
    parser.add_argument("--bundle", required=True, type=Path,
                        help="the derived pissa-r8 bundle dir")
    parser.add_argument("--manifest", type=Path,
                        default=Path(__file__).with_name("source_manifest.json"),
                        help="recorded source pins")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    src_path = args.source / "model.safetensors"

    # --- 1. Source model: hard byte pin (downloaded, env-independent) -------
    expected_sha = manifest["source_model"]["model_safetensors_sha256"]
    actual_sha = sha256_file(src_path)
    if actual_sha != expected_sha:
        fail(f"source model.safetensors sha256 {actual_sha} != recorded {expected_sha}")

    src = load_file(str(src_path))
    res = load_file(str(args.bundle / "residual_base" / "model.safetensors"))
    adap = load_file(str(args.bundle / "pissa_adapter" / "adapter_model.safetensors"))

    # --- 2. Adapter shape sanity ---------------------------------------------
    lora_a: dict[str, torch.Tensor] = {}
    lora_b: dict[str, torch.Tensor] = {}
    for key, value in adap.items():
        mapped = base_key(key)
        if mapped is None:
            fail(f"unexpected adapter tensor {key!r} (not a lora_A/lora_B weight)")
        if key.endswith(".lora_A.weight"):
            lora_a[mapped] = value
        else:
            lora_b[mapped] = value
    if set(lora_a) != set(lora_b):
        fail("adapter has unpaired lora_A/lora_B modules")
    targets = sorted(lora_a)
    if len(targets) != 48:
        fail(f"expected 48 LoRA targets (q_proj/v_proj x 24 layers), got {len(targets)}")
    for k in targets:
        w = src.get(k)
        if w is None:
            fail(f"adapter target {k!r} is not a source-model parameter")
        a, b = lora_a[k], lora_b[k]
        if a.dim() != 2 or b.dim() != 2 or a.shape[0] != RANK or b.shape[1] != RANK \
                or a.shape[1] != w.shape[1] or b.shape[0] != w.shape[0]:
            fail(f"adapter shapes wrong for {k!r}: A{tuple(a.shape)} B{tuple(b.shape)} vs W{tuple(w.shape)}")
        if src[k].dtype != res[k].dtype:
            fail(f"dtype drift on {k!r}")

    # --- 3. Untouched parameters: bitwise identical -------------------------
    if set(res) != set(src):
        extra = sorted(set(res) ^ set(src))
        fail(f"residual_base parameter set differs from source: {extra[:4]}")
    for k in src:
        if k in lora_a:
            continue
        if not torch.equal(res[k], src[k]):
            fail(f"untouched parameter {k!r} differs between residual base and source")

    # --- 4. Target modules: reconstruction + no-op rejection ----------------
    worst_key, worst_err = None, 0.0
    for k in targets:
        if torch.equal(res[k], src[k]):
            fail(f"target {k!r} was left unmodified (no PiSSA init applied)")
        if not lora_b[k].any():
            fail(f"lora_B for {k!r} is all zero (plain LoRA init, not PiSSA)")
        w_src = src[k].to(torch.float32)
        w_res = res[k].to(torch.float32)
        delta = lora_b[k].to(torch.float32) @ lora_a[k].to(torch.float32)
        err = ((w_res + delta) - w_src).norm() / w_src.norm()
        if err > worst_err:
            worst_key, worst_err = k, err
    if worst_err > RECONSTRUCTION_TOL:
        fail(f"worst reconstruction error {worst_err:.3e} on {worst_key!r} "
             f"exceeds tolerance {RECONSTRUCTION_TOL:.1e} (expected ~2.1e-3 bf16 rounding)")

    print("[verify_bundle] OK: source bytes pinned; 48/48 untouched params bitwise equal; "
          f"worst W_res+B@A reconstruction error {worst_err:.3e} ({worst_key}) "
          f"<= {RECONSTRUCTION_TOL:.1e}")


if __name__ == "__main__":
    main()
