"""Budget the BF16 staging arena before vLLM decides how big the KV cache is.

The problem
-----------
``_ensure_staging`` allocates on the first forward that sees a real ``kv_cache``,
which is *after* vLLM's memory-profiling pass has already sized the KV cache from
whatever was free.  The arena is therefore invisible to the profiler, and at 160
seats it is 1.45 GiB on the 1.5B and 7.4 GiB on the 4B.

Today every launcher absorbs that with a hand-picked ``GPU_MEMORY_UTILIZATION``
(0.90 for the 1.5B, 0.85 for the 4B).  Those constants are correct only for the
``max_num_seqs`` they were chosen at, they are not derived from anything, and
nothing checks them -- change the concurrency and the run OOMs after loading the
model, minutes into the job, for a reason the traceback does not name.

What this does
--------------
Computes the arena from the model config and the engine's own settings, using the
same arithmetic ``_ensure_staging`` uses, and subtracts it from
``gpu_memory_utilization`` so the profiler leaves room.  The subtraction is the
point: a warning that the operator has to act on is the same magic constant one
step removed.

    rows  = max_seqs x (sink_pages + tail_pages)
    bytes = rows x block_size x n_kv_heads x head_dim x 2 (bf16) x 2 (K,V) x n_layers

Only applies when per-request staging is on.  The shared arena is sized from
``staging_tokens`` and does not scale with concurrency, which is precisely why it
was small enough to go unnoticed and also why it did not work.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

__all__ = ["arena_bytes", "adjust_utilization", "total_device_memory"]


def total_device_memory(device_index: int = 0) -> int:
    """Total bytes on the GPU, WITHOUT initializing a CUDA context.

    This function is why `torch.cuda.get_device_properties` is not used here,
    and the reason is not stylistic.  `get_device_properties` calls
    `torch.cuda._lazy_init()`, so merely asking how big the GPU is creates a
    CUDA context in the parent process.  vLLM then refuses to fork:

        WARNING [system_utils.py:157] We must use the `spawn` multiprocessing
        start method. Overriding VLLM_WORKER_MULTIPROC_METHOD to 'spawn'.
        Reasons: CUDA is initialized

    Under `spawn` the EngineCore worker is a fresh interpreter, so every
    in-process monkeypatch this repo relies on -- per-request BF16 staging, the
    GQA decode kernel, the cudagraph fix, the coverage probe -- is applied only
    in the parent and the worker silently runs stock vLLM.  The patches still
    print their "active" banners, because the parent really did apply them.

    That is not hypothetical.  Every OSCAR eval that called this budget helper
    (jobs 57947, 58089-58090, 58204-58205, 58236-58239, 58364-58365, 58375-58376,
    58459-58462, 58470-58472) ran the stock backend, which is why their logs JIT
    `_oscar_window_bounds_kernel` rather than `_oscar_window_bounds_kernel_pr`.
    Runs that did NOT call it (the per-request staging arms) forked and got the
    patches.  The measurement that separated "eager" from "compiled" was
    therefore also a measurement of "stock arena" against itself -- valid, but
    it was never comparing the code we thought we were testing.

    NVML answers the same question and creates no context; see the test in the
    module docstring of opqad/integrations/vllm_plugin.py for the belt-and-braces half of
    this fix.
    """
    # CUDA_VISIBLE_DEVICES indexes into NVML's enumeration, so a bare
    # nvmlDeviceGetHandleByIndex(device_index) would read the wrong card on a
    # shared node.  Map through it when it is set.
    vis = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    phys: "int | str" = device_index
    if vis:
        entries = [e.strip() for e in vis.split(",") if e.strip()]
        if device_index < len(entries):
            e = entries[device_index]
            phys = int(e) if e.isdigit() else e  # may be a GPU-<uuid>
    try:
        import pynvml

        pynvml.nvmlInit()
        try:
            if isinstance(phys, int):
                h = pynvml.nvmlDeviceGetHandleByIndex(phys)
            else:
                h = pynvml.nvmlDeviceGetHandleByUUID(phys.encode())
            return int(pynvml.nvmlDeviceGetMemoryInfo(h).total)
        finally:
            pynvml.nvmlShutdown()
    except Exception:  # noqa: BLE001
        pass

    # Fallback: nvidia-smi.  Slower, but it keeps this working on a node where
    # nvidia-ml-py is missing rather than falling back to a CUDA context.
    import subprocess

    q = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.total",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=30, check=True,
    )
    mibs = [int(x) for x in q.stdout.split() if x.isdigit()]
    if not mibs:
        raise RuntimeError("nvidia-smi returned no memory.total rows")
    idx = phys if isinstance(phys, int) and phys < len(mibs) else 0
    return mibs[idx] * 2**20


def _read_config(model_path: str) -> dict:
    p = Path(model_path) / "config.json"
    with open(p) as f:
        cfg = json.load(f)
    # Some checkpoints nest the language-model config.
    return cfg.get("text_config", cfg)


def arena_bytes(
    model_path: str,
    max_num_seqs: int,
    *,
    sink_tokens: int = 64,
    recent_tokens: int = 256,
    block_size: int = 16,
) -> tuple[int, dict]:
    """Bytes the per-request staging arena will allocate, plus the terms.

    Returns ``(total_bytes, detail)``.  The detail dict is returned rather than
    logged from here so the caller can print it next to its own banner; the
    arithmetic is the part worth showing, because a number on its own invites
    exactly the "just lower the utilization a bit" response this replaces.
    """
    c = _read_config(model_path)
    n_layers = int(c["num_hidden_layers"])
    n_kv = int(c.get("num_key_value_heads", c["num_attention_heads"]))
    head_dim = int(
        c.get("head_dim") or (c["hidden_size"] // c["num_attention_heads"])
    )

    # Same terms as _ensure_staging, including the spare tail page.
    sink_eff = (sink_tokens // block_size) * block_size
    sink_pages = sink_eff // block_size
    tail_pages = (recent_tokens + block_size - 1) // block_size + 1
    pages_per_req = sink_pages + tail_pages
    rows = max_num_seqs * pages_per_req

    per_layer = rows * block_size * n_kv * head_dim * 2 * 2  # bf16, K and V
    total = per_layer * n_layers

    detail = dict(
        n_layers=n_layers, n_kv_heads=n_kv, head_dim=head_dim,
        block_size=block_size, max_num_seqs=max_num_seqs,
        sink_tokens=sink_tokens, sink_eff=sink_eff, sink_pages=sink_pages,
        recent_tokens=recent_tokens, tail_pages=tail_pages,
        pages_per_req=pages_per_req, rows=rows,
        per_layer_mib=per_layer / 2**20, total_gib=total / 2**30,
    )
    if sink_tokens > 0 and sink_eff == 0:
        detail["sink_warning"] = (
            f"sink_tokens={sink_tokens} < block_size={block_size}, so serving "
            f"rounds the sink down to ZERO pages while the training quantizer "
            f"still protects {sink_tokens} tokens exactly.  That is the unsafe "
            f"direction: the student learns to lean on a BF16 prefix deployment "
            f"does not have."
        )
    return total, detail


def adjust_utilization(
    model_path: str,
    max_num_seqs: int,
    gpu_memory_utilization: float,
    *,
    sink_tokens: int = 64,
    recent_tokens: int = 256,
    block_size: int = 16,
    device_index: int = 0,
    verbose: bool = True,
) -> float:
    """Return a utilization that leaves room for the arena.

    Falls back to the input unchanged (with a printed reason) if anything cannot
    be determined -- a wrong reduction would shrink the KV cache for no reason,
    which is its own kind of silent damage.
    """
    if os.environ.get("VLLM_OSCAR_PER_REQUEST_STAGING", "").strip() != "1":
        return gpu_memory_utilization
    try:
        total_mem = total_device_memory(device_index)
        need, d = arena_bytes(
            model_path, max_num_seqs, sink_tokens=sink_tokens,
            recent_tokens=recent_tokens, block_size=block_size,
        )
    except Exception as exc:  # noqa: BLE001
        if verbose:
            print(f"[oscar-budget] could not size the staging arena, leaving "
                  f"gpu_memory_utilization at {gpu_memory_utilization}: "
                  f"{type(exc).__name__}: {exc}", flush=True)
        return gpu_memory_utilization

    frac = need / total_mem
    adjusted = gpu_memory_utilization - frac
    if verbose:
        print(
            f"[oscar-budget] per-request staging arena: {d['rows']} rows "
            f"({d['pages_per_req']}/req x {max_num_seqs} seats) x {d['block_size']} "
            f"x {d['n_kv_heads']} kv x {d['head_dim']} x 2B x 2(K,V) x "
            f"{d['n_layers']} layers = {d['total_gib']:.2f} GiB "
            f"({d['per_layer_mib']:.1f} MiB/layer)",
            flush=True,
        )
        print(
            f"[oscar-budget] that is {100*frac:.1f}% of this GPU, and it is "
            f"allocated AFTER vLLM profiles memory, so it is invisible to the "
            f"profiler.  gpu_memory_utilization {gpu_memory_utilization:.3f} -> "
            f"{adjusted:.3f}",
            flush=True,
        )
        if "sink_warning" in d:
            print(f"[oscar-budget] WARNING: {d['sink_warning']}", flush=True)

    if adjusted < 0.2:
        print(
            f"[oscar-budget] REFUSING to reduce to {adjusted:.3f}: the arena "
            f"({d['total_gib']:.2f} GiB) does not leave a workable KV cache at "
            f"max_num_seqs={max_num_seqs}.  Lower MAX_NUM_SEQS or "
            f"VLLM_OSCAR_RECENT_TOKENS rather than running a configuration that will OOM "
            f"after the model is loaded.",
            flush=True,
        )
        raise SystemExit(2)
    return adjusted
