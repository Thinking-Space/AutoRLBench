"""Read the BF16 staging window AFTER the step, from tensors a graph replay writes.

The instrument this replaces, and why it could not work
-------------------------------------------------------
`oscar_staging_probe.py` wraps the window-bounds Triton kernel and samples its
launch arguments.  Under `cudagraph_mode=FULL_DECODE_ONLY` a replayed graph
never re-enters Python, so the wrapper only ever sees the eager dummy profiling
run.  Commit 688f2fd tried to fix that by keeping the capture-time argument
tensors and reading them back between replays -- and a liveness check killed it:
`seq=[0, 0]` on every captured bucket (job 57761).  Those arguments are
`torch.empty(B)` scratch allocated inside `_decode_attention_windowed`, so each
capture froze a different graph-pool temporary and the readback saw capture-time
dummies.  The conclusion was recorded honestly: the probe is blind under
capture, and blindness is not evidence that the window is fine.

The distinction the fix missed is between graph-pool temporaries and persistent
buffers.  The staging arena is the second kind: `_ensure_staging` allocates
`_oscar_stage_k`, `_oscar_stage_v`, `_oscar_slot_owner`, `_oscar_region_blk0`
and `_oscar_region_of` once, at engine start, and every capture records
*pointers into them*.  A replay therefore writes the live step's tags into the
very tensors Python still holds.  Reading them after `execute_model` returns --
outside the forward, where Python runs on every step whether or not attention
was graphed -- sees exactly what the replay did.

What it reports
---------------
Per sampled step, over the live requests:

    sink   fraction of requests whose every sink slot is tagged with that
           request's own block id -- the paper's first-S-tokens guarantee
    tail   mean number of trailing tokens, out of `recent_tokens`, that are
           both in the recent window and tagged -- the last-R guarantee
    dead   requests holding no region at all (arena oversubscribed)

The same three quantities the in-kernel probe reports, computed from different
tensors by different code, which is the point: run **eager** with both probes on
and they must agree.  That agreement is what licenses reading this one under
`QAT_OSCAR_FULL_CUDAGRAPH=1`, where the other is structurally blind.  A
disagreement in eager means this module is wrong, not that the window is.

Scope
-----
`QAT_OSCAR_REPLAY_PROBE=<interval>` (0/unset = off).  Diagnostic only: it reads,
never writes, and every read is a host sync, so it does not belong in a
production rollout.  Nothing in the shared vLLM install is edited.
"""

from __future__ import annotations

import os

__all__ = ["interval", "enabled", "apply", "maybe_apply", "coverage"]

ENV_FLAG = "QAT_OSCAR_REPLAY_PROBE"

_applied = False
_DONE = False
_steps = 0


def interval() -> int:
    raw = os.environ.get(ENV_FLAG, "").strip()
    if not raw:
        return 0
    try:
        return max(0, int(raw))
    except ValueError:
        return 0


def enabled() -> bool:
    return interval() > 0


def _oscar_layers(runner):
    """The attention modules carrying a staging arena, in model order.

    Two sources, because `static_forward_context` is the documented one but is
    not where this build kept them: j104336 reported "no layer has a staging
    arena" on every sampled step of a run whose staging was demonstrably live
    (the in-kernel probe was printing `matched=16` at the same time).  A probe
    that reports absence when it simply cannot see is worse than no probe -- it
    is the same failure the withdrawn cudagraph readback had -- so fall back to
    walking the model, and say which source answered.
    """
    seen = []
    try:
        ctx = runner.vllm_config.compilation_config.static_forward_context
        seen = [(name, mod) for name, mod in ctx.items()
                if getattr(mod, "_oscar_stage_ready", False)]
    except Exception:  # noqa: BLE001
        ctx = None
    if seen:
        return seen
    try:
        model = runner.get_model()
    except Exception:  # noqa: BLE001
        model = getattr(runner, "model", None)
    if model is None:
        return []
    return [(name, mod) for name, mod in model.named_modules()
            if getattr(mod, "_oscar_stage_ready", False)]


def _why_no_layers(runner) -> str:
    """One-time diagnostic: distinguish 'staging is off' from 'probe is blind'."""
    bits = []
    try:
        ctx = runner.vllm_config.compilation_config.static_forward_context
        bits.append(f"static_forward_context={len(ctx)} entries, "
                    f"{sum(1 for m in ctx.values() if hasattr(m, 'impl'))} with .impl, "
                    f"{sum(1 for m in ctx.values() if getattr(m, '_oscar_stage_ready', False))} ready")
    except Exception as exc:  # noqa: BLE001
        bits.append(f"static_forward_context unavailable: {type(exc).__name__}")
    try:
        model = runner.get_model()
        mods = list(model.named_modules())
        bits.append(f"model has {len(mods)} modules, "
                    f"{sum(1 for _, m in mods if getattr(m, '_oscar_stage_ready', False))} ready")
    except Exception as exc:  # noqa: BLE001
        bits.append(f"get_model unavailable: {type(exc).__name__}")
    return "; ".join(bits)


def coverage(layer, impl, bt, seq, n_reqs: int):
    """Sink/tail coverage of one layer, from the arena and the live block table.

    `bt` is `[n_reqs, max_pages]` of block ids and `seq` is `[n_reqs]` of
    sequence lengths, both as the kernels saw them this step.  Mirrors
    `_oscar_window_bounds_kernel_pr`'s definition of "staged": a slot counts
    only when the owner tag equals the block id the request actually holds for
    that page, which is what makes a stale tag from a previous tenant read as
    missing rather than as coverage.
    """
    import torch  # noqa: PLC0415

    ppr = impl.pages_per_req
    sp, tp = impl.sink_pages, impl.tail_pages
    bs = impl.stage_block
    S_eff, W = impl.sink_eff, impl.cfg.recent_tokens
    S = impl.stage_slots
    dev = bt.device

    owner = layer._oscar_slot_owner                       # [rows, bs]
    reg = layer._oscar_region_of[:n_reqs].to(torch.int64)  # [n_reqs]
    dead = int((reg >= S).sum())
    # Park the region-less on the trash region so the gathers stay in bounds;
    # they are excluded from every statistic below by `alive`.
    reg_safe = reg.clamp(max=S)
    alive = reg < S

    offs = torch.arange(bs, device=dev)

    # --- sink: pages [0, sp) of the request, in the sink half of its region ---
    if sp > 0:
        spg = torch.arange(sp, device=dev)
        srow = reg_safe.view(-1, 1) * ppr + spg.view(1, sp)            # [R, sp]
        sown = owner[srow]                                             # [R,sp,bs]
        sblk = bt[:n_reqs, :sp].to(torch.int64).unsqueeze(-1)
        spos = (spg.view(1, sp, 1) * bs) + offs.view(1, 1, bs)
        # A sink slot past S_eff is not part of the window, so it is not owed.
        want = spos < S_eff
        ok = ((sown == sblk) | ~want).all(dim=-1).all(dim=-1)           # [R]
        # Only requests long enough to have left the sink behind are owed one.
        owed = alive & (seq[:n_reqs] > S_eff)
        sink_full = float(ok[owed].float().mean()) if int(owed.sum()) else float("nan")
        n_owed = int(owed.sum())
    else:
        sink_full, n_owed = float("nan"), 0

    # --- tail: the last tp pages, counted back from seq-1 ---
    last_page = (seq[:n_reqs] - 1) // bs
    pg = (last_page - (tp - 1)).unsqueeze(1) + torch.arange(tp, device=dev)
    pg_ok = pg >= 0
    maxpg = bt.shape[1]
    tblk = torch.gather(bt[:n_reqs].to(torch.int64), 1, pg.clamp(0, maxpg - 1))
    trow = reg_safe.view(-1, 1) * ppr + sp + (pg.clamp(min=0) % tp)
    town = owner[trow]                                                 # [R,tp,bs]
    pos = (pg * bs).unsqueeze(-1) + offs.view(1, 1, bs)
    inwin = (pos >= (seq[:n_reqs] - W).view(-1, 1, 1)) & (
        pos < seq[:n_reqs].view(-1, 1, 1))
    staged = (town == tblk.unsqueeze(-1)) & pg_ok.unsqueeze(-1) & inwin
    depth = staged.sum(dim=(1, 2)).float()                             # [R]
    tail_mean = float(depth[alive].mean()) if int(alive.sum()) else float("nan")

    tagged_rows = int((owner >= 0).any(dim=1).sum())
    return {
        "sink_full": sink_full, "n_owed": n_owed,
        "tail_mean": tail_mean, "tail_of": W,
        "dead": dead, "n_reqs": n_reqs,
        "tagged_rows": tagged_rows, "rows": int(owner.shape[0]),
    }


def _report(runner, tag: str) -> None:
    import torch  # noqa: PLC0415

    layers = _oscar_layers(runner)
    if not layers:
        print(f"[oscar-replay] {tag}: no layer has a staging arena "
              f"(per-request staging off, or no OSCAR backend) -- {_why_no_layers(runner)}",
              flush=True)
        return
    ib = runner.input_batch
    n_reqs = int(ib.num_reqs)
    if n_reqs <= 0:
        return
    # Signature drifted: this build's `BlockTable.get_device_tensor` takes
    # `num_reqs` and returns `block_table.gpu[:num_reqs]`; older ones took none
    # and returned the whole padded buffer.  j104404 lost every coverage sample
    # to the TypeError, so accept both rather than pin one.
    _bt = ib.block_table[0]
    try:
        bt = _bt.get_device_tensor(n_reqs)
    except TypeError:
        bt = _bt.get_device_tensor()
    # `runner.seq_lens` is the persistent device buffer `_prepare_inputs` fills
    # and the attention metadata points at, so it holds exactly the lengths the
    # kernels used this step.  `input_batch.num_computed_tokens_cpu` is the
    # scheduler's bookkeeping and is a step out of phase, which would shift the
    # recent window by one token and make the tail number wrong at the edges.
    seq = runner.seq_lens[:n_reqs].to(torch.int64)

    # First and middle layer: a window that dies under capture dies on all of
    # them, and printing 28 lines per sample buries the number.
    for name, mod in (layers[0], layers[len(layers) // 2]):
        impl = mod.impl
        if not hasattr(impl, "pages_per_req"):
            continue
        c = coverage(mod, impl, bt, seq, min(n_reqs, bt.shape[0]))
        print(f"[oscar-replay] {tag} {name}: reqs={c['n_reqs']} "
              f"sink={c['sink_full'] * 100:.1f}% full (of {c['n_owed']} owed) "
              f"tail={c['tail_mean']:.1f}/{c['tail_of']} "
              f"dead={c['dead']} "
              f"rows_tagged={c['tagged_rows']}/{c['rows']}", flush=True)


def apply(verbose: bool = True) -> bool:
    """Wrap GPUModelRunner.execute_model.  Idempotent; returns True if installed."""
    global _DONE, _applied
    if _DONE:
        return True
    n = interval()
    if not n:
        return False
    try:
        from vllm.v1.worker.gpu_model_runner import (  # noqa: PLC0415
            GPUModelRunner,
        )
    except Exception as exc:  # noqa: BLE001
        if verbose:
            print(f"[oscar-replay] not applied: {type(exc).__name__}: {exc}",
                  flush=True)
        return False

    orig = GPUModelRunner.execute_model

    def execute_model(self, *args, **kwargs):
        out = orig(self, *args, **kwargs)
        global _steps
        _steps += 1
        if _steps % n == 0:
            try:
                _report(self, f"step {_steps}")
            except Exception as exc:  # noqa: BLE001
                # A diagnostic must never take the engine down.
                print(f"[oscar-replay] probe error: {type(exc).__name__}: "
                      f"{exc}", flush=True)
        return out

    GPUModelRunner.execute_model = execute_model
    _DONE = True
    _applied = True
    if verbose:
        print(f"[oscar-replay] post-step arena probe on, 1 step in {n}. "
              f"Reads the persistent staging arena after execute_model, so it "
              f"sees what a CUDA-graph replay wrote -- unlike the in-kernel "
              f"probe, which a replay never reaches.", flush=True)
    return True


def maybe_apply(verbose: bool = True) -> bool:
    return apply(verbose=verbose) if enabled() else False
