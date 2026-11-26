"""Opt-in: let vLLM put OSCAR's decode attention inside the CUDA graph.

The k2v3 and BF16 arms already do this by default -- 33854 and 33855 log
`FULL=23 (largest=160)` alongside `PIECEWISE=39 (largest=320)`, while the
OSCAR arm logs only the PIECEWISE line and a downgrade warning.  So this does
not give OSCAR something the other arms lack; it stops OSCAR being the only
arm denied it, which is also why the three-way 8B comparison was not a fair
speed comparison before.

What this changes
-----------------
`oscar_attn.py` declares `_cudagraph_support = AttentionCGSupport.NEVER`, so a
request for `cudagraph_mode=FULL_DECODE_ONLY` is silently downgraded to
`PIECEWISE` -- torch.compile still runs and still graphs the model, but the
graph is split at every attention op.  Two things stand in the way of lifting
that, and this module removes both *in this process only*.  It never edits the
shared vLLM install, which other people's jobs run out of.

  1. `_cudagraph_support` -> UNIFORM_BATCH, the tier `turboquant_attn.py:201`
     already claims and the weakest one that permits a decode-only graph.

  2. `_staging_write`'s

         req = torch.repeat_interleave(_oscar_arange(seq.shape[0], dev), q_lens)

     is a host-side tensor whose extent depends on runtime values, which is
     illegal while a stream is capturing -- it is the actual error thrown
     ("CUDA error: operation not permitted when stream is capturing"), not the
     "data-dependent block allocation" the backend's comment blames.  Under
     capture every request contributes exactly one token, so `req` is just
     `arange(n_reqs)`; the original construction is kept for every other case.

What it is worth, and why it is believed safe
---------------------------------------------
Measured on the 1.5B with the calibrated rotation, output tok/s, PIECEWISE ->
FULL (jobs 34014 and 34034):

    batch    8     253.25 ->  692.83   2.74x
    batch   32     946.57 -> 2528.80   2.67x
    batch  128    2986.86 -> 6777.23   2.27x
    batch  160    3423.05 -> 7484.78   2.19x   <- production concurrency

The ratio does not collapse with batch the way EVAL_ENFORCE_EAGER's did.

On correctness, in order of how much each measurement constrains:

  * One decode call of one layer, captured and replayed, is **bit-identical**
    to eager (0.000e+00), and replay tracks in-place mutation of seq_lens,
    query and block_table exactly -- err 0.000e+00 against signals of 0.70,
    1.97 and 0.21.  Capture leaves the staging arena untouched.  (Job 33984.)
  * The harness is deterministic: the same config twice is identical 8/8 with
    worst |dlogprob| 0.000e+00.  (Job 33998.)
  * End to end, PIECEWISE vs FULL differ, but by less than a perturbation this
    project already accepts.  Changing `max_num_seqs` from 8 to 3 -- both
    PIECEWISE, and a knob the AIME launchers moved 64 -> 160 last week --
    gives per-step median 1.573e-02 and worst pre-flip 9.253e-01; PIECEWISE vs
    FULL gives 1.563e-02 and 6.060e-01.  (Job 34033.)
  * The difference is 0.000e+00 at the position the prefill produces and
    ~1.6e-2 at the first decode position, which is about one bf16 ulp on a
    logit of that size.  A wrong block table or a shifted seq_lens would be
    O(1) wrong at that first step, not one ulp.  It is not the sink/recent
    window (7.308e-01 with the window disabled) and not `_oscar_arange`'s
    memoisation (6.060e-01, identical to the last digit with it off).
    (Job 34012.)

Default off
-----------
Gated on QAT_OSCAR_FULL_CUDAGRAPH=1.  Enabling it changes numerics by about as
much as re-running at a different concurrency, which is small but not nothing,
so arms that are meant to be compared to already-finished runs should leave it
alone.
"""
from __future__ import annotations

import os

ENV_FLAG = "QAT_OSCAR_FULL_CUDAGRAPH"

_applied = False


def enabled() -> bool:
    return os.environ.get(ENV_FLAG, "0") not in ("", "0", "false", "False")


def apply(verbose: bool = True) -> bool:
    """Patch the in-process OSCAR backend.  Returns True if it was applied.

    Safe to call when vLLM has no OSCAR backend (a k2v3 or BF16 arm): the
    import fails and this returns False rather than raising.  Must be called
    before the engine resolves its compilation config, i.e. before LLM(...).
    """
    global _applied
    if _applied or not enabled():
        return False
    try:
        import torch
        import triton
        # oscar_attn.py:29 imports AttentionCGSupport from
        # vllm.v1.attention.backend; vllm.v1.attention.backends.abstract does
        # not exist in this build.
        from vllm.v1.attention.backend import AttentionCGSupport
        import vllm.v1.attention.backends.oscar_attn as _oa
    except Exception as exc:  # noqa: BLE001
        if verbose:
            print(f"[oscar-cg] not applied: {type(exc).__name__}: {exc}",
                  flush=True)
        return False

    _oa.OscarMetadataBuilder._cudagraph_support = AttentionCGSupport.UNIFORM_BATCH

    def _staging_write(self, layer, key, value, attn_metadata):
        N = attn_metadata.num_actual_tokens
        if N == 0:
            return
        dev = key.device
        slot = attn_metadata.slot_mapping[:N].to(torch.int64)
        seq = attn_metadata.seq_lens.to(torch.int64)
        qsl = attn_metadata.query_start_loc.to(torch.int64)
        # num_reqs is seq.shape[0], not qsl.shape[0] - 1: query_start_loc has
        # one entry per *token* boundary, so an 8192-token prefill makes the
        # latter 8192 while seq has one row per request.
        n_reqs = seq.shape[0]
        # Decide uniformity from CPU-side metadata.  A .item() on a CUDA tensor
        # is the very sync that is illegal inside a capturing stream, so it
        # cannot be used to guard a path that runs during capture.
        qsl_cpu = attn_metadata.query_start_loc_cpu
        if qsl_cpu is not None and qsl_cpu.shape[0] == n_reqs + 1:
            uniform = bool(((qsl_cpu[1:] - qsl_cpu[:-1]) == 1).all())
        else:
            uniform = N == n_reqs
        if uniform:
            req = _oa._oscar_arange(n_reqs, dev)
        else:
            req = torch.repeat_interleave(
                _oa._oscar_arange(n_reqs, dev), qsl[1:] - qsl[:-1]
            )
        if req.dtype != torch.int64:
            # _oscar_arange caches by (n, device) and other callers want int32;
            # keep this row int64 to match what the original expression built.
            # The Triton kernel types req_ptr from the tensor passed at launch,
            # so a dtype change would silently force a JIT recompile per step.
            req = req.to(torch.int64)
        pos = seq[req] - qsl[req + 1] + _oa._oscar_arange(N, dev)

        bs = self.stage_block
        sk, sv = layer._oscar_stage_k, layer._oscar_stage_v
        own = layer._oscar_slot_owner
        n_heads, head_dim = key.shape[1], key.shape[2]
        BLOCK_D = triton.next_power_of_2(head_dim)
        for _phase in (0, 1):
            _oa._oscar_staging_write_kernel[(N, n_heads if _phase else 1)](
                key, value, sk, sv, own,
                slot, pos,
                key.stride(0), key.stride(1),
                sk.stride(0), sk.stride(1), sk.stride(2),
                own.stride(0), own.stride(1),
                seq, req,
                N, n_heads, head_dim, bs, layer._oscar_stage_rows,
                self.sink_eff, self.cfg.recent_tokens,
                BLOCK_D=BLOCK_D, PHASE=_phase,
            )

    _oa.OscarAttentionImpl._staging_write = _staging_write
    _applied = True
    if verbose:
        print("[oscar-cg] full decode graphs enabled for OscarAttentionBackend "
              "(in-process patch; shared install untouched). vLLM's default "
              "FULL_AND_PIECEWISE now applies, as it already does for k2v3 "
              "and BF16 -- expect both a FULL= and a PIECEWISE= line.", flush=True)
    return True
