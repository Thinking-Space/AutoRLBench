"""Apply this repo's OSCAR patches in *every* vLLM process, not just the parent.

The failure this exists to prevent
----------------------------------
All four OSCAR patches are in-process monkeypatches applied from
``opqad/evaluation/eval_vllm_turboquant.py`` before ``LLM(...)`` is constructed:

    oscar_cudagraph_patch   lifts AttentionCGSupport.NEVER
    oscar_staging_patch     per-request BF16 staging regions
    oscar_staging_probe     window-coverage instrumentation
    oscar_gqa_patch         GQA-reuse INT2 decode kernel

They reach the EngineCore worker only because vLLM forks it, inheriting the
parent's module state.  vLLM falls back to ``spawn`` whenever CUDA is already
initialized in the parent, and under ``spawn`` the worker is a fresh interpreter
that imports stock vLLM.  The patches then apply in the parent -- which prints
their banners, so the log looks right -- while every kernel that actually runs
is the stock one.

That happened for real: ``opqad/runtime/oscar_stage_budget`` called
``torch.cuda.get_device_properties`` merely to read the card's size, which is
enough to initialize CUDA.  Detection is possible after the fact only by reading
which Triton kernels JIT-compiled (``_oscar_window_bounds_kernel`` vs
``..._pr``), which nobody does routinely.

Registering as a ``vllm.general_plugins`` entry point removes the dependence on
fork entirely.  vLLM's own comment on the group
(``vllm/plugins/__init__.py:13``) is explicit: "Default plugins group will be
loaded in all processes (process0, engine core process and worker processes)",
and ``load_general_plugins`` guards against double-loading per process.  Each
``apply``/``maybe_apply`` below is independently idempotent as well, so applying
in both the parent and a forked child is a no-op the second time.

Discovery
---------
``importlib.metadata.entry_points`` scans ``sys.path`` for ``*.dist-info``
directories, so a dist-info at the repo root is discovered as soon as the repo
is on ``PYTHONPATH`` -- which the maintained evaluation launchers already guarantee.  Nothing is installed into the shared environment at
``/shared/*/bitdistiller/envs/vllm-0.19.1-test``; other people's jobs run out of
it and it is never edited.

Every patch stays opt-in on its own environment variable, so loading this plugin
in an unrelated vLLM process (a k2v3 or BF16 arm, someone else's job that happens
to inherit PYTHONPATH) changes nothing.
"""
from __future__ import annotations

import os

_DONE = False


def register() -> None:
    """Entry point.  Applies whatever the environment has enabled.

    Never raises: a plugin that throws takes the engine down, and none of these
    patches is worth failing a run over.  vLLM logs the exception itself if one
    escapes, but the per-patch handling here keeps one failure from suppressing
    the others.
    """
    global _DONE
    if _DONE:
        return
    _DONE = True

    if os.environ.get("QAT_OSCAR_PLUGIN", "1").strip() == "0":
        return

    tag = "parent" if os.environ.get("VLLM_ENGINE_ROLE") is None else "worker"
    applied: list[str] = []
    # Patches whose module said "already installed" when we asked.  These are
    # ACTIVE -- they were just installed by someone else first, normally
    # opqad/evaluation/eval_vllm_turboquant.py:_make_llm, which applies the same set
    # verbosely before the engine is built.  Without this, the banner reads as
    # if the patch were off: job 62871's eval printed `[oscar-gqa] decode
    # kernel=packed ON` and then `applied: quant-all-layers, v-stride-fix,
    # per-request-staging` two lines later.  A gate line that under-reports the
    # thing it gates is worse than no gate line.
    already: list[str] = []
    # Only `_applied` is consulted, and only the four modules that use it can
    # land here.  The other four (oscar_layer_skip_patch, oscar_v_stride_patch,
    # oscar_staging_patch) already `return True` when re-applied, so they go
    # straight into `applied` -- and their `_DONE`/`_PATCHED` flags are NOT
    # equivalent: reading those would risk reporting a disabled patch as live,
    # which is the opposite of what this banner is for.  `_applied` is set only
    # after an install that required `enabled()`, so True there means live.
    def _installed(module) -> bool:
        return bool(getattr(module, "_applied", False))

    def _try(name: str, fn, module=None) -> None:
        try:
            if fn():
                applied.append(name)
            elif module is not None and _installed(module):
                already.append(name)
        except Exception as exc:  # noqa: BLE001
            print(f"[oscar-plugin] {name} failed: {type(exc).__name__}: {exc}",
                  flush=True)

    # Order matters between exactly two of these: the staging patch rebinds the
    # window-bounds kernel that the probe wraps, so the probe must come second
    # or it wraps a kernel nothing will launch.  Same ordering as
    # opqad/evaluation/eval_vllm_turboquant.py:_make_llm.
    from opqad.quantization import hybrid_kv_fakequant
    from opqad.integrations import kivi_vllm_fakequant
    from opqad.runtime import oscar_cudagraph_patch
    from opqad.runtime import oscar_gqa_patch
    from opqad.runtime import oscar_layer_skip_patch
    from opqad.runtime import oscar_quantile_patch
    from opqad.diagnostics import oscar_replay_probe
    from opqad.runtime import oscar_staging_patch
    from opqad.diagnostics import oscar_staging_probe
    from opqad.runtime import oscar_v_stride_patch
    from opqad.kernels import tq_decode_lut
    from opqad.kernels import tq_decode_tune
    from opqad.kernels import tq_decode_v2

    _try("quant-all-layers", oscar_layer_skip_patch.maybe_apply,
         oscar_layer_skip_patch)
    # Always on: the stock staging kernel reads V through K's strides (wrong
    # on Qwen3).  Must precede per-request staging, which rebinds the same
    # method with a kernel that takes V's own strides.
    _try("v-stride-fix", oscar_v_stride_patch.apply, oscar_v_stride_patch)
    _try("cudagraph", lambda: oscar_cudagraph_patch.apply(verbose=False),
         oscar_cudagraph_patch)
    _try("per-request-staging", oscar_staging_patch.maybe_apply,
         oscar_staging_patch)
    _try("staging-probe", lambda: oscar_staging_probe.apply(verbose=False),
         oscar_staging_probe)
    _try("gqa-decode", lambda: oscar_gqa_patch.maybe_apply(verbose=False),
         oscar_gqa_patch)
    # Always on.  Rebinds `_rotate_clip` to take the clip threshold as an order
    # statistic instead of torch.quantile.  Bitwise-identical output; it exists
    # purely to stop sorting 128 numbers twice per layer per token.
    # Independent of everything above -- it touches the KV-store path, not the
    # staging arena, the decode launcher or the compilation config.
    _try("fast-quantile",
         lambda: oscar_quantile_patch.maybe_apply(verbose=False),
         oscar_quantile_patch)
    # Diagnostic, off by default (QAT_OSCAR_REPLAY_PROBE=<interval>).  Reads the
    # staging arena AFTER execute_model, which is the only vantage point that
    # survives cudagraph replay -- the in-kernel probe above never runs during
    # one.  Costs host syncs on the sampled steps, so it is for settling the
    # capture question, not for production.
    _try("replay-probe",
         lambda: oscar_replay_probe.maybe_apply(verbose=False),
         oscar_replay_probe)
    # Unrelated to OSCAR: the KIVI baseline arm (QAT_KIVI_FAKEQUANT=1).  It
    # wraps the FlashAttention backend, not the OSCAR one, so it never runs in
    # the same process as an oscar_int2 arm.
    _try("kivi-fakequant",
         lambda: kivi_vllm_fakequant.maybe_apply(verbose=False),
         kivi_vllm_fakequant)
    # The hybrid arm (QAT_HYBRID_FAKEQUANT=1): KIVI's per-channel K, OSCAR's
    # rotated per-token V, OSCAR's BF16 sink+recent window.  Mutually exclusive
    # with the KIVI arm -- both wrap the same backend forward, and running both
    # would quantize each block twice under two different rules.
    if kivi_vllm_fakequant.enabled() and hybrid_kv_fakequant.enabled():
        raise RuntimeError(
            "QAT_KIVI_FAKEQUANT and QAT_HYBRID_FAKEQUANT are both set; they "
            "wrap the same attention forward and the result would be neither "
            "scheme."
        )
    _try("hybrid-fakequant",
         lambda: hybrid_kv_fakequant.maybe_apply(verbose=False),
         hybrid_kv_fakequant)
    # Unrelated to OSCAR: k2v3's decode kernel launch-config retune
    # (QAT_TQ_DECODE_TUNE=1). Rebinds turboquant_attn's decode entry point,
    # which never loads in an OSCAR/BF16 process, so this is a no-op there.
    _try("tq-decode-tune", lambda: tq_decode_tune.apply(verbose=False),
         tq_decode_tune)
    # k2v3's ported upstream v2 decode kernel (QAT_TQ_DECODE_V2=1). Replaces
    # the whole launcher, where tq_decode_tune only retunes the stock one's
    # launch params -- whichever applied second would silently win, so they
    # are mutually exclusive.
    # ... and tq_decode_lut (QAT_TQ_DECODE_LUT=1), which keeps the stock
    # grid and fp32 accumulation and rewrites only the key-dequant centroid
    # gather. All three rebind the same entry point, so at most one may be on.
    _tq_on = [n for n, m in (("QAT_TQ_DECODE_TUNE", tq_decode_tune),
                             ("QAT_TQ_DECODE_V2", tq_decode_v2),
                             ("QAT_TQ_DECODE_LUT", tq_decode_lut))
              if m.enabled()]
    if len(_tq_on) > 1:
        raise RuntimeError(
            f"{' and '.join(_tq_on)} are all set; they rebind the same decode "
            "entry point and only one would survive."
        )
    _try("tq-decode-v2", lambda: tq_decode_v2.apply(verbose=False),
         tq_decode_v2)
    _try("tq-decode-lut", lambda: tq_decode_lut.apply(verbose=False),
         tq_decode_lut)

    # One line, in every process, naming what is live.  The whole class of bug
    # above was invisible because the only banners came from the parent.
    # `already:` is as load-bearing as `applied:` -- both mean the patch is
    # installed in this process; they differ only in who installed it.
    live = applied + [f"{n}(pre)" for n in already]
    print(f"[oscar-plugin] pid={os.getpid()} ({tag}) applied: "
          f"{', '.join(live) if live else '<nothing enabled>'}",
          flush=True)
