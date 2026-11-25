"""Install the TurboQuant fake quantizer at the point inference quantizes.

Where the quantizer has to go
-----------------------------
Both HF and vLLM apply RoPE *before* the KV cache::

    modeling_qwen2.py:158   query_states, key_states = apply_rotary_pos_emb(...)
    modeling_qwen2.py:163   past_key_values.update(key_states, value_states, ...)

    vllm/model_executor/models/qwen2.py:233   q, k = self.rotary_emb(positions, q, k)
                                              -> attn -> do_kv_cache_update -> TQ

and ``triton_turboquant_store`` documents its input as ``raw keys (post-RoPE)``.

The pre-existing ``KVFakeQuantHook`` hooks ``k_proj``/``v_proj`` outputs, which
are **pre**-RoPE.  That is a tolerable approximation for a per-token symmetric
quantizer, but it is simply the wrong operator for TurboQuant: TurboQuant
normalizes K to the unit sphere and quantizes its *direction* on a Hadamard-
rotated Lloyd-Max grid, and RoPE changes that direction.  Quantizing before
RoPE trains against a different quantizer than the one deployed -- silently.

So this module hooks the attention interface instead of the projections.
``AttentionInterface.register`` is transformers' supported extension point, and
the registered callable receives ``(module, query, key, value, ...)`` with
key/value already post-RoPE, still un-repeated (``num_key_value_heads``), and
with ``head_dim`` last -- exactly the layout the quantizer wants.  It also
receives the attention module, so ``module.layer_idx`` drives the boundary-skip
list without any extra plumbing.

This is preferable to copying ``Qwen2Attention.forward``: the copy would pin us
to one transformers version and diverge silently on upgrade.
"""

from __future__ import annotations

import contextlib
import itertools
import threading
from typing import Any, Iterable, Optional

import torch

from opqad.quantization.oscar_fake_quant import (
    OSCAR_PRESETS,
    OscarFakeQuantConfig,
    oscar_fake_quant,
)
from opqad.kernels.oscar_windowed_attn import require_for_oscar as _require_oscar_window
from opqad.quantization.int4_ptkh_fake_quant import (
    INT4_PTKH_PRESETS,
    Int4PTKHConfig,
    int4_ptkh_fake_quant,
)
from opqad.quantization.turboquant_fake_quant import (
    TurboQuantFakeQuantConfig,
    build_hadamard,
    lloyd_max_centroids,
    oscar_quant_all_layers,
    skip_layer_indices,
    turboquant_fake_quant_keys,
    turboquant_fake_quant_values,
)

__all__ = [
    "TurboQuantAttnState",
    "get_installed_state",
    "install_turboquant_fake_quant",
    "uninstall_turboquant_fake_quant",
]

_ATTN_NAME = "qat_turboquant"
# ALL_ATTENTION_FUNCTIONS is a process-global registry, and ``_tq_attention``
# is a closure over one specific state/base_fn.  Registering every install
# under the same key means the *last* install wins for every model that names
# it -- so installing the capture-only hook on the frozen theta_0 reference
# silently rebinds the actor's attention to the reference's state (student K/V
# stop being recorded; the actor stops using its own quantizer).  Give each
# install its own key instead.
_ATTN_SEQ = itertools.count()


def _is_qat_attn_name(name) -> bool:
    return isinstance(name, str) and (name == _ATTN_NAME or name.startswith(_ATTN_NAME + "_"))


class TurboQuantAttnState:
    """Live state for one installed quantizer, including a call counter.

    The counter exists because every realistic failure here is silent: a wrong
    ``_attn_implementation``, an FSDP re-wrap that drops the config, or a skip
    list that covers everything all produce a training run that looks healthy
    and quantizes nothing.  ``assert_layers_quantized`` turns that into a loud
    failure.
    """

    def __init__(self, cfg, skip_layers, base_impl, num_layers):
        self.cfg = cfg
        self.skip_layers = frozenset(skip_layers)
        self.base_impl = base_impl
        # Set by install_turboquant_fake_quant to this install's unique key in
        # ALL_ATTENTION_FUNCTIONS; see _ATTN_SEQ.
        self.attn_name = _ATTN_NAME
        self.num_layers = num_layers
        self.quantized_layers: set[int] = set()
        self.calls = 0
        # Toggled by ``disabled()`` so the same weights can produce both the
        # quantized and the full-precision activations without a second model.
        self.enabled = True
        # Number of leading positions whose queries attend to *raw* K/V.
        #
        # vLLM's TurboQuant backend only quantizes what goes into the paged
        # cache.  A first-chunk prefill (max_query_len == max_seq_len) is
        # served by flash-attn on the raw post-RoPE K/V and never reads the
        # cache, so at inference a prompt token attends to unquantized K/V
        # while a generated token attends to the quantized cache.  A plain
        # teacher-forced training forward quantizes both.  Setting this to the
        # prompt length reproduces the inference split exactly; leaving it 0
        # keeps the stricter uniform behaviour.  See
        # ``tools/analysis/parity_forward_logprobs.py``, which measures the gap.
        self.unquantized_prefix = 0
        self._lock = threading.Lock()
        self._buffers: dict[Any, tuple[torch.Tensor, torch.Tensor]] = {}
        # KV capture for the KV-alignment term.  When ``record_kv`` is set the
        # wrapper stashes the dequantized K/V actually fed to SDPA (i.e. the
        # quantizer output, differentiable) per quantized layer under ``kv`` --
        # this is the student branch of ``opqad.losses.kv_align``.  ``capture_only``
        # runs the same wrapper as a pure pass-through: it skips the quantizer
        # and records the raw post-RoPE K/V.  That is how the frozen teacher
        # reference produces its full-precision K/V without a second operator:
        # the identical wrapper, identical interception point, quantizer off.
        # See ``opqad.losses.kv_align.reference_forward``.
        self.record_kv = False
        self._capture_only = False
        self.kv: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

    @contextlib.contextmanager
    def capturing_kv(self):
        """Record the dequantized K/V of the quantized forward per layer.

        Sets ``record_kv`` for the duration of a forward; the wrapper stashes
        ``(key, value)`` under ``state.kv[layer_idx]`` for every quantized layer
        that runs.  ``state.kv`` is cleared on entry so a wrapped-forward holder
        never reads a stale capture.  Exits restore ``record_kv`` to whatever it
        was.
        """
        prev = self.record_kv
        self.record_kv = True
        self.kv = {}
        try:
            yield self.kv
        finally:
            self.record_kv = prev

    @contextlib.contextmanager
    def capture_only(self):
        """Run the wrapped forward in full precision and record the raw K/V.

        This is the frozen-teacher mode: the quantizer is bypassed (equivalent
        to ``disabled()``) and the raw post-RoPE K/V at the same interception
        point are recorded under ``state.kv``.  Unlike ``disabled()`` this also
        populates ``state.kv``, so the KV-alignment reference forward needs no
        second operator and no cross-worker transport.
        """
        prev_enabled, prev_capture = self.enabled, self._capture_only
        self.enabled = True
        self._capture_only = True
        self.kv = {}
        try:
            yield self.kv
        finally:
            self.enabled = prev_enabled
            self._capture_only = prev_capture

    @contextlib.contextmanager
    def disabled(self):
        """Temporarily run the model in full precision.

        This is what makes block alignment cheap for QAT: the reference
        activations are the model's *own* fp forward, so there is no second
        model to hold and no hidden states to ship between Ray workers.
        """
        prev = self.enabled
        self.enabled = False
        try:
            yield self
        finally:
            self.enabled = prev

    @contextlib.contextmanager
    def prefill_exact(self, prompt_len: int):
        """Attend to raw K/V for the first ``prompt_len`` query positions.

        Mirrors what vLLM actually does: quantization applies to cache reads,
        and a first-chunk prefill does not read the cache.
        """
        prev = self.unquantized_prefix
        self.unquantized_prefix = max(0, int(prompt_len))
        try:
            yield self
        finally:
            self.unquantized_prefix = prev

    @property
    def expected_layers(self) -> set[int]:
        return {i for i in range(self.num_layers) if i not in self.skip_layers}

    def buffers_for(self, device, dtype=torch.float32):
        key = (str(device), self.cfg.head_dim, self.cfg.key_quant_bits)
        buf = self._buffers.get(key)
        if buf is None:
            buf = (
                build_hadamard(self.cfg.head_dim, device, dtype),
                lloyd_max_centroids(self.cfg.head_dim, self.cfg.key_quant_bits, device),
            )
            self._buffers[key] = buf
        return buf

    def note(self, layer_idx: int) -> None:
        with self._lock:
            self.calls += 1
            if layer_idx is not None:
                self.quantized_layers.add(int(layer_idx))

    def reset_counters(self) -> None:
        with self._lock:
            self.calls = 0
            self.quantized_layers.clear()

    def assert_layers_quantized(self) -> None:
        missing = self.expected_layers - self.quantized_layers
        extra = self.quantized_layers & self.skip_layers
        if missing or extra:
            raise RuntimeError(
                "TurboQuant fake quantization did not run where expected: "
                f"missing={sorted(missing)}, unexpectedly_quantized={sorted(extra)}, "
                f"calls={self.calls}"
            )

    def metadata(self) -> dict:
        return {
            "scheme": getattr(self.cfg, "scheme_name", "turboquant"),
            "key_quant_bits": self.cfg.key_quant_bits,
            "value_quant_bits": self.cfg.value_quant_bits,
            "norm_correction": self.cfg.norm_correction,
            "head_dim": self.cfg.head_dim,
            "num_layers": self.num_layers,
            "skip_layers": sorted(self.skip_layers),
            "quantized_layers": sorted(self.quantized_layers),
            "calls": self.calls,
            "enabled": self.enabled,
            "unquantized_prefix": self.unquantized_prefix,
            "base_attn_impl": self.base_impl,
            "hook_point": "attention_interface_post_rope",
        }


def _iter_configs(model) -> Iterable[Any]:
    seen = set()
    for cfg in (getattr(model, "config", None), getattr(model, "generation_config", None)):
        if cfg is not None and id(cfg) not in seen:
            seen.add(id(cfg))
            yield cfg
    for mod in model.modules():
        cfg = getattr(mod, "config", None)
        if cfg is not None and id(cfg) not in seen:
            seen.add(id(cfg))
            yield cfg
        sub = getattr(cfg, "text_config", None) if cfg is not None else None
        if sub is not None and id(sub) not in seen:
            seen.add(id(sub))
            yield sub


def install_turboquant_fake_quant(
    model: torch.nn.Module,
    preset: str | TurboQuantFakeQuantConfig,
    *,
    head_dim: Optional[int] = None,
    num_layers: Optional[int] = None,
    skip_layers: Optional[Iterable[int]] = None,
    base_impl: Optional[str] = None,
    capture_only: bool = False,
) -> TurboQuantAttnState:
    """Route this model's attention through the TurboQuant fake quantizer.

    Args:
        model: an HF causal LM whose attention goes through ALL_ATTENTION_FUNCTIONS.
        preset: a preset name (e.g. ``"turboquant_3bit_nc"``) or an explicit config.
        head_dim / num_layers: read from ``model.config`` when omitted.
        skip_layers: defaults to vLLM's boundary rule (first 2 and last 2) for
            TurboQuant / INT4-PTKH, and to the empty set for OSCAR presets,
            which quantize every layer unconditionally. Passing it explicitly
            is for tests; production paths leave it None so both sides derive
            the same list.
        base_impl: attention implementation to delegate to; defaults to the
            model's current one (``sdpa`` here).
    """
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    cfg_obj = getattr(model, "config", None)
    text_cfg = getattr(cfg_obj, "text_config", cfg_obj)
    if head_dim is None:
        head_dim = getattr(text_cfg, "head_dim", None) or (
            text_cfg.hidden_size // text_cfg.num_attention_heads
        )
    if num_layers is None:
        num_layers = text_cfg.num_hidden_layers

    # Two quantizer families share this hook because they share an interception
    # point (post-RoPE, un-repeated K/V, head_dim last) and everything around
    # it -- skip layers, capture, disable, the call counter.  They differ only
    # in the operator, so the dispatch is a single branch in ``_quantize``.
    if isinstance(preset, (TurboQuantFakeQuantConfig, Int4PTKHConfig, OscarFakeQuantConfig)):
        tq_cfg = preset
    elif preset in OSCAR_PRESETS:
        tq_cfg = OscarFakeQuantConfig.from_preset(preset, head_dim)
    elif preset in INT4_PTKH_PRESETS:
        tq_cfg = Int4PTKHConfig.from_preset(preset, head_dim)
    else:
        tq_cfg = TurboQuantFakeQuantConfig.from_preset(preset, head_dim)
    if tq_cfg.return_rotated:
        raise ValueError("return_rotated is for parity testing only; attention needs the original basis")

    # Fail here, not in the first forward: an OSCAR arm launched without the
    # window should stop before it holds a node, not 40 minutes in.  The same
    # reason oscar_quant_all_layers is checked at install time.
    if isinstance(tq_cfg, OscarFakeQuantConfig):
        from opqad.kernels.oscar_windowed_attn import require_for_oscar

        require_for_oscar()
    if skip_layers is None:
        # OSCAR quantizes every layer, unconditionally -- see
        # `oscar_quant_all_layers`, which is now a guard rather than a choice.
        # Calling it here is what makes a stale `VLLM_OSCAR_QUANT_ALL_LAYERS=0`
        # in someone's launcher fail at install time instead of silently
        # producing a skip-4 loss.
        #
        # Scoped to OSCAR on purpose: the boundary rule still applies to the
        # TurboQuant / INT4-PTKH arms, which were all measured under it.
        if isinstance(tq_cfg, OscarFakeQuantConfig):
            oscar_quant_all_layers()
            skip_layers = []
        else:
            skip_layers = skip_layer_indices(num_layers)

    current = getattr(text_cfg, "_attn_implementation", "sdpa")
    if base_impl is None:
        base_impl = "sdpa" if _is_qat_attn_name(current) else current
    if _is_qat_attn_name(base_impl):
        raise ValueError("base_impl cannot be the TurboQuant wrapper itself")

    state = TurboQuantAttnState(tq_cfg, skip_layers, base_impl, num_layers)
    # A capture-only install is a *persistent* property of that model, not a
    # per-forward one.  The frozen theta_0 teacher is installed this way so its
    # attention runs through the same wrapper (identical interception point)
    # with the quantizer permanently bypassed.  Without this the flag was only
    # ever set inside ``capture_only()``, so any *other* forward of the frozen
    # model -- notably block alignment's reference forward, which wraps nothing
    # because a frozen model needs no ``disabled()`` -- silently quantized its
    # own teacher and drove the block-align loss to exactly 0.
    state._capture_only = bool(capture_only)
    attn_name = f"{_ATTN_NAME}_{next(_ATTN_SEQ)}"
    state.attn_name = attn_name
    base_fn = ALL_ATTENTION_FUNCTIONS[base_impl]

    def _quantize(key, value, layer_idx):
        if key.shape[-1] != tq_cfg.head_dim:
            raise RuntimeError(
                f"layer {layer_idx}: head_dim {key.shape[-1]} != configured "
                f"{tq_cfg.head_dim}; the quantizer would group the wrong coordinates"
            )
        if isinstance(tq_cfg, OscarFakeQuantConfig):
            # Rotate -> group-wise INT2 -> un-rotate, matching the vLLM
            # oscar_int2 backend at this same interception point.  layer_idx
            # selects this layer's rotation: the zoo fits one matrix per layer,
            # and until it was threaded through here the quantizer resolved no
            # matrix at all and ran identity.
            return oscar_fake_quant(key, value, tq_cfg, layer_idx)

        if isinstance(tq_cfg, Int4PTKHConfig):
            # No Hadamard matrix or centroid table: the INT4 path rotates with
            # an in-place butterfly and quantizes on a uniform grid.
            return int4_ptkh_fake_quant(key, value, tq_cfg)
        h, centroids = state.buffers_for(key.device)
        return (
            turboquant_fake_quant_keys(key, tq_cfg, hadamard=h, centroids=centroids),
            turboquant_fake_quant_values(value, tq_cfg),
        )

    def _quantize_no_sink(key, value, layer_idx):
        """_quantize with sink protection off; the window mask handles it."""
        import dataclasses

        return oscar_fake_quant(
            key, value, dataclasses.replace(tq_cfg, sink_tokens=0), layer_idx
        )

    def _tq_attention(module, query, key, value, attention_mask, **kwargs):
        layer_idx = getattr(module, "layer_idx", None)
        # ``capture_only`` is the frozen-teacher mode: the quantizer is off AND
        # the raw post-RoPE K/V are recorded, so the reference forward yields
        # the full-precision K/V at the identical interception point.  It is
        # set only during a KV-alignment reference forward, never for training
        # or rollout, so a capture-only state cannot silently produce a
        # no-quantization actor.
        if state._capture_only:
            if layer_idx is not None:
                state.kv[int(layer_idx)] = (key, value)
            return base_fn(module, query, key, value, attention_mask, **kwargs)

        if not state.enabled or (
            layer_idx is not None and int(layer_idx) in state.skip_layers
        ):
            return base_fn(module, query, key, value, attention_mask, **kwargs)

        prefix = state.unquantized_prefix
        q_len = query.shape[-2]

        # OSCAR attends through its sink/recent window.  Not a knob -- see
        # oscar_windowed_attn.require_for_oscar, which is called once at install
        # time so an unset flag fails before a GPU-hour is spent.  k2v3 and the
        # INT4 schemes have no window to reproduce and are untouched.
        #
        # The other three conditions used to be silent fall-throughs into
        # `_quantize`, which applies sink_tokens as an ABSOLUTE column index.
        # Against verl's left-padded prompts that protects padding instead of
        # real tokens, so for OSCAR they now raise: a wrong objective that runs
        # is worse than a crash.
        if isinstance(tq_cfg, OscarFakeQuantConfig):
            if key.shape[-2] != q_len:
                raise NotImplementedError(
                    f"OSCAR windowed attention needs a self-attention forward with "
                    f"no cached keys, got q_len={q_len} kv_len={key.shape[-2]}. "
                    "Training never caches (use_cache=False), so this is a call "
                    "the window cannot serve -- refusing rather than falling back "
                    "to the absolute-position sink."
                )
            if state.record_kv:
                raise NotImplementedError(
                    "record_kv (the KV-alignment reference capture) with OSCAR: "
                    "the capture path quantizes through _quantize, whose sink is "
                    "an absolute column index and therefore wrong under left "
                    "padding. Set kv_align_coef=0 or capture from the windowed "
                    "path instead."
                )

        if isinstance(tq_cfg, OscarFakeQuantConfig):
            from opqad.kernels.oscar_windowed_attn import windowed_fake_quant_attention

            # sink is honoured by the mask here, so the quantizer must not also
            # honour it -- it would be applied twice, and the second time on the
            # wrong tensor (the one the mask never reads).
            kq, vq = _quantize_no_sink(key, value, layer_idx)
            G = query.shape[-3] // key.shape[-3]
            rep = (lambda t: t.repeat_interleave(G, dim=-3)) if G > 1 else (lambda t: t)
            state.note(layer_idx)
            out = windowed_fake_quant_attention(
                query, rep(key), rep(value), rep(kq), rep(vq),
                sink=tq_cfg.sink_tokens,
                recent=tq_cfg.recent_tokens,
                attention_mask=attention_mask,
                scale=module.scaling if hasattr(module, "scaling") else
                      query.shape[-1] ** -0.5,
            )
            # transformers' attention interface returns (out, weights) with the
            # output in [B, T, H, D]; sdpa's wrapper transposes, so match it.
            return out.transpose(1, 2).contiguous(), None

        if not (0 < prefix < q_len):
            key, value = _quantize(key, value, layer_idx)
            if state.record_kv and layer_idx is not None:
                state.kv[int(layer_idx)] = (key, value)
            state.note(layer_idx)
            return base_fn(module, query, key, value, attention_mask, **kwargs)

        # Split-query mode.  The prompt block is served exactly (raw K/V,
        # causal within the block); the response block reads the quantized
        # K/V of the whole sequence with an offset-causal mask.  Explicit
        # masks are required because the response block's query length no
        # longer equals its key length, so ``is_causal`` would mask the wrong
        # triangle.
        if key.shape[-2] != q_len:
            raise NotImplementedError(
                "unquantized_prefix requires a self-attention forward with no "
                f"cached keys, got q_len={q_len} kv_len={key.shape[-2]}"
            )

        q_key, q_value = _quantize(key, value, layer_idx)
        state.note(layer_idx)

        if attention_mask is None:
            # transformers hands the attention interface ``None`` when the
            # batch has no padding and the mask is purely causal; rebuild it
            # explicitly, because the response block's query length no longer
            # equals its key length and ``is_causal`` would mask the wrong
            # triangle.
            pos = torch.arange(q_len, device=query.device)
            prompt_mask = (pos[:prefix, None] >= pos[None, :prefix])[None, None]
            response_mask = (pos[prefix:, None] >= pos[None, :])[None, None]
        elif attention_mask.shape[-1] != q_len:
            raise NotImplementedError(
                "unquantized_prefix needs a square (q_len x q_len) attention "
                f"mask, got {tuple(attention_mask.shape)} for q_len={q_len}"
            )
        else:
            # Slicing the caller's mask preserves padding and any custom
            # masking, and stays correct for both boolean and additive masks.
            prompt_mask = attention_mask[..., :prefix, :prefix]
            response_mask = attention_mask[..., prefix:, :]

        prompt_out, _ = base_fn(
            module,
            query[..., :prefix, :],
            key[..., :prefix, :],
            value[..., :prefix, :],
            prompt_mask,
            **{**kwargs, "is_causal": False},
        )
        response_out, _ = base_fn(
            module,
            query[..., prefix:, :],
            q_key,
            q_value,
            response_mask,
            **{**kwargs, "is_causal": False},
        )
        # base_fn returns (batch, seq, heads, head_dim).
        return torch.cat([prompt_out, response_out], dim=1), None

    ALL_ATTENTION_FUNCTIONS.register(attn_name, _tq_attention)
    _register_mask_function(attn_name, base_impl)
    for cfg in _iter_configs(model):
        if hasattr(cfg, "_attn_implementation"):
            cfg._attn_implementation = attn_name

    setattr(model, "_qat_turboquant_state", state)
    return state


def _register_mask_function(attn_name: str, base_impl: str) -> None:
    """Give our attention name the same mask semantics as the one it wraps.

    transformers picks the mask builder by ``config._attn_implementation``:

        # masking_utils.py
        if config._attn_implementation not in ALL_MASK_ATTENTION_FUNCTIONS._global_mapping:
            return True, None, None, None, None

    The comment there reads "for TGI/vLLM backends, or other custom attention
    without equivalent mask creation: we don't need a mask!".  That is a
    reasonable default for a backend that builds its own masks -- and a silent
    disaster for a wrapper like this one, which just forwards whatever it is
    given to SDPA.  Without this registration the padding mask is dropped
    entirely: SDPA falls back to plain ``is_causal``, and every left-padded
    prompt token is attended to as if it were real text.

    That is not a cosmetic issue.  veRL left-pads every prompt to
    ``data.max_prompt_length``, so the actor was scoring tokens with ~1958
    pad tokens in context while the rollout engine saw none -- which is
    exactly the actor/rollout log-prob gap ``qat/rollout_logprob_bias``
    reports, and it scales with the amount of padding.
    """
    try:
        from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
    except Exception:  # pragma: no cover - older transformers without the registry
        return
    mapping = ALL_MASK_ATTENTION_FUNCTIONS._global_mapping
    if attn_name in mapping:
        return
    # Mirror the wrapped implementation; sdpa is the sane default because that
    # is what this wrapper ultimately calls.
    source = base_impl if base_impl in mapping else "sdpa"
    ALL_MASK_ATTENTION_FUNCTIONS.register(attn_name, mapping[source])


def get_installed_state(model: torch.nn.Module):
    """Find the installed state on a possibly-wrapped model.

    VeRL hands ``update_policy`` an FSDP-wrapped module, while the plugin
    installs on the inner HF model.  Walk the usual wrapper attributes rather
    than assuming a particular nesting depth; returns ``None`` when QAT is off,
    which is what keeps the alignment term fully opt-in.
    """
    seen = set()
    cur = model
    for _ in range(8):
        if cur is None or id(cur) in seen:
            break
        seen.add(id(cur))
        state = getattr(cur, "_qat_turboquant_state", None)
        if state is not None:
            return state
        cur = (
            getattr(cur, "_fsdp_wrapped_module", None)
            or getattr(cur, "module", None)
            or getattr(cur, "_orig_mod", None)
        )
    return None


def uninstall_turboquant_fake_quant(model: torch.nn.Module) -> bool:
    """Restore the model's original attention implementation."""
    state = getattr(model, "_qat_turboquant_state", None)
    if state is None:
        return False
    for cfg in _iter_configs(model):
        if _is_qat_attn_name(getattr(cfg, "_attn_implementation", None)):
            cfg._attn_implementation = state.base_impl
    delattr(model, "_qat_turboquant_state")
    return True
