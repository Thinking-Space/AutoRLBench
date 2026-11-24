"""Prove at runtime that a vLLM rollout really ran the requested TurboQuant cache.

Why this exists
---------------
``qat_kv.verl_plugin`` already has a ``rollout_kv_verified`` machinery, but it
only understands the *legacy* prototype: a hand-written Qwen2 fused-QKV
fake-dequant patch whose proof is "our patched forward was hit".  Native
TurboQuant is a completely different thing -- it is vLLM's own engine feature,
selected via ``--kv-cache-dtype``, and the legacy plugin is not even loaded on
that path (``VERL_USE_EXTERNAL_MODULES`` is emptied).  So the legacy verifier
can never fire, and every native run reported ``rollout_kv_verified=false``
while advertising ``rollout_kv_runtime_verification=required``.  This module
supplies the missing proof.

What counts as proof
--------------------
Checking that ``cache_config.cache_dtype`` echoes the requested string is not
enough: it only shows the engine parsed the flag.  The evidence collected here
is structural and much harder to fake:

1. the engine's resolved cache dtype equals the requested preset;
2. every attention layer that is *not* on vLLM's boundary-skip list holds a
   ``TurboQuantAttentionImpl`` instance;
3. every layer that *is* on that list does not;
4. those impls' ``tq_config`` bit-widths equal the preset table the training
   side quantizes with;
5. vLLM's boundary-skip set equals the one the training-side hook computed.

(5) is the important one for QAT: if inference skips {0,1,26,27} and training
fake-quantizes a different set, the student is trained for a quantizer it is
never deployed with -- silently, with no error and plausible-looking losses.

A structural check was chosen over instrumenting ``TurboQuantAttentionImpl.
forward`` with a hit counter on purpose.  Wrapping that method puts a Python
frame in the attention hot path and risks perturbing cudagraph capture, and it
would still undercount (captured decode graphs replay kernels without
re-entering Python).  If the running model's attention layers *are* TurboQuant
impls, generation cannot have gone through anything else.

Every check is fail-closed: an inconclusive traversal raises rather than
returning an optimistic verdict.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Optional

LOGGER = logging.getLogger(__name__)

TURBOQUANT_PREFIX = "turboquant_"
RECORD_ENV = "MOPD_ROLLOUT_KV_VERIFICATION_RECORD"

__all__ = [
    "RECORD_ENV",
    "TURBOQUANT_PREFIX",
    "collect_rollout_kv_evidence",
    "is_native_turboquant_dtype",
    "verify_int4_ptkh_rollout",
    "verify_native_quantized_rollout",
    "verify_native_turboquant_rollout",
    "write_verification_record",
]

_TRAILING_INT = re.compile(r"(\d+)")


def is_native_turboquant_dtype(value: Any) -> bool:
    return isinstance(value, str) and value.startswith(TURBOQUANT_PREFIX)


def _layer_index(name: str) -> Optional[int]:
    """Pull the decoder layer index out of a static-forward-context key.

    Keys look like ``model.layers.17.self_attn``.  vLLM's own
    ``extract_layer_index`` asserts there is exactly one integer in the name;
    this is the same rule, but returns ``None`` instead of raising so an
    unexpected key becomes recorded evidence rather than a crash.
    """
    ints = _TRAILING_INT.findall(name)
    if len(ints) != 1:
        return None
    return int(ints[0])


def _vllm_config(inference_engine: Any) -> Any:
    for holder in (
        getattr(inference_engine, "llm_engine", None),
        inference_engine,
    ):
        cfg = getattr(holder, "vllm_config", None)
        if cfg is not None:
            return cfg
    return None


def _cache_config(inference_engine: Any, vllm_config: Any) -> Any:
    if vllm_config is not None:
        cache_config = getattr(vllm_config, "cache_config", None)
        if cache_config is not None:
            return cache_config
    return getattr(getattr(inference_engine, "llm_engine", None), "cache_config", None)


def _attention_layers(vllm_config: Any) -> dict[str, Any]:
    """Return the engine's attention layer registry.

    ``compilation_config.static_forward_context`` is where every ``Attention``
    module registers itself at construction -- but it is populated in the
    process that *built the model*.  With
    ``distributed_executor_backend="external_launcher"`` (what veRL uses) that
    is this process.  With vLLM V1's default multiprocessing executor it is the
    EngineCore subprocess, and the registry seen here is empty; that is what
    ``worker_kv_evidence`` below is for.
    """
    compilation = getattr(vllm_config, "compilation_config", None)
    ctx = getattr(compilation, "static_forward_context", None) or {}
    return {name: layer for name, layer in ctx.items() if hasattr(layer, "impl")}


def worker_kv_evidence(worker: Any) -> dict[str, Any]:
    """Collect evidence inside a vLLM worker.  Must stay module-level.

    ``LLM.collective_rpc`` ships this callable to every worker, which may be a
    different process.  It therefore has to be importable there and must return
    something picklable -- hence plain dicts of ints and strings, never live
    module references.
    """
    from qat_kv.native_tq_verify import _collect_from_vllm_config, _layers_from_model

    vllm_config = getattr(worker, "vllm_config", None)
    cache_config = getattr(vllm_config, "cache_config", None)
    requested = str(getattr(cache_config, "cache_dtype", "")) if cache_config else ""
    evidence = _collect_from_vllm_config(vllm_config, requested)
    if not evidence["turboquant_layers"] and not evidence["non_turboquant_layers"]:
        # ``static_forward_context`` can be empty even in the worker (it is
        # keyed off whichever config object was current during model build).
        # Walking the constructed model is independent of that bookkeeping.
        model = getattr(getattr(worker, "model_runner", None), "model", None)
        layers = _layers_from_model(model)
        if layers:
            evidence = _collect_from_layers(layers, cache_config, requested)
            evidence["layer_source"] = "model_walk"
    return evidence


def _layers_from_model(model: Any) -> dict[str, Any]:
    """Find attention modules by walking the model.  Never raises."""
    if model is None:
        return {}
    try:
        named = list(model.named_modules())
    except Exception:  # pragma: no cover - exotic model wrappers
        return {}
    return {
        name: mod
        for name, mod in named
        if hasattr(mod, "impl") and getattr(mod, "impl", None) is not None
    }


def _collect_from_vllm_config(vllm_config: Any, requested_dtype: str) -> dict[str, Any]:
    """Evidence from one process's view of the engine.  Never raises."""

    cache_config = getattr(vllm_config, "cache_config", None)
    evidence = _collect_from_layers(
        _attention_layers(vllm_config), cache_config, requested_dtype
    )
    evidence["layer_source"] = "static_forward_context"
    return evidence


def _collect_from_layers(
    layers: dict[str, Any], cache_config: Any, requested_dtype: str
) -> dict[str, Any]:
    """Classify a set of attention modules.  Never raises."""

    runtime_dtype = getattr(cache_config, "cache_dtype", None)
    raw_skip = list(getattr(cache_config, "kv_cache_dtype_skip_layers", None) or ())

    numeric_skip: list[int] = []
    non_numeric_skip: list[str] = []
    for entry in raw_skip:
        text = str(entry)
        if text.lstrip("-").isdigit():
            numeric_skip.append(int(text))
        else:
            non_numeric_skip.append(text)

    turboquant_layers: list[int] = []
    plain_layers: list[int] = []
    unindexed: list[str] = []
    impl_classes: set[str] = set()
    tq_configs: set[tuple] = set()
    # Evidence for presets that do *not* get their own impl class.  int4 rides
    # inside TritonAttentionImpl (vLLM 0.25.1 lists it in
    # ``TritonAttentionBackend.supported_kv_cache_dtypes``), so "which class is
    # this layer" cannot separate quantized from skipped layers.  What can is
    # ``Attention.kv_cache_dtype``: the layer ctor copies the cache dtype into
    # it and overwrites it with "auto" for every layer named in
    # ``kv_cache_dtype_skip_layers``.  That is the same per-layer partition the
    # TurboQuant path proves via impl classes, read off a different attribute.
    layer_kv_dtypes: list[list] = []
    head_sizes: set[int] = set()
    all_impl_classes: set[str] = set()

    for name, layer in sorted(layers.items()):
        impl = getattr(layer, "impl", None)
        cls = type(impl)
        qualname = f"{cls.__module__}.{cls.__qualname__}"
        idx = _layer_index(name)
        if idx is None:
            unindexed.append(name)
            continue
        all_impl_classes.add(qualname)
        layer_dtype = getattr(layer, "kv_cache_dtype", None)
        # Lists, not a dict: this record is both pickled across ``collective_rpc``
        # and json-dumped, and json turns int keys into strings.
        layer_kv_dtypes.append([idx, None if layer_dtype is None else str(layer_dtype)])
        head_size = getattr(layer, "head_size", None)
        if head_size is None:
            head_size = getattr(impl, "head_size", None)
        if isinstance(head_size, int):
            head_sizes.add(head_size)
        if "TurboQuant" in cls.__name__:
            impl_classes.add(qualname)
            turboquant_layers.append(idx)
            tq_config = getattr(impl, "tq_config", None)
            tq_configs.add(
                (
                    getattr(tq_config, "head_dim", None),
                    getattr(tq_config, "key_quant_bits", None),
                    getattr(tq_config, "value_quant_bits", None),
                    bool(getattr(tq_config, "norm_correction", False)),
                )
            )
        else:
            plain_layers.append(idx)

    evidence: dict[str, Any] = {
        "requested_kv_cache_dtype": requested_dtype,
        "runtime_kv_cache_dtype": (
            str(runtime_dtype) if runtime_dtype is not None else None
        ),
        "runtime_skip_layers": sorted(numeric_skip),
        "runtime_skip_layers_non_numeric": sorted(non_numeric_skip),
        "attention_layers_seen": len(layers),
        "turboquant_layers": sorted(turboquant_layers),
        "non_turboquant_layers": sorted(plain_layers),
        "attention_layers_without_index": sorted(unindexed),
        "layer_kv_cache_dtypes": sorted(layer_kv_dtypes, key=lambda p: p[0]),
        "head_sizes": sorted(head_sizes),
        "impl_classes": sorted(all_impl_classes),
        "turboquant_impl_classes": sorted(impl_classes),
        "turboquant_configs": [
            {
                "head_dim": head_dim,
                "key_quant_bits": key_bits,
                "value_quant_bits": value_bits,
                "norm_correction": nc,
            }
            for head_dim, key_bits, value_bits, nc in sorted(
                tq_configs, key=lambda c: tuple(-1 if x is None else int(x) for x in c)
            )
        ],
    }
    return evidence


def collect_rollout_kv_evidence(
    inference_engine: Any, requested_dtype: str
) -> dict[str, Any]:
    """Gather runtime facts about the constructed engine.  Never raises.

    Asks the workers first via ``collective_rpc`` -- that is the only view that
    is correct under vLLM V1's default multiprocessing executor, where the
    model is built in an EngineCore subprocess and this process's layer
    registry is empty.  Falls back to the in-process registry (veRL's
    ``external_launcher`` mode, and any engine without ``collective_rpc``).
    """

    evidence: dict[str, Any] | None = None
    rpc_error: str | None = None
    num_ranks = None

    rpc = getattr(inference_engine, "collective_rpc", None)
    if callable(rpc):
        try:
            per_rank = rpc(worker_kv_evidence)
            per_rank = [r for r in (per_rank or ()) if isinstance(r, dict)]
            if per_rank:
                num_ranks = len(per_rank)
                # Tensor parallelism shards heads, not layers, so every rank
                # must see the same layer classification.  A disagreement is
                # itself a finding, so surface it instead of picking rank 0.
                keys = (
                    "turboquant_layers",
                    "non_turboquant_layers",
                    "turboquant_configs",
                    # int4's partition lives here rather than in the layer
                    # lists, so omitting it would let ranks disagree about the
                    # int4 skip set without the check noticing.
                    "layer_kv_cache_dtypes",
                )
                if any(
                    any(r.get(k) != per_rank[0].get(k) for k in keys) for r in per_rank[1:]
                ):
                    rpc_error = "vLLM ranks disagree about which layers are quantized"
                else:
                    evidence = dict(per_rank[0])
        except Exception as exc:  # pragma: no cover - engine dependent
            rpc_error = f"{type(exc).__name__}: {exc}"
            LOGGER.debug("collective_rpc evidence failed", exc_info=True)

    if evidence is None:
        evidence = _collect_from_vllm_config(
            _vllm_config(inference_engine), requested_dtype
        )
        evidence["evidence_source"] = "in_process_forward_context"
    else:
        evidence["evidence_source"] = "worker_collective_rpc"

    # The caller asked for this preset; the worker only knows what the engine
    # resolved.  Keep the request authoritative so a silent rename is caught.
    evidence["requested_kv_cache_dtype"] = requested_dtype
    evidence["num_ranks"] = num_ranks
    if rpc_error:
        evidence["collective_rpc_error"] = rpc_error
    return evidence


def verify_native_turboquant_rollout(
    inference_engine: Any,
    requested_dtype: str,
    *,
    num_hidden_layers: Optional[int] = None,
) -> dict[str, Any]:
    """Verify the engine, or raise ``RuntimeError`` explaining what is missing.

    Returns the evidence record with ``rollout_kv_verified`` set to ``True``.
    """

    from .turboquant_fake_quant import TQ_FAKE_QUANT_PRESETS, skip_layer_indices

    record = collect_rollout_kv_evidence(inference_engine, requested_dtype)
    failures: list[str] = []

    runtime_dtype = record["runtime_kv_cache_dtype"]
    if runtime_dtype is None or runtime_dtype.lower() != requested_dtype.lower():
        failures.append(
            f"engine cache dtype is {runtime_dtype!r}, requested {requested_dtype!r}"
        )

    if not record["turboquant_layers"]:
        failures.append(
            "no attention layer holds a TurboQuantAttentionImpl "
            f"({record['attention_layers_seen']} attention layers inspected)"
        )
    if record["attention_layers_without_index"]:
        failures.append(
            "attention layers with unparseable indices: "
            f"{record['attention_layers_without_index']}"
        )

    # vLLM's own skip list must be exactly the set of layers left unquantized.
    runtime_skip = set(record["runtime_skip_layers"])
    plain = set(record["non_turboquant_layers"])
    if plain != runtime_skip:
        failures.append(
            f"layers running in full precision {sorted(plain)} != "
            f"engine skip list {sorted(runtime_skip)}"
        )

    # And the training-side hook must have computed the same skip set, or the
    # student is fake-quantizing layers that run in fp at inference (or worse,
    # leaving quantized layers unmodelled).
    total_layers = num_hidden_layers
    if total_layers is None:
        # Prefer the model config over the number of registered attention
        # layers: a hybrid or partially-registered model would make the
        # layer count silently wrong, and the skip set derived from it too.
        model_config = getattr(_vllm_config(inference_engine), "model_config", None)
        hf_text_config = getattr(model_config, "hf_text_config", None)
        total_layers = getattr(hf_text_config, "num_hidden_layers", None)
    if total_layers is None and record["attention_layers_seen"]:
        total_layers = record["attention_layers_seen"]
    if total_layers:
        training_skip = set(skip_layer_indices(int(total_layers)))
        record["training_skip_layers"] = sorted(training_skip)
        record["num_hidden_layers"] = int(total_layers)
        if training_skip != runtime_skip:
            failures.append(
                f"training-side skip layers {sorted(training_skip)} != engine skip "
                f"list {sorted(runtime_skip)} for {total_layers} layers"
            )
    else:
        failures.append("could not determine the model's attention layer count")

    preset = TQ_FAKE_QUANT_PRESETS.get(requested_dtype)
    if preset is None:
        failures.append(
            f"{requested_dtype!r} has no training-side preset; the actor cannot "
            "fake-quantize what the rollout is running"
        )
    else:
        configs = record["turboquant_configs"]
        if len(configs) != 1:
            failures.append(f"attention layers disagree on TurboQuant config: {configs}")
        elif (
            configs[0]["key_quant_bits"] != preset["key_quant_bits"]
            or configs[0]["value_quant_bits"] != preset["value_quant_bits"]
            or configs[0]["norm_correction"] != preset["norm_correction"]
        ):
            failures.append(
                f"engine TurboQuant config {configs[0]} != training preset {preset}"
            )

    record["failures"] = failures
    record["rollout_kv_verified"] = not failures
    if failures:
        # Say where the evidence came from.  "0 layers inspected" is
        # indistinguishable between "the engine is wrong" and "we looked in the
        # wrong process", and those need opposite responses.
        provenance = (
            f"evidence_source={record.get('evidence_source')!r}, "
            f"layer_source={record.get('layer_source')!r}, "
            f"num_ranks={record.get('num_ranks')!r}"
        )
        if record.get("collective_rpc_error"):
            provenance += f", collective_rpc_error={record['collective_rpc_error']!r}"
        raise RuntimeError(
            "native TurboQuant rollout verification failed: "
            + "; ".join(failures)
            + f" [{provenance}]"
        )
    return record


def verify_int4_ptkh_rollout(
    inference_engine: Any,
    requested_dtype: str,
    *,
    num_hidden_layers: Optional[int] = None,
) -> dict[str, Any]:
    """``verify_native_turboquant_rollout``'s counterpart for ``int4_per_token_head``.

    The TurboQuant proof cannot be reused verbatim.  TurboQuant ships its own
    ``TurboQuantAttentionImpl``, so "does this layer hold that class" separates
    quantized from skipped layers.  int4 has no such class: vLLM 0.25.1 runs it
    inside the ordinary ``TritonAttentionImpl``, selected by the cache dtype
    alone.  The equivalent structural signal is ``Attention.kv_cache_dtype``,
    which the layer ctor sets to the cache dtype and forces to ``"auto"`` for
    every layer listed in ``kv_cache_dtype_skip_layers``.

    The check that matters is the same one as for TurboQuant: the set of layers
    the engine leaves in full precision must equal the set the training-side
    hook declines to fake-quantize.  A mismatch trains the student for a
    quantizer it is never served with, silently.
    """

    from .int4_ptkh_fake_quant import Int4PTKHConfig, is_int4_ptkh_scheme
    from .turboquant_fake_quant import skip_layer_indices

    if not is_int4_ptkh_scheme(requested_dtype):
        raise ValueError(
            f"{requested_dtype!r} is not an int4 per-token-head preset; "
            "this verifier does not apply to it"
        )

    record = collect_rollout_kv_evidence(inference_engine, requested_dtype)
    failures: list[str] = []

    runtime_dtype = record["runtime_kv_cache_dtype"]
    if runtime_dtype is None or runtime_dtype.lower() != requested_dtype.lower():
        failures.append(
            f"engine cache dtype is {runtime_dtype!r}, requested {requested_dtype!r}"
        )

    if record["attention_layers_without_index"]:
        failures.append(
            "attention layers with unparseable indices: "
            f"{record['attention_layers_without_index']}"
        )

    # Partition the layers by what each one says its own cache dtype is.
    pairs = record.get("layer_kv_cache_dtypes") or []
    if not pairs:
        failures.append(
            "no attention layer reported a kv_cache_dtype "
            f"({record['attention_layers_seen']} attention layers inspected)"
        )
    quantized = {idx for idx, dt in pairs if dt is not None and dt.lower() == requested_dtype.lower()}
    unquantized = {idx for idx, dt in pairs if idx not in quantized}
    record["int4_layers"] = sorted(quantized)
    record["non_int4_layers"] = sorted(unquantized)
    # Anything neither int4 nor the plain "auto" of a skipped layer means a
    # third dtype crept in, which no part of this pipeline models.
    unexpected = sorted(
        (idx, dt) for idx, dt in pairs
        if idx not in quantized and (dt is None or dt.lower() != "auto")
    )
    if unexpected:
        failures.append(f"layers with an unexpected kv_cache_dtype: {unexpected}")
    if pairs and not quantized:
        failures.append(
            f"no attention layer runs {requested_dtype!r}; layer dtypes were {pairs}"
        )

    runtime_skip = set(record["runtime_skip_layers"])
    if unquantized != runtime_skip:
        failures.append(
            f"layers running in full precision {sorted(unquantized)} != "
            f"engine skip list {sorted(runtime_skip)}"
        )

    total_layers = num_hidden_layers
    if total_layers is None:
        model_config = getattr(_vllm_config(inference_engine), "model_config", None)
        hf_text_config = getattr(model_config, "hf_text_config", None)
        total_layers = getattr(hf_text_config, "num_hidden_layers", None)
    if total_layers is None and record["attention_layers_seen"]:
        total_layers = record["attention_layers_seen"]
    if total_layers:
        training_skip = set(skip_layer_indices(int(total_layers)))
        record["training_skip_layers"] = sorted(training_skip)
        record["num_hidden_layers"] = int(total_layers)
        if training_skip != runtime_skip:
            failures.append(
                f"training-side skip layers {sorted(training_skip)} != engine skip "
                f"list {sorted(runtime_skip)} for {total_layers} layers"
            )
    else:
        failures.append("could not determine the model's attention layer count")

    # The rotation is over head_size, so a head_dim the training config refuses
    # (non power of two) or disagrees on would be a different quantizer.
    head_sizes = record.get("head_sizes") or []
    if len(head_sizes) != 1:
        failures.append(f"attention layers disagree on head_size: {head_sizes}")
    else:
        try:
            Int4PTKHConfig.from_preset(requested_dtype, int(head_sizes[0]))
        except Exception as exc:
            failures.append(
                f"engine head_size={head_sizes[0]} is not usable by the training-side "
                f"{requested_dtype} quantizer: {type(exc).__name__}: {exc}"
            )

    record["failures"] = failures
    record["rollout_kv_verified"] = not failures
    if failures:
        provenance = (
            f"evidence_source={record.get('evidence_source')!r}, "
            f"layer_source={record.get('layer_source')!r}, "
            f"num_ranks={record.get('num_ranks')!r}"
        )
        if record.get("collective_rpc_error"):
            provenance += f", collective_rpc_error={record['collective_rpc_error']!r}"
        raise RuntimeError(
            "native int4 rollout verification failed: "
            + "; ".join(failures)
            + f" [{provenance}]"
        )
    return record


def verify_native_quantized_rollout(
    inference_engine: Any,
    requested_dtype: str,
    *,
    num_hidden_layers: Optional[int] = None,
) -> dict[str, Any]:
    """Dispatch to the verifier that matches ``requested_dtype``.

    Callers should use this rather than testing the preset name themselves.
    ``vllm_rollout_spmd`` gated the whole verification block on
    ``dtype.startswith("turboquant_")``, which meant job 12091 ran a full int4
    gate with the engine proof silently skipped and still reported success --
    a fail-open in the one check that exists to prevent silent mismatches.
    An unknown preset raises: there is no safe default here.
    """

    from .int4_ptkh_fake_quant import is_int4_ptkh_scheme

    if is_native_turboquant_dtype(requested_dtype):
        return verify_native_turboquant_rollout(
            inference_engine, requested_dtype, num_hidden_layers=num_hidden_layers
        )
    if is_int4_ptkh_scheme(requested_dtype):
        return verify_int4_ptkh_rollout(
            inference_engine, requested_dtype, num_hidden_layers=num_hidden_layers
        )
    raise ValueError(
        f"no rollout KV verifier for {requested_dtype!r}; refusing to run a "
        "quantized rollout whose engine cannot be proven to match the actor"
    )


def write_verification_record(
    record: dict[str, Any], path: Optional[str] = None
) -> Optional[str]:
    """Persist the record so a finished run can be audited after the fact.

    Best effort: a run must not die because a log directory is unwritable.
    """
    target = path or os.environ.get(RECORD_ENV) or ""
    if not target:
        return None
    try:
        out = Path(target)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    except Exception:  # pragma: no cover - filesystem dependent
        LOGGER.warning("could not write KV verification record to %s", target, exc_info=True)
        return None
    return str(target)
