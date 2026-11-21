#!/usr/bin/env python3
"""Compare full-precision and LLM-QAT KV fake-quantized HF execution.

This is a small *diagnostic* runner, not a training or serving path.  It
deliberately reuses :func:`qat_kv.attach_kv_fake_quant_hooks` and the
repository's ``models.utils_quant.SymQuantizer`` instead of implementing a
second quantizer.  A fixed prompt is run through a prefill call followed by
one-token incremental calls.  For each quantized mode the full-precision and
quantized calls are interleaved, so their cache trajectories never need to be
copied to CPU or serialized.

The report contains:

* model/quantizer/placement metadata (including the fact that this is fake
  dequantization, not a packed integer cache);
* per-token final-logit error, top-1 agreement and top-k overlap; and
* per-layer, per-token K/V cache error for both prefill and decode.

The default command uses a tiny prompt and greedy tokens selected from the
full-precision prefill.  Supplying ``--decode-ids`` makes the trajectory fully
reproducible across invocations.  ``--modes fp,2,4`` is useful for the requested
FP/KV2/KV4 gate, while ``--modes 4`` avoids running the reference three times.

Examples::

    python scripts/kv_parity.py --model /path/to/qwen2.5-7b-instruct \
        --prompt 'Solve 1+1.' --prefill-tokens 128 --decode-tokens 4 \
        --modes fp,2,4 --preserve-projection-bias \
        --output /historical/shared/qat/r/kv-parity.json

No model is loaded when this module is imported; this keeps the metric helpers
usable in CPU-only unit tests.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import platform
import re
import time
from types import SimpleNamespace
from typing import Any, Optional

import torch


DEFAULT_PROMPT = "Solve the following problem: What is 1 + 1?"
DEFAULT_CLIP_RANGE = (-2.0, 2.0)


@dataclass(frozen=True)
class ParityOptions:
    """Runtime choices that affect a parity report."""

    modes: tuple[int, ...] = (0, 2, 4)
    prefill_tokens: int = 0
    decode_tokens: int = 4
    preserve_projection_bias: bool = True
    layerwise: bool = False
    clip_range: tuple[float, float] = DEFAULT_CLIP_RANGE
    cache_chunk_size: int = 256
    logit_chunk_size: int = 8
    top_k: int = 10
    decode_ids: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not self.modes:
            raise ValueError("modes must not be empty")
        if any(mode not in (0, 2, 4) for mode in self.modes):
            raise ValueError(f"modes must contain only 0, 2, or 4, got {self.modes}")
        if self.prefill_tokens < 0:
            raise ValueError("prefill_tokens must be non-negative")
        if self.decode_tokens < 0:
            raise ValueError("decode_tokens must be non-negative")
        if self.cache_chunk_size < 1 or isinstance(self.cache_chunk_size, bool):
            raise ValueError("cache_chunk_size must be a positive integer")
        if self.logit_chunk_size < 1 or isinstance(self.logit_chunk_size, bool):
            raise ValueError("logit_chunk_size must be a positive integer")
        if self.top_k < 1 or isinstance(self.top_k, bool):
            raise ValueError("top_k must be a positive integer")
        if len(self.clip_range) != 2:
            raise ValueError("clip_range must contain two values")
        lo, hi = (float(self.clip_range[0]), float(self.clip_range[1]))
        if not math.isfinite(lo) or not math.isfinite(hi) or lo >= hi:
            raise ValueError("clip_range must be finite and min < max")
        object.__setattr__(self, "clip_range", (lo, hi))
        if self.decode_ids and len(self.decode_ids) != self.decode_tokens:
            raise ValueError(
                "decode_ids length must equal decode_tokens when supplied"
            )


def _device_of_model(model: Any) -> torch.device:
    """Find the input device without requiring a Transformers class."""

    get_embeddings = getattr(model, "get_input_embeddings", None)
    if callable(get_embeddings):
        try:
            embedding = get_embeddings()
            weight = getattr(embedding, "weight", None)
            if isinstance(weight, torch.Tensor):
                return weight.device
        except Exception:
            pass
    try:
        parameter = next(model.parameters())
        return parameter.device
    except (AttributeError, StopIteration):
        return torch.device("cpu")


def _dtype_name(model: Any) -> str:
    try:
        parameter = next(model.parameters())
        return str(parameter.dtype).replace("torch.", "")
    except (AttributeError, StopIteration):
        return "unknown"


def _as_tensor(value: Any, *, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(value)!r}")
    return value


def _legacy_cache(cache: Any) -> Optional[Sequence[Any]]:
    """Normalize HF legacy and DynamicCache objects to a layer sequence."""

    if cache is None:
        return None
    converter = getattr(cache, "to_legacy_cache", None)
    if callable(converter):
        try:
            converted = converter()
            if converted is not None:
                cache = converted
        except Exception:
            # Some cache implementations intentionally do not expose a legacy
            # view.  The structural paths below still cover them.
            pass
    if hasattr(cache, "layers"):
        try:
            layers = getattr(cache, "layers")
            result = []
            for layer in layers:
                key = getattr(layer, "keys", None)
                value = getattr(layer, "values", None)
                result.append((key, value))
            return result
        except Exception:
            pass
    if hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
        try:
            return list(zip(cache.key_cache, cache.value_cache))
        except Exception:
            pass
    if isinstance(cache, (tuple, list)):
        return cache
    raise TypeError(
        "unsupported HF cache type; expected DynamicCache/legacy tuple, got "
        f"{type(cache)!r}"
    )


def cache_pairs(cache: Any) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Return materialized ``(key, value)`` tensors from an HF cache.

    This helper intentionally rejects malformed layers rather than silently
    reporting a zero error.  A few cache implementations use ``None`` for an
    uninitialized sliding-window layer; those layers are skipped by the model
    itself and are represented as an empty list only when the entire cache is
    empty.
    """

    layers = _legacy_cache(cache)
    if layers is None:
        return []
    pairs: list[tuple[torch.Tensor, torch.Tensor]] = []
    for index, layer in enumerate(layers):
        if not isinstance(layer, (tuple, list)) or len(layer) < 2:
            raise TypeError(f"cache layer {index} is not a (key, value) pair")
        key, value = layer[0], layer[1]
        if key is None or value is None:
            continue
        _as_tensor(key, name=f"cache[{index}].key")
        _as_tensor(value, name=f"cache[{index}].value")
        if key.ndim < 2 or value.ndim < 2:
            raise ValueError(
                f"cache[{index}] tensors must have at least 2 dimensions, "
                f"got {key.shape} and {value.shape}"
            )
        if key.shape[:-2] != value.shape[:-2] or key.shape[-2] != value.shape[-2]:
            raise ValueError(
                f"cache[{index}] key/value sequence shape mismatch: "
                f"{tuple(key.shape)} vs {tuple(value.shape)}"
            )
        pairs.append((key, value))
    return pairs


def cache_seq_len(cache: Any) -> int:
    """Get the current sequence length from a cache without materializing it."""

    if cache is None:
        return 0
    getter = getattr(cache, "get_seq_length", None)
    if callable(getter):
        try:
            return int(getter())
        except Exception:
            pass
    pairs = cache_pairs(cache)
    return int(pairs[0][0].shape[-2]) if pairs else 0


def _finite_stats(diff: torch.Tensor, *, count: int) -> dict[str, Any]:
    """Summarize a float difference tensor without retaining activations."""

    finite = bool(torch.isfinite(diff).all().item())
    safe = diff if finite else torch.nan_to_num(diff, nan=0.0, posinf=0.0, neginf=0.0)
    abs_diff = safe.abs()
    return {
        "max_abs": float(abs_diff.max().item()) if abs_diff.numel() else 0.0,
        "mean_abs": float(abs_diff.sum().item() / max(1, count)),
        "rmse": float(torch.sqrt((safe * safe).sum() / max(1, count)).item()),
        "finite": finite,
    }


def tensor_error_stats(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    *,
    token_axis: int = -2,
    chunk_size: int = 256,
    token_offset: int = 0,
) -> dict[str, Any]:
    """Compute overall and compact per-token absolute-error statistics.

    The token axis defaults to ``-2``, matching HF K/V tensors shaped
    ``[batch, heads, sequence, head_dim]``.  Logits callers can pass axis 1.
    Chunks are converted to FP32 one at a time, which avoids a second full
    FP32 copy of a 16K context cache.
    """

    reference = _as_tensor(reference, name="reference")
    candidate = _as_tensor(candidate, name="candidate")
    if tuple(reference.shape) != tuple(candidate.shape):
        raise ValueError(
            f"reference/candidate shape mismatch: {tuple(reference.shape)} vs "
            f"{tuple(candidate.shape)}"
        )
    if not reference.is_floating_point() or not candidate.is_floating_point():
        raise TypeError("error statistics require floating-point tensors")
    axis = token_axis if token_axis >= 0 else reference.ndim + token_axis
    if axis < 0 or axis >= reference.ndim:
        raise ValueError(f"token_axis={token_axis} is invalid for rank {reference.ndim}")
    token_count = int(reference.shape[axis])
    if chunk_size < 1 or isinstance(chunk_size, bool):
        raise ValueError("chunk_size must be a positive integer")

    overall_count = 0
    overall_sum = 0.0
    overall_sq = 0.0
    overall_max = 0.0
    finite = True
    per_token_max: list[float] = []
    per_token_mean: list[float] = []
    per_token_rmse: list[float] = []

    for start in range(0, token_count, chunk_size):
        width = min(chunk_size, token_count - start)
        ref_chunk = reference.narrow(axis, start, width).float()
        cand_chunk = candidate.narrow(axis, start, width).float()
        diff = cand_chunk - ref_chunk
        perm = diff.movedim(axis, -2)
        feature = int(perm.shape[-1]) if perm.ndim else 1
        rows = perm.reshape(-1, width, feature)
        abs_rows = rows.abs()
        sq_rows = rows * rows
        token_numel = rows.shape[0] * feature
        per_token_max.extend(abs_rows.amax(dim=(0, 2)).tolist())
        per_token_mean.extend((abs_rows.sum(dim=(0, 2)) / max(1, token_numel)).tolist())
        per_token_rmse.extend(
            torch.sqrt(sq_rows.sum(dim=(0, 2)) / max(1, token_numel)).tolist()
        )
        count = int(diff.numel())
        finite_chunk = bool(torch.isfinite(diff).all().item())
        finite = finite and finite_chunk
        safe = diff if finite_chunk else torch.nan_to_num(diff, nan=0.0, posinf=0.0, neginf=0.0)
        overall_count += count
        overall_sum += float(safe.abs().sum().item())
        overall_sq += float((safe * safe).sum().item())
        if count:
            overall_max = max(overall_max, float(safe.abs().max().item()))

    return {
        "shape": list(reference.shape),
        "max_abs": overall_max,
        "mean_abs": overall_sum / max(1, overall_count),
        "rmse": math.sqrt(overall_sq / max(1, overall_count)),
        "finite": finite,
        "token_indices": list(range(int(token_offset), int(token_offset) + token_count)),
        "per_token_max_abs": per_token_max,
        "per_token_mean_abs": per_token_mean,
        "per_token_rmse": per_token_rmse,
    }


def cache_error_report(
    reference_cache: Any,
    candidate_cache: Any,
    *,
    token_start: int = 0,
    token_end: Optional[int] = None,
    chunk_size: int = 256,
) -> dict[str, Any]:
    """Compare corresponding K/V cache slices layer by layer."""

    reference_pairs = cache_pairs(reference_cache)
    candidate_pairs = cache_pairs(candidate_cache)
    if len(reference_pairs) != len(candidate_pairs):
        raise ValueError(
            "reference/candidate cache layer count mismatch: "
            f"{len(reference_pairs)} vs {len(candidate_pairs)}"
        )
    if not reference_pairs:
        return {"layer_count": 0, "layers": [], "token_start": token_start, "token_end": token_end}

    full_length = int(reference_pairs[0][0].shape[-2])
    if token_end is None:
        token_end = full_length
    if not (0 <= token_start <= token_end <= full_length):
        raise ValueError(
            f"invalid cache token range [{token_start}, {token_end}) for length {full_length}"
        )
    layers: list[dict[str, Any]] = []
    for index, ((ref_k, ref_v), (cand_k, cand_v)) in enumerate(
        zip(reference_pairs, candidate_pairs)
    ):
        if tuple(ref_k.shape) != tuple(cand_k.shape) or tuple(ref_v.shape) != tuple(cand_v.shape):
            raise ValueError(
                f"cache layer {index} shape mismatch: "
                f"K {tuple(ref_k.shape)} vs {tuple(cand_k.shape)}, "
                f"V {tuple(ref_v.shape)} vs {tuple(cand_v.shape)}"
            )
        ref_k_slice = ref_k.narrow(-2, token_start, token_end - token_start)
        cand_k_slice = cand_k.narrow(-2, token_start, token_end - token_start)
        ref_v_slice = ref_v.narrow(-2, token_start, token_end - token_start)
        cand_v_slice = cand_v.narrow(-2, token_start, token_end - token_start)
        layers.append(
            {
                "layer": index,
                "key": tensor_error_stats(
                    ref_k_slice,
                    cand_k_slice,
                    token_axis=-2,
                    chunk_size=chunk_size,
                    token_offset=token_start,
                ),
                "value": tensor_error_stats(
                    ref_v_slice,
                    cand_v_slice,
                    token_axis=-2,
                    chunk_size=chunk_size,
                    token_offset=token_start,
                ),
            }
        )
    return {
        "layer_count": len(layers),
        "cache_length": full_length,
        "token_start": token_start,
        "token_end": token_end,
        "layers": layers,
    }


def logit_error_report(
    reference_logits: torch.Tensor,
    candidate_logits: torch.Tensor,
    *,
    token_offset: int = 0,
    chunk_size: int = 8,
    top_k: int = 10,
) -> dict[str, Any]:
    """Compare final logits in FP32, including top-k agreement."""

    reference_logits = _as_tensor(reference_logits, name="reference_logits")
    candidate_logits = _as_tensor(candidate_logits, name="candidate_logits")
    if tuple(reference_logits.shape) != tuple(candidate_logits.shape):
        raise ValueError(
            "reference/candidate logits shape mismatch: "
            f"{tuple(reference_logits.shape)} vs {tuple(candidate_logits.shape)}"
        )
    if reference_logits.ndim < 2:
        raise ValueError("logits must have at least [tokens, vocab] dimensions")
    vocab = int(reference_logits.shape[-1])
    token_axis = reference_logits.ndim - 2
    token_count = int(reference_logits.shape[token_axis])
    if top_k < 1 or chunk_size < 1 or isinstance(chunk_size, bool):
        raise ValueError("top_k and chunk_size must be positive integers")
    k = min(int(top_k), vocab)
    overall = tensor_error_stats(
        reference_logits,
        candidate_logits,
        token_axis=token_axis,
        chunk_size=chunk_size,
        token_offset=token_offset,
    )
    top1_matches: list[bool] = []
    topk_overlaps: list[float] = []
    kl_values: list[float] = []
    # Flatten all leading dimensions except token/vocabulary.  HF causal LM
    # output is normally [B,T,V], and this also handles a tiny mock [T,V].
    ref = reference_logits.movedim(token_axis, -2).reshape(-1, token_count, vocab)
    cand = candidate_logits.movedim(token_axis, -2).reshape(-1, token_count, vocab)
    for start in range(0, token_count, chunk_size):
        width = min(chunk_size, token_count - start)
        ref_chunk = ref[:, start : start + width, :].float()
        cand_chunk = cand[:, start : start + width, :].float()
        ref_top = torch.topk(ref_chunk, k=k, dim=-1).indices
        cand_top = torch.topk(cand_chunk, k=k, dim=-1).indices
        top1_matches.extend((ref_top[..., 0] == cand_top[..., 0]).reshape(-1).tolist())
        overlap = (
            ref_top.unsqueeze(-1).eq(cand_top.unsqueeze(-2)).any(dim=-1).float().mean(dim=-1)
        )
        topk_overlaps.extend(overlap.reshape(-1).tolist())
        # KL is computed in FP32 but never retained as a vocabulary-sized
        # tensor beyond this chunk.  It is useful for very small logit shifts
        # that top-k agreement alone can hide.
        ref_logp = torch.log_softmax(ref_chunk, dim=-1)
        cand_logp = torch.log_softmax(cand_chunk, dim=-1)
        kl = (ref_logp.exp() * (ref_logp - cand_logp)).sum(dim=-1)
        kl_values.extend(kl.reshape(-1).tolist())
    overall.update(
        {
            "vocab_size": vocab,
            "top_k": k,
            "top1_match_rate": sum(bool(x) for x in top1_matches) / max(1, len(top1_matches)),
            "topk_overlap_mean": sum(float(x) for x in topk_overlaps) / max(1, len(topk_overlaps)),
            "kl_ref_to_candidate_mean": sum(float(x) for x in kl_values) / max(1, len(kl_values)),
            "per_token_top1_match": [bool(x) for x in top1_matches],
            "per_token_topk_overlap": [float(x) for x in topk_overlaps],
            "per_token_kl_ref_to_candidate": [float(x) for x in kl_values],
        }
    )
    return overall


def _output_logits(output: Any) -> torch.Tensor:
    logits = getattr(output, "logits", None)
    if logits is None and isinstance(output, dict):
        logits = output.get("logits")
    if logits is None and isinstance(output, (tuple, list)):
        logits = output[0]
    return _as_tensor(logits, name="model output logits")


def _output_cache(output: Any) -> Any:
    for name in ("past_key_values", "past_key_value", "cache_params"):
        cache = getattr(output, name, None)
        if cache is None and isinstance(output, dict):
            cache = output.get(name)
        if cache is not None:
            return cache
    if isinstance(output, (tuple, list)) and len(output) > 1:
        return output[1]
    return None


def _forward_hf(
    model: Any,
    input_ids: torch.Tensor,
    cache: Any,
    *,
    cache_length: int,
) -> Any:
    """Call a modern or legacy HF causal LM with a cache.

    Most Qwen2 releases infer ``cache_position``.  A retry is provided for
    model forks that require it explicitly; no broad exception swallowing is
    used, so genuine model errors remain visible.
    """

    attention_length = int(cache_length) + int(input_ids.shape[-1])
    kwargs: dict[str, Any] = {
        "input_ids": input_ids,
        "attention_mask": torch.ones(
            (input_ids.shape[0], attention_length),
            dtype=torch.long,
            device=input_ids.device,
        ),
        "use_cache": True,
        "return_dict": True,
    }
    if cache is not None:
        kwargs["past_key_values"] = cache
    try:
        return model(**kwargs)
    except (TypeError, ValueError) as exc:
        message = str(exc).lower()
        if "cache_position" not in message and "cache position" not in message:
            raise
        kwargs["cache_position"] = torch.arange(
            cache_length,
            cache_length + input_ids.shape[-1],
            dtype=torch.long,
            device=input_ids.device,
        )
        return model(**kwargs)


def _set_hook(model: Any, hook: Any, enabled: bool) -> None:
    if hook is None:
        return
    installed = bool(getattr(hook, "installed", False))
    if enabled and not installed:
        hook.install(model)
    elif not enabled and installed:
        hook.remove()


def _mode_name(bits: int) -> str:
    return "fp" if bits == 0 else f"kv{bits}"


def _model_metadata(model: Any) -> dict[str, Any]:
    config = getattr(model, "config", None)
    return {
        "class": model.__class__.__name__,
        "module": model.__class__.__module__,
        "dtype": _dtype_name(model),
        "device": str(_device_of_model(model)),
        "architecture": list(getattr(config, "architectures", []) or []),
        "model_type": getattr(config, "model_type", None),
        "num_hidden_layers": getattr(config, "num_hidden_layers", None),
        "num_attention_heads": getattr(config, "num_attention_heads", None),
        "num_key_value_heads": getattr(config, "num_key_value_heads", None),
        "hidden_size": getattr(config, "hidden_size", None),
        "transformers_version": _package_version("transformers"),
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
    }


def _package_version(name: str) -> Optional[str]:
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:
        return None


def _run_pair(
    model: Any,
    input_ids: torch.Tensor,
    *,
    bits: int,
    options: ParityOptions,
    decode_ids: Optional[Sequence[int]] = None,
) -> dict[str, Any]:
    """Run one quantized mode against an interleaved FP reference."""

    hook = None
    hooked_names: list[str] = []
    if bits > 0:
        from qat_kv import attach_kv_fake_quant_hooks

        hook = attach_kv_fake_quant_hooks(
            model,
            bits=bits,
            layerwise=options.layerwise,
            preserve_projection_bias=options.preserve_projection_bias,
            clip_range=options.clip_range,
        )
        hooked_names = list(getattr(hook, "hooked_names", ()))

    fp_cache = None
    candidate_cache = None
    current_length = 0
    phases: list[dict[str, Any]] = []
    selected_ids: list[int] = []
    started = time.perf_counter()
    try:
        # Prefill is one complete call.  Truncating from the left keeps the
        # requested context budget while preserving the most recent prompt.
        prefill_ids = input_ids
        if options.prefill_tokens and prefill_ids.shape[-1] > options.prefill_tokens:
            prefill_ids = prefill_ids[:, -options.prefill_tokens :]
        _set_hook(model, hook, False)
        fp_output = _forward_hf(model, prefill_ids, None, cache_length=0)
        fp_cache = _output_cache(fp_output)
        if fp_cache is None:
            raise RuntimeError("model did not return past_key_values with use_cache=True")
        _set_hook(model, hook, True)
        candidate_output = _forward_hf(model, prefill_ids, None, cache_length=0)
        candidate_cache = _output_cache(candidate_output)
        if candidate_cache is None:
            raise RuntimeError("quantized model did not return past_key_values")
        prefill_length = int(prefill_ids.shape[-1])
        current_length = prefill_length
        prefill_logits = _output_logits(fp_output)
        candidate_prefill_logits = _output_logits(candidate_output)
        phases.append(
            {
                "name": "prefill",
                "input_tokens": prefill_length,
                "cache_length": cache_seq_len(fp_cache),
                "logits": logit_error_report(
                    prefill_logits,
                    candidate_prefill_logits,
                    token_offset=0,
                    chunk_size=options.logit_chunk_size,
                    top_k=options.top_k,
                ),
                "cache": cache_error_report(
                    fp_cache,
                    candidate_cache,
                    token_start=0,
                    token_end=prefill_length,
                    chunk_size=options.cache_chunk_size,
                ),
            }
        )

        # Decode tokens are either supplied explicitly or selected greedily
        # from the FP reference.  Both branches receive exactly the same IDs;
        # this isolates cache/logit error from sampling-policy divergence.
        if decode_ids is None:
            first_next = int(prefill_logits[:, -1, :].float().argmax(dim=-1)[0].item())
            planned_ids: list[int] = [first_next]
        else:
            planned_ids = [int(item) for item in decode_ids]
        for step in range(options.decode_tokens):
            if step >= len(planned_ids):
                # For greedy mode, choose the next token from the latest FP
                # output.  Explicit IDs must have been length-validated.
                raise RuntimeError("decode token plan is shorter than decode_tokens")
            token_id = int(planned_ids[step])
            selected_ids.append(token_id)
            token = torch.tensor([[token_id]], dtype=torch.long, device=input_ids.device)
            _set_hook(model, hook, False)
            fp_output = _forward_hf(
                model, token, fp_cache, cache_length=current_length
            )
            fp_cache = _output_cache(fp_output)
            _set_hook(model, hook, True)
            candidate_output = _forward_hf(
                model, token, candidate_cache, cache_length=current_length
            )
            candidate_cache = _output_cache(candidate_output)
            if fp_cache is None or candidate_cache is None:
                raise RuntimeError("incremental model call dropped past_key_values")
            fp_logits = _output_logits(fp_output)
            candidate_logits = _output_logits(candidate_output)
            phases.append(
                {
                    "name": "decode",
                    "step": step,
                    "token_id": token_id,
                    "cache_length": cache_seq_len(fp_cache),
                    "logits": logit_error_report(
                        fp_logits,
                        candidate_logits,
                        token_offset=current_length,
                        chunk_size=options.logit_chunk_size,
                        top_k=options.top_k,
                    ),
                    # Only the newly appended token is compared here.  The
                    # prefill phase already reported all earlier positions;
                    # this keeps long-context decode O(number of new tokens).
                    "cache": cache_error_report(
                        fp_cache,
                        candidate_cache,
                        token_start=current_length,
                        token_end=current_length + 1,
                        chunk_size=options.cache_chunk_size,
                    ),
                }
            )
            current_length += 1
            if decode_ids is None and step + 1 < options.decode_tokens:
                # The next planned ID is selected from the FP output only.
                planned_ids.append(int(fp_logits[:, -1, :].float().argmax(dim=-1)[0].item()))
    finally:
        _set_hook(model, hook, False)
        if hook is not None:
            hook.remove()
    return {
        "mode": _mode_name(bits),
        "bits": bits,
        "reference": "full_precision_same_model_weights",
        "elapsed_seconds": time.perf_counter() - started,
        # Capture this before ``remove`` clears the hook's live-name list.
        # It is the concrete evidence that every requested projection seam was
        # found, rather than merely the requested leaf-name policy.
        "hooked_names": hooked_names,
        "hooked_count": len(hooked_names),
        "selected_decode_ids": selected_ids,
        "phases": phases,
    }


def run_parity(
    model: Any,
    input_ids: torch.Tensor,
    *,
    options: Optional[ParityOptions] = None,
) -> dict[str, Any]:
    """Run FP/KV2/KV4 parity on an already-loaded HF-like model.

    This public function is intentionally model-agnostic and is the entry
    point used by the tiny CPU tests.  ``input_ids`` must be shape ``[B,T]``;
    parity currently uses batch size one so greedy token selection is
    unambiguous.
    """

    if options is None:
        options = ParityOptions()
    if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2:
        raise ValueError("input_ids must be a rank-2 [batch, tokens] tensor")
    if input_ids.shape[0] != 1:
        raise ValueError("parity currently requires batch size 1")
    if input_ids.dtype not in (torch.int64, torch.int32):
        raise TypeError("input_ids must be an integer tensor")
    model_device = _device_of_model(model)
    input_ids = input_ids.to(model_device)
    if options.prefill_tokens and input_ids.shape[-1] < 1:
        raise ValueError("input_ids must contain at least one token")
    decode_ids = options.decode_ids if options.decode_ids else None
    reports = []
    for bits in options.modes:
        reports.append(
            _run_pair(
                model,
                input_ids,
                bits=bits,
                options=options,
                decode_ids=decode_ids,
            )
        )
    return {
        "schema_version": 1,
        "model": _model_metadata(model),
        "quantizer": {
            "implementation": "models.utils_quant.SymQuantizer",
            "bits_modes": list(options.modes),
            "axis": "per-token last dimension before projection reshape/RoPE",
            "layerwise": options.layerwise,
            "clip_range_backward_only": list(options.clip_range),
            "epsilon": 1e-6,
            "ste": True,
            "storage": "floating_point_dequantized_fake_quant",
            "packed_integer_cache": False,
            "projection_names": ["k_proj", "v_proj"],
            "preserve_projection_bias": options.preserve_projection_bias,
            "hf_bias_preservation": (
                "post_forward_approximate"
                if options.preserve_projection_bias
                else "disabled"
            ),
        },
        "protocol": {
            "prefill_tokens_requested": options.prefill_tokens,
            "decode_tokens": options.decode_tokens,
            "decode_policy": "fixed_ids" if decode_ids is not None else "greedy_from_fp_reference",
            "decode_ids": list(decode_ids or []),
            "cache_comparison": "prefill_all_tokens_then_decode_new_token_only",
            "same_weights": True,
        },
        "modes": reports,
    }


def _resolve_dtype(name: str, device: torch.device) -> torch.dtype:
    if name == "fp32":
        return torch.float32
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        if device.type == "cuda" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("bf16 requested but CUDA device does not support it")
        return torch.bfloat16
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float32


def _load_model(args: argparse.Namespace, device: torch.device):
    try:
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("kv_parity requires transformers") from exc
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    load_kwargs: dict[str, Any] = {
        "torch_dtype": _resolve_dtype(args.dtype, device),
        "low_cpu_mem_usage": True,
    }
    if args.attn_implementation:
        load_kwargs["attn_implementation"] = args.attn_implementation
    if args.device_map == "auto":
        # ``device_map=auto`` must be passed at load time; merely skipping the
        # later ``.to(device)`` would otherwise leave the complete model on
        # CPU while reporting a requested CUDA device.
        load_kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)
    if args.device_map == "none":
        model = model.to(device)
    model.eval()
    if getattr(getattr(model, "config", None), "use_cache", True) is False:
        model.config.use_cache = True
    return model, tokenizer, transformers.__version__


def _encode(tokenizer: Any, prompt: str, *, chat_template: bool, device: torch.device) -> torch.Tensor:
    if chat_template and hasattr(tokenizer, "apply_chat_template"):
        try:
            ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            )
        except Exception:
            ids = tokenizer(prompt, return_tensors="pt").input_ids
    else:
        ids = tokenizer(prompt, return_tensors="pt").input_ids
    if isinstance(ids, dict):
        ids = ids["input_ids"]
    return ids.to(device)


def _read_prompt(args: argparse.Namespace) -> tuple[str, Optional[str]]:
    """Resolve an inline or UTF-8 file prompt and retain its provenance."""

    if args.prompt_file is not None:
        try:
            return args.prompt_file.read_text(encoding="utf-8"), str(args.prompt_file.resolve())
        except (OSError, UnicodeError) as exc:
            raise SystemExit(f"cannot read --prompt-file {args.prompt_file}: {exc}") from exc
    return args.prompt if args.prompt is not None else DEFAULT_PROMPT, None


def _parse_modes(text: str) -> tuple[int, ...]:
    result: list[int] = []
    for item in text.split(","):
        item = item.strip().lower()
        if item in {"fp", "fp32", "0"}:
            bit = 0
        elif item.startswith("kv"):
            bit = int(item[2:])
        else:
            bit = int(item)
        if bit not in (0, 2, 4):
            raise argparse.ArgumentTypeError("modes must use fp/2/4")
        if bit not in result:
            result.append(bit)
    if not result:
        raise argparse.ArgumentTypeError("modes must not be empty")
    return tuple(result)


def _parse_decode_ids(text: str) -> tuple[int, ...]:
    if not text.strip():
        return ()
    try:
        values = tuple(int(item.strip()) for item in text.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("decode-ids must be comma-separated integers") from exc
    if any(value < 0 for value in values):
        raise argparse.ArgumentTypeError("decode-ids must be non-negative")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="HF model path or repository ID")
    prompt_group = parser.add_mutually_exclusive_group()
    prompt_group.add_argument("--prompt", default=None)
    prompt_group.add_argument(
        "--prompt-file",
        type=Path,
        default=None,
        help="read the prompt as UTF-8 (mutually exclusive with --prompt)",
    )
    parser.add_argument("--chat-template", action="store_true")
    parser.add_argument("--prefill-tokens", type=int, default=0)
    parser.add_argument("--decode-tokens", type=int, default=4)
    parser.add_argument("--decode-ids", type=_parse_decode_ids, default=())
    parser.add_argument("--modes", type=_parse_modes, default=(0, 2, 4),
                        help="comma-separated fp/2/4 modes (default: fp,2,4)")
    parser.add_argument("--preserve-projection-bias", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--layerwise", action="store_true")
    parser.add_argument("--clip-range", default="-2,2")
    parser.add_argument("--cache-chunk-size", type=int, default=256)
    parser.add_argument("--logit-chunk-size", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--dtype", choices=("auto", "bf16", "fp16", "fp32"), default="auto")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--device-map", choices=("none", "auto"), default="none")
    parser.add_argument("--attn-implementation", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    prompt, prompt_file = _read_prompt(args)
    try:
        clip_range = tuple(float(item.strip()) for item in args.clip_range.split(","))
    except (TypeError, ValueError) as exc:
        raise SystemExit("--clip-range must be two comma-separated numbers") from exc
    if len(clip_range) != 2:
        raise SystemExit("--clip-range must be two comma-separated numbers")
    if args.decode_ids and len(args.decode_ids) != args.decode_tokens:
        raise SystemExit("--decode-ids length must equal --decode-tokens")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
    model, tokenizer, _transformers_version = _load_model(args, device)
    input_ids = _encode(tokenizer, prompt, chat_template=args.chat_template, device=_device_of_model(model))
    options = ParityOptions(
        modes=args.modes,
        prefill_tokens=args.prefill_tokens,
        decode_tokens=args.decode_tokens,
        preserve_projection_bias=args.preserve_projection_bias,
        layerwise=args.layerwise,
        clip_range=clip_range,
        cache_chunk_size=args.cache_chunk_size,
        logit_chunk_size=args.logit_chunk_size,
        top_k=args.top_k,
        decode_ids=args.decode_ids,
    )
    report = run_parity(model, input_ids, options=options)
    report["runtime"] = {
        "device_requested": args.device,
        "device_map": args.device_map,
        "dtype_requested": args.dtype,
    }
    report["input"] = {
        "prompt": prompt,
        "prompt_file": prompt_file,
        "prompt_tokens": int(input_ids.shape[-1]),
        "token_ids": input_ids[0].detach().cpu().tolist(),
    }
    report["command"] = " ".join(str(item) for item in ["python", *(["scripts/kv_parity.py"] + list(os.sys.argv[1:]))])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps({
        "output": str(args.output),
        "modes": [item["mode"] for item in report["modes"]],
        "prompt_tokens": report["input"]["prompt_tokens"],
    }, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI exercised separately
    raise SystemExit(main())
