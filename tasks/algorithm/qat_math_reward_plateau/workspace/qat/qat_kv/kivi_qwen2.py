"""Official-KIVI cache/attention adapter for Qwen2 models.

The quantization and packed matmul routines are imported from the official
KIVI checkout (``jy-yuan/KIVI``).  This file only adapts their Mistral/Llama
attention boundary to the Qwen2 attention API used by the two models in this
repository.  There is deliberately no Python fallback to a different cache:
when packed KIVI rollout is requested, the official CUDA extension must be
available.

Training and rollout use the same placement (after RoPE, before attention):

* training (``use_cache=False``): quantize/dequantize with the repository's
  LLM-QAT STE operator;
* generation (``use_cache=True``): KIVI's packed prefix plus full-precision
  residual window, and KIVI's official packed-cache GEMV kernel.

Qwen2 has separate K/V projection biases while the original KIVI Llama and
Mistral models do not.  The projection output (including the learned bias) is
  passed to KIVI, matching the model's actual attention input.  The existing
  exact/approximate *pre-RoPE* bias handling is a separate LLM-QAT path and is
  not silently mixed into this cache implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import os
import sys
import contextvars
from types import MethodType
from typing import Any, Optional

import torch
from torch import nn

try:
    from verl.workers.rollout.base import BaseRollout
    from verl.workers.rollout.hf_rollout import HFRollout as _LegacyHFRollout
except ImportError:  # CPU/unit-test imports do not require VeRL.
    BaseRollout = object  # type: ignore[assignment,misc]
    _LegacyHFRollout = object  # type: ignore[assignment,misc]


# The pinned VeRL worker constructs a rollout from (config, model_config,
# device_mesh), while its legacy HF rollout implementation expects the actor
# module as a positional argument.  The worker plugin sets this context just
# around construction so the adapter can use the already-built actor without
# copying VeRL's training loop.
_ROLLOUT_MODULE_CONTEXT: contextvars.ContextVar[Any] = contextvars.ContextVar(
    "qat_kivi_rollout_module", default=None
)


def set_kivi_rollout_module(module: Any) -> contextvars.Token:
    return _ROLLOUT_MODULE_CONTEXT.set(module)


def reset_kivi_rollout_module(token: contextvars.Token) -> None:
    _ROLLOUT_MODULE_CONTEXT.reset(token)


class KIVIHFRollout(_LegacyHFRollout):
    """VeRL-compatible HF rollout using the actor's KIVI-patched module.

    The pinned Open-MOPD worker uses the newer ``BaseRollout`` constructor,
    while its bundled legacy ``HFRollout`` contains the actual generation
    implementation but an obsolete constructor and abstract lifecycle.  This
    adapter preserves that VeRL generation code and only supplies the API
    bridge; it does not implement a separate RL loop or optimizer.
    """

    def __init__(self, *, config: Any, model_config: Any = None, device_mesh: Any = None):
        module = _ROLLOUT_MODULE_CONTEXT.get()
        if module is None:
            raise RuntimeError("KIVI HF rollout construction requires the actor module context")
        if BaseRollout is not object:
            BaseRollout.__init__(self, config, model_config, device_mesh)
        else:
            self.config = config
        self.module = module
        self.model_config = model_config
        self.device_mesh = device_mesh
        
        # Some Qwen tokenizers intentionally leave ``pad_token_id`` unset.
        # The legacy VeRL HF rollout uses it when padding an early-EOS
        # sequence, so generation would otherwise fail only after the long
        # prefill had completed.  Hugging Face's standard fallback is EOS.
        generation_config = getattr(module, "generation_config", None)
        model_cfg = getattr(module, "config", None)

        def _token_id(value: Any) -> Optional[int]:
            if value is None:
                return None
            if isinstance(value, torch.Tensor):
                value = value.detach().reshape(-1).tolist()
            elif hasattr(value, "tolist") and not isinstance(value, (list, tuple)):
                value = value.tolist()
            if isinstance(value, (list, tuple)):
                if not value:
                    return None
                value = value[0]
            return int(value)

        self._pad_token_id = _token_id(getattr(generation_config, "pad_token_id", None))
        if self._pad_token_id is None:
            self._pad_token_id = _token_id(getattr(model_cfg, "pad_token_id", None))
        if self._pad_token_id is None:
            self._pad_token_id = _token_id(getattr(generation_config, "eos_token_id", None))
        if self._pad_token_id is None:
            self._pad_token_id = _token_id(getattr(model_cfg, "eos_token_id", None))
        if self._pad_token_id is None:
            raise ValueError("KIVI HF rollout requires pad_token_id or eos_token_id")
    def _valid_prompt_lengths(self, prompts) -> list[int]:
        """Return valid (non-left-pad) lengths for every row."""
        mask = prompts.batch["attention_mask"]
        input_ids = prompts.batch["input_ids"]
        if mask.ndim != 2 or input_ids.ndim != 2 or mask.shape != input_ids.shape:
            raise ValueError("KIVI HF rollout expects [batch, width] input_ids and attention_mask")
        bool_mask = mask.to(dtype=torch.bool)
        # KIVI prefill trimming relies on the common left-padding convention;
        # reject right-padding or holes instead of packing synthetic rows.
        if bool_mask.shape[1] > 1 and torch.any(bool_mask[:, :-1] & ~bool_mask[:, 1:]):
            raise ValueError("KIVI HF rollout requires left-padded, contiguous prompt masks")
        lengths = bool_mask.sum(dim=1)
        if torch.any(lengths <= 0):
            raise ValueError("KIVI HF rollout received an all-padding prompt")
        return [int(x) for x in lengths.detach().cpu().tolist()]

    def _trim_left_padding(self, prompts, valid_length: int):
        """Trim a common left pad from an equal-length bucket."""
        from tensordict import TensorDict
        from verl import DataProto

        input_ids = prompts.batch["input_ids"]
        attention_mask = prompts.batch["attention_mask"]
        position_ids = prompts.batch["position_ids"]
        width = input_ids.shape[1]
        left_padding = width - valid_length
        if left_padding <= 0:
            return prompts, 0
        # All rows in a bucket have the same valid length and VeRL uses
        # left-padding, so the same trailing columns are real prompt tokens.
        indices = torch.arange(left_padding, width, device=input_ids.device)
        trimmed = {}
        for key, value in prompts.batch.items():
            if key in {"input_ids", "attention_mask", "position_ids"} and value.ndim >= 2 and value.shape[1] == width:
                trimmed[key] = value.index_select(1, indices.to(value.device))
            else:
                trimmed[key] = value
        return (
            DataProto(
                batch=TensorDict(trimmed, batch_size=prompts.batch.batch_size, device=prompts.batch.device),
                non_tensor_batch=prompts.non_tensor_batch,
                meta_info=prompts.meta_info,
            ),
            left_padding,
        )

    @staticmethod
    def _restore_left_padding(result, original_prompts, left_padding: int):
        if not left_padding:
            return result
        input_ids = original_prompts.batch["input_ids"]
        attention_mask = original_prompts.batch["attention_mask"]
        position_ids = original_prompts.batch["position_ids"]
        prefixes = {
            "prompts": input_ids[:, :left_padding],
            "input_ids": input_ids[:, :left_padding],
            "attention_mask": attention_mask[:, :left_padding],
            "position_ids": position_ids[:, :left_padding],
        }
        for key, prefix in prefixes.items():
            result.batch[key] = torch.cat((prefix.to(result.batch[key].device), result.batch[key]), dim=1)
        return result

    def generate_sequences(self, prompts):
        """Generate KIVI rollouts in safe equal-length buckets.

        KIVI's packed representation supports a leading batch dimension, but
        one cache object has one token boundary for all rows.  We therefore
        bucket by valid prompt length.  EOS-enabled generation remains
        row-wise because Transformers keeps finished rows in a batch and
        feeds them padding tokens; that would violate KIVI's cache contract.
        Batching is enabled for equal-length buckets when EOS is explicitly
        disabled, which gives a deterministic first parity target without
        silently changing EOS semantics.
        """
        from verl import DataProto

        batch_size = int(prompts.batch.batch_size[0])
        if batch_size <= 0:
            raise ValueError("KIVI HF rollout received an empty batch")
        if prompts.meta_info.get("pad_token_id") is None:
            prompts.meta_info["pad_token_id"] = self._pad_token_id
        lengths = self._valid_prompt_lengths(prompts)
        eos_token_id = prompts.meta_info.get("eos_token_id")
        ignore_eos = bool(prompts.meta_info.get("ignore_eos", getattr(self.config, "ignore_eos", False)))
        eos_enabled = eos_token_id is not None and not ignore_eos

        # Never batch rows for which HF generation would append post-EOS pad
        # tokens to the cache.  This is an explicit safety rule, not a silent
        # performance fallback.
        # The official extension's multi-row Qwen GQA path is not yet
        # numerically stable after several packed-prefix updates (long-cache
        # parity can diverge while one-row parity remains exact).  Keep the
        # safe production default row-wise; callers may opt into the bucketed
        # path only for a dedicated parity experiment.
        allow_batched = bool(prompts.meta_info.get("kivi_batch_generation", False))
        if eos_enabled or not allow_batched:
            groups = [[i] for i in range(batch_size)]
        else:
            buckets: dict[int, list[int]] = {}
            for index, length in enumerate(lengths):
                buckets.setdefault(length, []).append(index)
            groups = list(buckets.values())

        outputs = []
        output_order: list[int] = []
        for indices in groups:
            bucket = prompts.select_idxs(indices)
            # ``DataProto`` metadata is shared by indexed views.  Copy it so
            # honoring ignore_eos for this call cannot mutate later buckets.
            bucket.meta_info = dict(bucket.meta_info)
            if ignore_eos:
                bucket.meta_info["eos_token_id"] = None
            trimmed_bucket, left_padding = self._trim_left_padding(bucket, lengths[indices[0]])
            trimmed_bucket.meta_info = dict(trimmed_bucket.meta_info)
            result = self._generate_minibatch(trimmed_bucket)
            result = self._restore_left_padding(result, bucket, left_padding)
            outputs.append(result)
            output_order.extend(indices)

        result = DataProto.concat(outputs)
        if output_order != list(range(batch_size)):
            inverse = [0] * batch_size
            for output_index, original_index in enumerate(output_order):
                inverse[original_index] = output_index
            result.reorder(torch.tensor(inverse, dtype=torch.long))
        return result

    async def resume(self, tags: list[str]):
        return None

    async def update_weights(self, weights, **kwargs):
        # The rollout and actor intentionally share the same module object;
        # VeRL has already updated it before invoking this lifecycle method.
        return None

    async def release(self):
        return None



def _kivi_ste_affine_quant(x: torch.Tensor, *, dim: int, group_size: int, bits: int) -> torch.Tensor:
    """Official KIVI asymmetric min/max quantize-dequantize with STE.

    The arithmetic follows ``quant/new_pack.py``.  The straight-through
    wrapper is the only training-specific part: forward sees the dequantized
    KIVI value while backward treats it as the identity map.
    """

    if x.shape[dim] % group_size:
        raise ValueError(
            f"KIVI fake quant dimension {x.shape[dim]} is not divisible by group_size={group_size}"
        )
    moved = x.movedim(dim, -1).contiguous()
    grouped = moved.reshape(*moved.shape[:-1], moved.shape[-1] // group_size, group_size)
    x_fp32 = grouped.float()
    mn = x_fp32.amin(dim=-1, keepdim=True)
    mx = x_fp32.amax(dim=-1, keepdim=True)
    qmax = float(2**bits - 1)
    scale = (mx - mn) / qmax
    safe_scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    code = ((x_fp32 - mn) / safe_scale).clamp(0, qmax).round()
    restored = (code * safe_scale + mn).reshape_as(moved).to(dtype=x.dtype)
    restored = restored.movedim(-1, dim)
    return x + (restored - x).detach()


def _kivi_fake_quant_key_ste(x: torch.Tensor, cfg: KIVIQwen2Config) -> torch.Tensor:
    """KIVI key rule: asymmetric groups over the token axis, residual tail."""

    quant_len = (x.shape[-2] // cfg.residual_length) * cfg.residual_length
    if quant_len == 0:
        return x
    prefix = _kivi_ste_affine_quant(
        x[..., :quant_len, :], dim=-2, group_size=cfg.group_size, bits=cfg.bits
    )
    return torch.cat((prefix, x[..., quant_len:, :]), dim=-2)


def _kivi_fake_quant_value_ste(x: torch.Tensor, cfg: KIVIQwen2Config) -> torch.Tensor:
    """KIVI value rule: asymmetric groups over the head/channel axis."""

    return _kivi_ste_affine_quant(x, dim=-1, group_size=cfg.group_size, bits=cfg.bits)


KIVI_COMMIT = "876b4d2d08e3b1d5f70d0969c299d8c7c42ddfb6"
DEFAULT_KIVI_ROOTS = (
    "/historical/shared/qat/third_party/KIVI",
    "/tmp/kivi.iceD4D",
)


@dataclass(frozen=True)
class KIVIQwen2Config:
    """KIVI parameters used by the official implementation."""

    bits: int = 2
    group_size: int = 32
    residual_length: int = 32
    qat_clip_range: tuple[float, float] = (-2.0, 2.0)

    def __post_init__(self) -> None:
        if isinstance(self.bits, bool) or self.bits not in (2, 4, 8):
            raise ValueError("official KIVI supports bits 2, 4, or 8")
        if isinstance(self.group_size, bool) or self.group_size <= 0:
            raise ValueError("group_size must be positive")
        if isinstance(self.residual_length, bool) or self.residual_length <= 0:
            raise ValueError("residual_length must be positive")
        if self.residual_length % self.group_size:
            raise ValueError("residual_length must be divisible by group_size")
        if len(self.qat_clip_range) != 2 or not (
            float(self.qat_clip_range[0]) < float(self.qat_clip_range[1])
        ):
            raise ValueError("qat_clip_range must be (low, high)")


_KERNELS: Optional[tuple[Any, Any]] = None
_TRITON_COMPAT_READY = False


def _ensure_triton_sm103a_compat() -> None:
    """Make the pinned official Triton kernels usable on CUDA 13/SM103a.

    KIVI's upstream Triton code is unchanged.  The installed Triton release
    only knows CUDA-12 PTX versions, while this node's system ``ptxas`` is
    CUDA 13.  For this known combination we select PTX 9.0 and the system
    assembler; on other devices this function is a no-op.
    """

    global _TRITON_COMPAT_READY
    if _TRITON_COMPAT_READY or not torch.cuda.is_available():
        return
    capability = torch.cuda.get_device_capability()
    if tuple(capability) != (10, 3):
        _TRITON_COMPAT_READY = True
        return
    # KIVI is invoked concurrently by all FSDP ranks.  Triton writes a
    # metadata file during compilation; sharing one cache lets ranks race and
    # observe a half-written file.  Give each rank a stable private cache.
    if "TRITON_CACHE_DIR" not in os.environ:
        rank = os.environ.get("RANK") or os.environ.get("LOCAL_RANK") or str(os.getpid())
        cache_root = "/historical/shared/qat/ray/triton-kivi-cache"
        os.environ["TRITON_CACHE_DIR"] = os.path.join(cache_root, f"rank-{rank}")
        os.makedirs(os.environ["TRITON_CACHE_DIR"], exist_ok=True)
    try:
        import triton.backends.nvidia.compiler as triton_compiler
    except Exception as exc:
        raise RuntimeError("KIVI requires Triton for its official CUDA pack kernel") from exc

    ptxas = "/usr/local/cuda/bin/ptxas"
    if os.path.isfile(ptxas):
        os.environ.setdefault("TRITON_PTXAS_PATH", ptxas)
    # Triton 3.x's PTX selector rejects CUDA 13 before invoking ptxas.  The
    # system assembler accepts PTX 9.0, so patch only this known SM/toolchain.
    triton_compiler.ptx_get_version = lambda _cuda_version: 90
    python_root = "/historical/shared/qat/tmp/Python-3.12.7"
    python_include = os.path.join(python_root, "Include")
    if os.path.isdir(python_include) and os.path.isfile(os.path.join(python_root, "pyconfig.h")):
        existing = os.environ.get("CPATH", "")
        paths = existing.split(os.pathsep) if existing else []
        for include_path in (python_include, python_root):
            if include_path not in paths:
                paths.insert(0, include_path)
        os.environ["CPATH"] = os.pathsep.join(paths)
    _TRITON_COMPAT_READY = True


def _kivi_roots() -> list[str]:
    configured = os.environ.get("KIVI_ROOT")
    roots = [configured] if configured else []
    roots.extend(DEFAULT_KIVI_ROOTS)
    result: list[str] = []
    for root in roots:
        if root and root not in result and os.path.isdir(root):
            result.append(root)
    return result


def load_official_kivi_kernels() -> tuple[Any, Any]:
    """Load official KIVI pack and packed-GEMV functions.

    The extension is intentionally loaded lazily, so importing QAT configs or
    running CPU unit tests does not require CUDA.  A missing extension is an
    explicit error rather than a silent fake-cache fallback.
    """

    global _KERNELS
    if _KERNELS is not None:
        return _KERNELS
    _ensure_triton_sm103a_compat()
    errors: list[str] = []
    for root in _kivi_roots():
        # ``quant.matmul`` imports ``kivi_gemv`` as a top-level extension.  A
        # build made with ``quant/setup.py build_ext --inplace`` puts the .so
        # in ``root/quant``, hence both paths are needed.
        for path in (root, os.path.join(root, "quant")):
            if path not in sys.path:
                sys.path.insert(0, path)
        try:
            pack = importlib.import_module("quant.new_pack")
            matmul = importlib.import_module("quant.matmul")
            if not hasattr(pack, "triton_quantize_and_pack_along_last_dim"):
                raise ImportError("official quant.new_pack has no pack function")
            if not hasattr(matmul, "cuda_bmm_fA_qB_outer"):
                raise ImportError("official quant.matmul has no packed GEMV function")
            # Importing matmul succeeds only when its official extension is
            # importable, but check explicitly to make the failure actionable.
            importlib.import_module("kivi_gemv")
            _KERNELS = (
                pack.triton_quantize_and_pack_along_last_dim,
                matmul.cuda_bmm_fA_qB_outer,
            )
            return _KERNELS
        except Exception as exc:  # keep trying an explicitly configured root
            errors.append(f"{root}: {type(exc).__name__}: {exc}")
    joined = "; ".join(errors) if errors else "no KIVI checkout found"
    raise RuntimeError(
        "official KIVI CUDA extension is unavailable. Build the pinned official "
        f"KIVI checkout ({KIVI_COMMIT}) with scripts/build_kivi_kernel.sh; {joined}"
    )


def _pack_key(key: torch.Tensor, cfg: KIVIQwen2Config) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pack a KIVI key prefix using the official function."""

    if key.shape[-2] == 0:
        raise ValueError("cannot pack an empty KIVI key prefix")
    pack, _ = load_official_kivi_kernels()
    # Official KIVI's key layout is [B, H, D, T] and quantizes over T.
    source = key.transpose(-1, -2).contiguous()
    # The shipped extension is half-only.  BF16 models use the same official
    # kernel after an explicit half conversion; the attention result is cast
    # back to the model dtype by the caller.
    source = source.to(dtype=torch.float16)
    if source.shape[0] == 1:
        code, scale, mn = pack(source, cfg.group_size, cfg.bits)
    else:
        packed = [pack(source[i : i + 1], cfg.group_size, cfg.bits) for i in range(source.shape[0])]
        code = torch.cat([item[0] for item in packed], dim=0)
        scale = torch.cat([item[1] for item in packed], dim=0)
        mn = torch.cat([item[2] for item in packed], dim=0)
    return code, scale.to(torch.float16), mn.to(torch.float16)


def _pack_value(value: torch.Tensor, cfg: KIVIQwen2Config) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pack a KIVI value prefix using the official function."""

    if value.shape[-2] == 0:
        raise ValueError("cannot pack an empty KIVI value prefix")
    pack, _ = load_official_kivi_kernels()
    source = value.contiguous().to(dtype=torch.float16)
    if source.shape[0] == 1:
        code, scale, mn = pack(source, cfg.group_size, cfg.bits)
    else:
        packed = [pack(source[i : i + 1], cfg.group_size, cfg.bits) for i in range(source.shape[0])]
        code = torch.cat([item[0] for item in packed], dim=0)
        scale = torch.cat([item[1] for item in packed], dim=0)
        mn = torch.cat([item[2] for item in packed], dim=0)
    return code, scale.to(torch.float16), mn.to(torch.float16)


class KIVILayer:
    """One cache layer: packed prefix and unquantized residual window."""

    is_sliding = False
    is_compileable = False

    def __init__(self, cfg: KIVIQwen2Config):
        self.cfg = cfg
        self.key_code: Optional[torch.Tensor] = None
        self.key_scale: Optional[torch.Tensor] = None
        self.key_min: Optional[torch.Tensor] = None
        self.key_full: Optional[torch.Tensor] = None
        self.value_code: Optional[torch.Tensor] = None
        self.value_scale: Optional[torch.Tensor] = None
        self.value_min: Optional[torch.Tensor] = None
        self.value_full: Optional[torch.Tensor] = None
        self.is_initialized = False
        self.device: Optional[torch.device] = None
        self.dtype: Optional[torch.dtype] = None

    def get_seq_length(self) -> int:
        quant_len = 0
        if self.key_code is not None:
            quant_len = self.key_code.shape[-1] * (32 // self.cfg.bits)
        full_len = 0 if self.key_full is None else self.key_full.shape[-2]
        return int(quant_len + full_len)

    def get_mask_sizes(self, cache_position: torch.Tensor) -> tuple[int, int]:
        return self.get_seq_length() + int(cache_position.shape[0]), 0

    def get_max_cache_shape(self) -> int:
        return -1

    def lazy_initialization(self, key_states: torch.Tensor) -> None:
        self.device = key_states.device
        self.dtype = key_states.dtype
        self.is_initialized = True

    def _append_key_block(self, block: torch.Tensor) -> None:
        code, scale, mn = _pack_key(block, self.cfg)
        if self.key_code is None:
            self.key_code, self.key_scale, self.key_min = code, scale, mn
        else:
            self.key_code = torch.cat((self.key_code, code), dim=-1)
            self.key_scale = torch.cat((self.key_scale, scale), dim=-1)
            self.key_min = torch.cat((self.key_min, mn), dim=-1)

    def _append_value_block(self, block: torch.Tensor) -> None:
        code, scale, mn = _pack_value(block, self.cfg)
        if self.value_code is None:
            self.value_code, self.value_scale, self.value_min = code, scale, mn
        else:
            self.value_code = torch.cat((self.value_code, code), dim=-2)
            self.value_scale = torch.cat((self.value_scale, scale), dim=-2)
            self.value_min = torch.cat((self.value_min, mn), dim=-2)

    def prefill(self, key_states: torch.Tensor, value_states: torch.Tensor) -> None:
        if key_states.shape != value_states.shape or key_states.ndim != 4:
            raise ValueError("KIVI cache states must be [batch, kv_heads, tokens, head_dim]")
        self.lazy_initialization(key_states)
        tokens = key_states.shape[-2]
        key_quant_len = (tokens // self.cfg.residual_length) * self.cfg.residual_length
        value_quant_len = max(0, tokens - self.cfg.residual_length)
        self.key_full = key_states[..., key_quant_len:, :].contiguous()
        self.value_full = value_states[..., value_quant_len:, :].contiguous()
        if key_quant_len:
            self._append_key_block(key_states[..., :key_quant_len, :].contiguous())
        if value_quant_len:
            self._append_value_block(value_states[..., :value_quant_len, :].contiguous())

    def append(self, key_states: torch.Tensor, value_states: torch.Tensor) -> None:
        if key_states.shape != value_states.shape or key_states.ndim != 4:
            raise ValueError("KIVI cache states must be [batch, kv_heads, tokens, head_dim]")
        if not self.is_initialized:
            self.prefill(key_states, value_states)
            return

        self.key_full = torch.cat((self.key_full, key_states), dim=-2) if self.key_full is not None else key_states
        self.value_full = torch.cat((self.value_full, value_states), dim=-2) if self.value_full is not None else value_states
        # Official decode path quantizes a complete key residual block and one
        # overflowing value token at a time.  The loop also handles a caller
        # that supplies more than one decode token.
        while self.key_full is not None and self.key_full.shape[-2] >= self.cfg.residual_length:
            block = self.key_full[..., : self.cfg.residual_length, :].contiguous()
            self._append_key_block(block)
            self.key_full = self.key_full[..., self.cfg.residual_length :, :].contiguous()
        while self.value_full is not None and self.value_full.shape[-2] > self.cfg.residual_length:
            block = self.value_full[..., :1, :].contiguous()
            self._append_value_block(block)
            self.value_full = self.value_full[..., 1:, :].contiguous()

    def quantized_key(self) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        return self.key_code, self.key_scale, self.key_min

    def quantized_value(self) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        return self.value_code, self.value_scale, self.value_min

    def full_key(self) -> Optional[torch.Tensor]:
        return self.key_full

    def full_value(self) -> Optional[torch.Tensor]:
        return self.value_full

    # CacheLayerMixin-compatible helpers used by beam generation and tests.
    def batch_repeat_interleave(self, repeats: int) -> None:
        for name in ("key_code", "key_scale", "key_min", "key_full", "value_code", "value_scale", "value_min", "value_full"):
            tensor = getattr(self, name)
            if tensor is not None:
                setattr(self, name, tensor.repeat_interleave(repeats, dim=0))

    def batch_select_indices(self, indices: torch.Tensor) -> None:
        for name in ("key_code", "key_scale", "key_min", "key_full", "value_code", "value_scale", "value_min", "value_full"):
            tensor = getattr(self, name)
            if tensor is not None:
                setattr(self, name, tensor.index_select(0, indices.to(tensor.device)))

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        self.batch_select_indices(beam_idx)

    def crop(self, max_length: int) -> None:
        if max_length < 0:
            max_length = self.get_seq_length() - abs(max_length)
        max_length = max(0, int(max_length))
        quant_len = 0 if self.key_code is None else self.key_code.shape[-1] * (32 // self.cfg.bits)
        if quant_len > max_length:
            # Reconstructing and repacking a cropped packed prefix is not part
            # of ordinary generation; reject instead of silently changing the
            # cache algorithm.
            raise NotImplementedError("cropping inside a packed KIVI prefix is unsupported")
        keep_full = max_length - quant_len
        if self.key_full is not None:
            self.key_full = self.key_full[..., :keep_full, :]
        if self.value_full is not None:
            self.value_full = self.value_full[..., :keep_full, :]

    def reset(self) -> None:
        for name in ("key_code", "key_scale", "key_min", "key_full", "value_code", "value_scale", "value_min", "value_full"):
            setattr(self, name, None)
        self.is_initialized = False

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor, cache_kwargs: Optional[dict[str, Any]] = None):
        self.append(key_states, value_states)
        # The patched Qwen2 attention does not consume this materialized
        # result.  Returning a full tensor keeps generic Cache callers sane.
        return self.materialize_key(), self.materialize_value()

    def materialize_key(self) -> torch.Tensor:
        if self.key_code is None:
            return self.key_full
        unpack = _official_unpack()
        # Official KIVI stores keys transposed as [B,H,D,T/pack].  Its
        # ``triton_quantize_and_pack_along_last_dim`` output is decoded by
        # the v-cache unpack helper and transposed back to [B,H,T,D].
        prefix = unpack[1](
            self.key_code,
            self.key_scale.unsqueeze(-1),
            self.key_min.unsqueeze(-1),
            self.cfg.group_size,
            self.cfg.bits,
        )
        prefix = prefix.transpose(-1, -2).to(self.dtype or prefix.dtype)
        return prefix if self.key_full is None else torch.cat((prefix, self.key_full), dim=-2)

    def materialize_value(self) -> torch.Tensor:
        if self.value_code is None:
            return self.value_full
        unpack = _official_unpack()
        prefix = unpack[1](
            self.value_code,
            self.value_scale.unsqueeze(-1),
            self.value_min.unsqueeze(-1),
            self.cfg.group_size,
            self.cfg.bits,
        )
        prefix = prefix.to(self.dtype or prefix.dtype)
        return prefix if self.value_full is None else torch.cat((prefix, self.value_full), dim=-2)


def _official_unpack() -> tuple[Any, Any]:
    """Load official dequant helpers for generic cache introspection."""

    for root in _kivi_roots():
        for path in (root, os.path.join(root, "quant")):
            if path not in sys.path:
                sys.path.insert(0, path)
        try:
            pack = importlib.import_module("quant.new_pack")
            return pack.unpack_and_dequant_kcache, pack.unpack_and_dequant_vcache
        except Exception:
            continue
    raise RuntimeError("official KIVI unpack helpers are unavailable")


class KIVICache:
    """Transformers ``Cache`` implementation backed by KIVI packed layers."""

    def __init__(self, config: Any, kivi_config: KIVIQwen2Config):
        from transformers.cache_utils import Cache

        self.config = config
        self.kivi_config = kivi_config
        n_layers = int(getattr(config, "num_hidden_layers", 0))
        if n_layers <= 0:
            raise ValueError("Qwen2 config must expose num_hidden_layers")
        self.layers = [KIVILayer(kivi_config) for _ in range(n_layers)]
        self.offloading = False
        self.only_non_sliding = False
        # Number of left-pad columns removed from the first prefill.  VeRL
        # keeps the original padded width in later decode masks; the KIVI
        # cache must remove the same columns before applying those masks.
        self._qat_kivi_left_padding = 0
        self._cache_base = Cache

    # Delegate the public Cache surface without inheriting from Cache.  This
    # avoids abstract/cache internals changing across Transformers versions.
    def get_seq_length(self, layer_idx: int = 0) -> int:
        return self.layers[layer_idx].get_seq_length() if layer_idx < len(self.layers) else 0

    def get_max_cache_shape(self) -> int:
        return -1

    def get_mask_sizes(self, cache_position: torch.Tensor, layer_idx: int) -> tuple[int, int]:
        return self.layers[layer_idx].get_mask_sizes(cache_position)

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor, layer_idx: int, cache_kwargs: Optional[dict[str, Any]] = None):
        return self.layers[layer_idx].update(key_states, value_states, cache_kwargs)

    def batch_repeat_interleave(self, repeats: int) -> None:
        for layer in self.layers:
            layer.batch_repeat_interleave(repeats)

    def batch_select_indices(self, indices: torch.Tensor) -> None:
        for layer in self.layers:
            layer.batch_select_indices(indices)

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        self.batch_select_indices(beam_idx)

    def crop(self, max_length: int) -> None:
        for layer in self.layers:
            layer.crop(max_length)

    def reset(self) -> None:
        for layer in self.layers:
            layer.reset()

    def is_initialized(self) -> bool:
        return any(layer.is_initialized for layer in self.layers)

    def to_legacy_cache(self):
        return tuple((layer.materialize_key(), layer.materialize_value()) for layer in self.layers)


def _repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return x
    b, h, *rest = x.shape
    return x[:, :, None, *([slice(None)] * len(rest))].expand(b, h, n_rep, *rest).reshape(b, h * n_rep, *rest)


def _apply_mask(scores: torch.Tensor, attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
    if attention_mask is None:
        return scores
    if attention_mask.shape[-1] != scores.shape[-1]:
        raise ValueError(
            f"KIVI attention mask length {attention_mask.shape[-1]} does not match cache length {scores.shape[-1]}"
        )
    return scores + attention_mask


def _drop_left_padding_for_kivi_prefill(
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Remove left-pad positions before invoking the official KIVI packer.

    VeRL pads prompts to one common width.  Packing those pad-token K/V rows
    can create an all-constant quantization group (zero scale), which the
    upstream KIVI CUDA pack kernel does not define.  Generation is already
    processed one prompt at a time by ``KIVIHFRollout``; therefore trimming a
    single row is equivalent to the unpadded prompt and leaves the official
    KIVI algorithm and kernels untouched.  The mask is sliced on its key
    dimension so causal/padding masking remains valid for the unpadded cache.
    """
    if attention_mask is None or attention_mask.ndim < 2:
        return key_states, value_states, attention_mask
    token_count = key_states.shape[-2]
    # The rollout adapter deliberately sends one prompt per call.  Refuse a
    # heterogeneous batch rather than silently choosing one row's indices.
    if attention_mask.shape[-1] < token_count:
        return key_states, value_states, attention_mask
    if attention_mask.ndim == 2:
        valid = attention_mask[:, -token_count:] > 0
    else:
        # Qwen2's prepared causal mask is [B,1,Q,K] (some versions use
        # [B,Q,K]); the final query row exposes the key-side pad mask.
        valid = attention_mask[..., -1, -token_count:]
        if valid.ndim > 2:
            valid = valid.reshape(valid.shape[0], -1, token_count)[:, -1, :]
        if valid.is_floating_point():
            valid = valid > -1.0e4
        else:
            valid = valid.to(dtype=torch.bool)
    if valid.shape[-1] != token_count:
        return key_states, value_states, attention_mask
    if valid.shape[0] != 1:
        counts = valid.sum(dim=-1)
        if not torch.all(counts == counts[0]):
            raise ValueError("KIVI prefill requires equal valid prompt lengths per batch row")
    indices = torch.nonzero(valid[0], as_tuple=False).flatten()
    if indices.numel() == token_count:
        return key_states, value_states, attention_mask
    if indices.numel() == 0:
        raise ValueError("KIVI prefill received an all-padding prompt")
    key_states = key_states.index_select(-2, indices.to(key_states.device))
    value_states = value_states.index_select(-2, indices.to(value_states.device))
    # Attention masks used by Qwen2 are normally [B,1,Q,K].  Keep this
    # generic for 2D/3D masks as well by selecting only the key dimension.
    attention_mask = attention_mask.index_select(-1, indices.to(attention_mask.device))
    return key_states, value_states, attention_mask


def _kivi_attention_forward(
    self: Any,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    past_key_values: Optional[Any] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs: Any,
):
    """Qwen2 attention forward using KIVI cache or QAT fake quant."""

    from transformers.models.qwen2.modeling_qwen2 import (
        ALL_ATTENTION_FUNCTIONS,
        apply_rotary_pos_emb,
        eager_attention_forward,
    )

    attention_interface = eager_attention_forward
    attn_impl = getattr(getattr(self, "config", None), "_attn_implementation", "eager")
    if attn_impl != "eager":
        attention_interface = ALL_ATTENTION_FUNCTIONS[attn_impl]

    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)
    query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    kivi_cache = past_key_values if isinstance(past_key_values, KIVICache) else None
    if kivi_cache is None:
        # Training path: same post-RoPE placement as KIVI, but use the
        # repository's STE fake quantizer so gradients reach the actor.
        if bool(getattr(self, "_qat_kivi_training_fake_quant", False)):
            key_states = _kivi_fake_quant_key_ste(
                key_states.contiguous(), self._qat_kivi_config
            )
            value_states = _kivi_fake_quant_value_ste(
                value_states.contiguous(), self._qat_kivi_config
            )
        # Preserve the official Qwen2 attention interface for non-cache calls.
        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        return self.o_proj(attn_output), attn_weights

    layer = kivi_cache.layers[self.layer_idx]
    if cache_position is None:
        raise ValueError("KIVI rollout requires cache_position")
    if layer.get_seq_length() == 0:
        original_token_count = key_states.shape[-2]
        key_states, value_states, attention_mask = _drop_left_padding_for_kivi_prefill(
            key_states, value_states, attention_mask
        )
        if kivi_cache is not None:
            kivi_cache._qat_kivi_left_padding = max(
                0, int(original_token_count - key_states.shape[-2])
            )
        layer.prefill(key_states, value_states)
        # Official KIVI uses an unquantized prefill attention; only the cache
        # representation is quantized at this point.  Use Transformers'
        # memory-efficient attention backend rather than materializing the
        # quadratic score matrix for long prompts.
        # ``eager_attention_forward`` (and the SDPA/flash attention
        # interfaces) repeat grouped K/V internally.  Pass the compact
        # Qwen2 K/V tensors here; pre-repeating them would multiply the
        # number of heads a second time.
        key_for_attention = key_states
        value_for_attention = value_states
        attn_output, _ = attention_interface(
            self,
            query_states,
            key_for_attention,
            value_for_attention,
            attention_mask,
            dropout=0.0,
            scaling=self.scaling,
            **kwargs,
        )
    else:
        layer.append(key_states, value_states)
        key_code, key_scale, key_min = layer.quantized_key()
        value_code, value_scale, value_min = layer.quantized_value()
        key_full = layer.full_key()
        value_full = layer.full_value()
        scores_parts: list[torch.Tensor] = []
        query_scaled = query_states.float() * float(self.scaling)
        if key_code is not None:
            _, packed_matmul = load_official_kivi_kernels()
            # Apply the attention scaling before the official FP16 packed
            # GEMV.  The upstream reference scales its FP16 result after the
            # kernel, but large late-layer dot products can overflow FP16
            # before that multiplication; scaling the query is algebraically
            # identical and keeps the kernel's accumulation in range.
            q = query_scaled.to(torch.float16)
            kcode = _repeat_kv(key_code, self.num_key_value_groups)
            kscale = _repeat_kv(key_scale, self.num_key_value_groups)
            kmin = _repeat_kv(key_min, self.num_key_value_groups)
            packed_scores = _kivi_gemv_per_batch(
                packed_matmul,
                self.group_size if hasattr(self, "group_size") else layer.cfg.group_size,
                q,
                kcode,
                kscale,
                kmin,
                layer.cfg.bits,
            )
            scores_parts.append(packed_scores.float())
        if key_full is not None and key_full.shape[-2] > 0:
            key_full_rep = _repeat_kv(key_full, self.num_key_value_groups)
            scores_parts.append(torch.matmul(query_scaled, key_full_rep.float().transpose(-1, -2)))
        if not scores_parts:
            raise RuntimeError("KIVI cache has no key states after append")
        scores = torch.cat(scores_parts, dim=-1)
        left_padding = int(getattr(kivi_cache, "_qat_kivi_left_padding", 0))
        if (
            left_padding
            and attention_mask is not None
            and attention_mask.shape[-1] > scores.shape[-1]
        ):
            attention_mask = attention_mask[..., left_padding:]
        scores = _apply_mask(scores, attention_mask)
        probs = torch.softmax(scores, dim=-1, dtype=torch.float32).to(query_states.dtype)
        # Value packing is along the head-dimension (last axis), so the
        # number of cached value tokens is the penultimate dimension.  Keys
        # pack along the token axis and therefore use ``shape[-1]`` above.
        quant_len = 0 if value_code is None else value_code.shape[-2]
        output_parts: list[torch.Tensor] = []
        if value_code is not None:
            _, packed_matmul = load_official_kivi_kernels()
            vcode = _repeat_kv(value_code, self.num_key_value_groups)
            vscale = _repeat_kv(value_scale, self.num_key_value_groups)
            vmin = _repeat_kv(value_min, self.num_key_value_groups)
            output_parts.append(
                _kivi_gemv_per_batch(
                    packed_matmul,
                    layer.cfg.group_size,
                    probs[..., :quant_len].to(torch.float16),
                    vcode,
                    vscale,
                    vmin,
                    layer.cfg.bits,
                ).to(query_states.dtype)
            )
        if value_full is not None and value_full.shape[-2] > 0:
            value_full_rep = _repeat_kv(value_full, self.num_key_value_groups)
            output_parts.append(torch.matmul(probs[..., quant_len:], value_full_rep))
        attn_output = sum(output_parts)

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    return self.o_proj(attn_output), None


def _is_qwen2_attention(module: Any) -> bool:
    return all(hasattr(module, name) for name in ("q_proj", "k_proj", "v_proj", "o_proj", "num_key_value_groups"))


def _kivi_gemv_per_batch(
    packed_matmul: Any,
    group_size: int,
    query: torch.Tensor,
    packed: torch.Tensor,
    scale: torch.Tensor,
    minimum: torch.Tensor,
    bits: int,
) -> torch.Tensor:
    """Use the official GEMV independently for each batch row.

    The shipped outer-dimension extension is reliable for one sample, but its
    grouped-head indexing diverges for Qwen GQA when a packed prefix grows and
    several samples are passed in one call.  Splitting only the batch axis
    keeps the official kernel and leaves the surrounding attention batched.
    """
    if query.shape[0] == 1:
        return packed_matmul(group_size, query, packed, scale, minimum, bits)
    return torch.cat(
        [
            packed_matmul(
                group_size,
                query[i : i + 1],
                packed[i : i + 1],
                scale[i : i + 1],
                minimum[i : i + 1],
                bits,
            )
            for i in range(query.shape[0])
        ],
        dim=0,
    )


def install_kivi_hf_rollout(
    model: nn.Module,
    *,
    bits: int = 2,
    group_size: int = 32,
    residual_length: int = 32,
    qat_clip_range: tuple[float, float] = (-2.0, 2.0),
    training_fake_quant: bool = True,
    require_kernel: bool = True,
) -> dict[str, Any]:
    """Install KIVI attention/cache on every Qwen2 attention module.

    This must run after ``from_pretrained`` and before FSDP wrapping.  It is
    intentionally actor-only; callers should never invoke it for the frozen
    reward/teacher model.
    """

    cfg = KIVIQwen2Config(bits, group_size, residual_length, qat_clip_range)
    if require_kernel:
        # Fail at model construction, before FSDP/Ray starts expensive work,
        # when the pinned official extension has not been built.
        load_official_kivi_kernels()
    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None:
        raise TypeError("KIVI Qwen2 adapter requires model.model.layers")
    attention_modules: list[Any] = []
    for layer in layers:
        attention = getattr(layer, "self_attn", None)
        if attention is None or not _is_qwen2_attention(attention):
            raise TypeError("model contains a non-Qwen2 attention layer; refusing partial KIVI install")
        if getattr(attention, "_qat_kivi_installed", False):
            attention_modules.append(attention)
            continue
        attention._qat_kivi_config = cfg
        attention._qat_kivi_training_fake_quant = bool(training_fake_quant)
        attention.forward = MethodType(_kivi_attention_forward, attention)
        attention._qat_kivi_installed = True
        attention_modules.append(attention)

    original_prepare = getattr(model, "_prepare_cache_for_generation", None)
    if not callable(original_prepare):
        raise TypeError("model has no Transformers generation cache preparation method")
    if not getattr(model, "_qat_kivi_prepare_installed", False):
        def _prepare(self: Any, generation_config: Any, model_kwargs: dict[str, Any], generation_mode: Any, batch_size: int, max_cache_length: int):
            implementation = getattr(generation_config, "cache_implementation", None)
            if implementation not in (None, "dynamic", "dynamic_full"):
                raise ValueError("official KIVI rollout requires the dynamic cache implementation")
            result = original_prepare(generation_config, model_kwargs, generation_mode, batch_size, max_cache_length)
            cache_name = "past_key_values"
            existing = model_kwargs.get(cache_name)
            if existing is None or existing.__class__.__name__ in {"DynamicCache", "DynamicSlidingWindowCache"}:
                model_kwargs[cache_name] = KIVICache(self.config.get_text_config(decoder=True), cfg)
            elif not isinstance(existing, KIVICache):
                raise TypeError("KIVI rollout received a non-dynamic cache")
            return result

        model._prepare_cache_for_generation = MethodType(_prepare, model)
        model._qat_kivi_prepare_installed = True

    metadata = {
        "enabled": True,
        "bits": cfg.bits,
        "group_size": cfg.group_size,
        "residual_length": cfg.residual_length,
        "cache": "official_kivi_packed_prefix_plus_residual",
        "kernel": "official_kivi_gemv",
        "kernel_verified": bool(require_kernel),
        "training": "fake_dequant_ste" if training_fake_quant else "disabled",
        "attention_modules": len(attention_modules),
        "model_type": getattr(getattr(model, "config", None), "model_type", None),
    }
    model._qat_kivi_metadata = metadata
    return metadata


__all__ = [
    "KIVI_COMMIT",
    "KIVIQwen2Config",
    "KIVICache",
    "KIVILayer",
    "install_kivi_hf_rollout",
    "load_official_kivi_kernels",
]
