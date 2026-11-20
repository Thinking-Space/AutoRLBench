# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""OSCAR INT2 KV-cache attention backend for vLLM.

Prefill: standard attention on uncompressed (BF16) K/V, then rotate, clip
         and store K/V as INT2 into a combined cache slot.
Decode:  rotate the query by ``R_k``, score against the INT2 keys, dequant
         the INT2 values, weighted-sum, then map the output back with
         ``R_v^T``.

Cache layout (no leading 2 dimension), mirroring TurboQuant:
    (num_blocks, block_size, num_kv_heads, slot_size_aligned)
with each slot ``[key_packed | value_packed]`` and each region
``[int2_data | fp16 scale | fp16 zero]``.
"""

import os
from dataclasses import dataclass
from typing import Any, ClassVar

import torch
import triton
import triton.language as tl
import torch.nn.functional as F

from vllm.config import get_current_vllm_config
from vllm.config.cache import CacheDType
from vllm.model_executor.layers.quantization.oscar.config import OscarConfig
from vllm.model_executor.layers.quantization.oscar.rotation import get_layer_rotation
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionImpl,
    AttentionLayer,
    AttentionMetadata,
    AttentionMetadataBuilder,
    AttentionType,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.attention.backends.fa_utils import (
    get_flash_attn_version,
    is_flash_attn_varlen_func_available,
)
from vllm.v1.attention.backends.utils import split_decodes_and_prefills
from vllm.v1.attention.ops.triton_oscar_decode import (
    oscar_decode_attention,
    oscar_full_dequant_kv,
)
from vllm.v1.attention.ops.triton_oscar_store import oscar_store

_HAS_FLASH_ATTN = is_flash_attn_varlen_func_available()
if _HAS_FLASH_ATTN:
    from vllm.v1.attention.backends.fa_utils import flash_attn_varlen_func


class OscarAttentionBackend(AttentionBackend):
    """Attention backend using OSCAR INT2 KV-cache compression."""

    accept_output_buffer: bool = True
    forward_includes_kv_cache_update: bool = False

    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16, torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = ["oscar_int2"]

    @staticmethod
    def get_name() -> str:
        return "OSCAR"

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [16, 32, 64, 128]

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        return attn_type == AttentionType.DECODER

    @classmethod
    def supports_per_head_quant_scales(cls) -> bool:
        return False

    @staticmethod
    def get_impl_cls() -> type["OscarAttentionImpl"]:
        return OscarAttentionImpl

    @staticmethod
    def get_builder_cls() -> type["OscarMetadataBuilder"]:
        return OscarMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "oscar_int2",
    ) -> tuple[int, ...]:
        # The engine passes "auto" for layers whose spec reports no KV
        # quantization. Only OSCAR-quantized layers reach this backend —
        # boundary-skip layers are given kv_cache_dtype="auto" in
        # Attention.__init__, so they select a native backend and form their
        # own attention group — hence "auto" here still means the OSCAR slot.
        cfg = OscarConfig.from_cache_dtype(
            OscarConfig.resolve_cache_dtype(cache_dtype_str), head_size
        )
        return (num_blocks, block_size, num_kv_heads, cfg.slot_size_aligned)

    @classmethod
    def supports_kv_cache_dtype(cls, kv_cache_dtype: CacheDType | None) -> bool:
        return kv_cache_dtype is not None and kv_cache_dtype.startswith("oscar_")

    @classmethod
    def supports_head_size(cls, head_size: int) -> bool:
        return head_size > 0


@dataclass
class OscarMetadata(AttentionMetadata):
    seq_lens: torch.Tensor
    slot_mapping: torch.Tensor
    block_table: torch.Tensor
    query_start_loc: torch.Tensor
    num_actual_tokens: int = 0
    max_query_len: int = 0
    max_seq_len: int = 0
    is_prefill: bool = False
    num_decodes: int = 0
    num_decode_tokens: int = 0
    # True when any request in the batch attends context beyond its own new
    # tokens (prefix-cache hit or chunked-prefill continuation). Guards the
    # flash-attn first-chunk fast path.
    has_context: bool = False
    query_start_loc_cpu: torch.Tensor | None = None
    seq_lens_cpu: torch.Tensor | None = None



_OSCAR_ARANGE_CACHE: dict = {}

# Which implementation serves the staged-BF16 sink+tail attention:
#   0  eager einsum over the gathered staging pages   (default)
#   1  _oscar_bf16_tail_kernel        -- online softmax, one position per step
#   2  _oscar_bf16_tail_tiled_kernel  -- online softmax, one page per step
#
# The default stays eager on purpose.  Both fused paths are checked against the
# eager one in tests/test_oscar_bf16_tail.py, but equivalence is not the same as
# being faster, and the eager form is a dense einsum on tensor cores while the
# fused ones serialise the reduction to avoid a [B, P, bs, Hk, D] fp32 gather.
# That trade has to be measured at the shapes this backend actually runs; until
# a benchmark picks a winner, a rollout engine should not be where it gets
# discovered.
_BF16_TAIL_MODE = int(os.environ.get("VLLM_OSCAR_BF16_TAIL_MODE", "0"))


def _oscar_arange(n: int, device) -> torch.Tensor:
    """Cached torch.arange(n, device=device).

    ``_decode_attention_windowed`` rebuilds `arange(bs)` and `arange(maxpg)`
    on every call -- once per layer per decode step -- although both depend
    only on static shapes.  py-spy put this function at the top of the
    training-side rollout profile (gen 1200s on the 1.5B OSCAR arm), so the
    allocation and launch cost is worth removing.
    """
    key = (n, str(device))
    t = _OSCAR_ARANGE_CACHE.get(key)
    if t is None:
        t = torch.arange(n, device=device)
        _OSCAR_ARANGE_CACHE[key] = t
    return t



@triton.jit
def _oscar_staging_write_kernel(
    key_ptr, val_ptr, stage_k_ptr, stage_v_ptr, owner_ptr,
    slot_ptr, pos_ptr,
    stride_kt, stride_kh, stride_sr, stride_so, stride_sh,
    stride_ot, stride_oo,
    seq_ptr, req_ptr,
    n_tokens, n_heads, head_dim, block_sz, n_rows,
    sink_eff, recent_tokens,
    BLOCK_D: tl.constexpr,
    PHASE: tl.constexpr,
):
    """One program per (token, kv-head): decide window membership and store.

    Replaces the eager implementation, which py-spy measured at ~24% of the
    rollout worker's самples: it built `repeat_interleave`, several boolean
    index_put_ calls and half a dozen intermediates per layer per decode step.
    The predicate and the store are both per-token, so they fuse trivially.

    Owner-tag semantics are preserved: the tag is written first and the data
    only by the writer whose tag survived, so a stale tag can never point at
    mixed contents (an evicted token falls back to its INT2 copy).
    """
    t = tl.program_id(0)
    h = tl.program_id(1)
    if t >= n_tokens:
        return

    slot = tl.load(slot_ptr + t).to(tl.int64)
    if slot < 0:
        return
    pos = tl.load(pos_ptr + t).to(tl.int64)
    r = tl.load(req_ptr + t).to(tl.int64)
    seq = tl.load(seq_ptr + r).to(tl.int64)

    in_recent = pos >= (seq - recent_tokens)
    in_sink = pos < sink_eff
    if not (in_recent or in_sink):
        return

    blk = slot // block_sz
    off = slot % block_sz
    row = blk % n_rows

    if PHASE == 0:
        # Phase 1: tags only.  Contended slots resolve to whichever program
        # stores last, exactly as index_put_ does.
        tl.store(owner_ptr + row * stride_ot + off * stride_oo, blk)
        return

    # Phase 2: the kernel boundary between phases is a device-wide barrier, so
    # every tag is final here and this read decides the single winner.
    won = tl.load(owner_ptr + row * stride_ot + off * stride_oo) == blk
    if not won:
        return

    d = tl.arange(0, BLOCK_D)
    mask = d < head_dim
    k = tl.load(key_ptr + t * stride_kt + h * stride_kh + d, mask=mask, other=0.0)
    v = tl.load(val_ptr + t * stride_kt + h * stride_kh + d, mask=mask, other=0.0)
    base = row * stride_sr + off * stride_so + h * stride_sh + d
    tl.store(stage_k_ptr + base, k.to(stage_k_ptr.dtype.element_ty), mask=mask)
    tl.store(stage_v_ptr + base, v.to(stage_v_ptr.dtype.element_ty), mask=mask)



@triton.jit
def _oscar_window_bounds_kernel(
    owner_ptr, bt_ptr, seq_ptr, cut_ptr, si_ptr,
    stride_ot, stride_oo, stride_bt,
    n_req, maxpg, block_sz, n_rows, S_nb, S_eff, W,
    BLOCK_BS: tl.constexpr,
):
    """One program per request: compute the BF16 tail cut and sink flag.

    The eager version materialises [B, TP, bs] and [B, S_nb, bs] boolean
    tensors, runs flip/cumprod/flip to find the contiguous staged suffix, then
    reduces to two [B] vectors.  py-spy put the enclosing function at ~37% of
    the rollout worker's samples.  The reduction is per-request and sequential
    in page order, so a single program per request does it with no
    intermediates at all.

    Semantics preserved exactly:
      sink_active = seq > S_eff AND every sink slot is staged
      cut         = earliest position p in [max(seq-W, si*bs), seq) such that
                    every slot from p to seq-1 is staged; seq if none
    """
    r = tl.program_id(0)
    if r >= n_req:
        return
    seq = tl.load(seq_ptr + r).to(tl.int64)
    offs = tl.arange(0, BLOCK_BS)
    off_ok = offs < block_sz

    # ---- sink: BF16 only when every sink position is staged ----
    sink_ok = seq > S_eff
    p = 0
    while p < S_nb:
        blk = tl.load(bt_ptr + r * stride_bt + p).to(tl.int64)
        own = tl.load(
            owner_ptr + (blk % n_rows) * stride_ot + offs * stride_oo,
            mask=off_ok, other=-1,
        )
        staged = (own == blk) | (~off_ok)
        sink_ok = sink_ok & (tl.sum(staged.to(tl.int32)) == BLOCK_BS)
        p += 1
    si = tl.where(sink_ok, S_nb, 0)
    tl.store(si_ptr + r, si)

    # ---- recent tail: walk backwards, stop at the first unstaged slot ----
    t0 = tl.maximum(seq - W, si * block_sz)
    cut = seq
    pos = seq - 1
    done = False
    while (pos >= t0) and (not done):
        pg = pos // block_sz
        blk = tl.load(
            bt_ptr + r * stride_bt + tl.minimum(pg, maxpg - 1)
        ).to(tl.int64)
        o = pos % block_sz
        own = tl.load(owner_ptr + (blk % n_rows) * stride_ot + o * stride_oo)
        if (own == blk) and (pg >= 0):
            cut = pos
            pos -= 1
        else:
            done = True
    tl.store(cut_ptr + r, cut)



@triton.jit
def _oscar_bf16_tail_kernel(
    q_ptr, stage_k_ptr, stage_v_ptr, blk_ptr, valid_ptr,
    o_ptr, lse_ptr,
    stride_qb, stride_qh, stride_sr, stride_so, stride_sh,
    stride_bb, stride_vb, stride_vp,
    stride_ob, stride_oh, stride_lb,
    n_pages, block_sz, n_rows, n_kv_heads, gqa, head_dim, scale,
    BLOCK_D: tl.constexpr,
):
    """Flash-style attention over the staged BF16 sink+tail segments.

    The eager form gathers the whole staging arena for the selected pages,
    upcasts it to fp32, scores every one of L = n_pages * block_sz positions
    and then masks most of them away.  Here each program owns one
    (batch, kv-head, gqa-slot) and streams the pages with an online softmax,
    so invalid positions cost a predicated load instead of a full row of the
    score matrix.

    Numerics follow the eager path exactly: max-subtracted exponentials, sum
    in fp32, and an LSE of -inf when the segment is empty so the caller's
    logaddexp merge is unchanged.
    """
    b = tl.program_id(0)
    kh = tl.program_id(1)
    gi = tl.program_id(2)

    d = tl.arange(0, BLOCK_D)
    dmask = d < head_dim
    qh_off = kh * gqa + gi
    q = tl.load(q_ptr + b * stride_qb + qh_off * stride_qh + d, mask=dmask, other=0.0).to(tl.float32)

    m_i = float("-inf")
    l_i = 0.0
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)

    p = 0
    while p < n_pages:
        blk = tl.load(blk_ptr + b * stride_bb + p).to(tl.int64)
        row = blk % n_rows
        o = 0
        while o < block_sz:
            ok = tl.load(valid_ptr + b * stride_vb + p * stride_vp + o)
            if ok != 0:
                base = row * stride_sr + o * stride_so + kh * stride_sh + d
                k = tl.load(stage_k_ptr + base, mask=dmask, other=0.0).to(tl.float32)
                v = tl.load(stage_v_ptr + base, mask=dmask, other=0.0).to(tl.float32)
                sc = tl.sum(q * k) * scale
                m_new = tl.maximum(m_i, sc)
                alpha = tl.exp(m_i - m_new)
                pexp = tl.exp(sc - m_new)
                acc = acc * alpha + v * pexp
                l_i = l_i * alpha + pexp
                m_i = m_new
            o += 1
        p += 1

    out = tl.where(l_i > 0, acc / tl.maximum(l_i, 1e-38), 0.0)
    tl.store(o_ptr + b * stride_ob + qh_off * stride_oh + d, out, mask=dmask)
    lse = tl.where(l_i > 0, m_i + tl.log(tl.maximum(l_i, 1e-38)), float("-inf"))
    tl.store(lse_ptr + b * stride_lb + qh_off, lse)


@triton.jit
def _oscar_bf16_tail_tiled_kernel(
    q_ptr, stage_k_ptr, stage_v_ptr, blk_ptr, valid_ptr,
    o_ptr, lse_ptr,
    stride_qb, stride_qh, stride_sr, stride_so, stride_sh,
    stride_bb, stride_vb, stride_vp,
    stride_ob, stride_oh, stride_lb,
    n_pages, block_sz, n_rows, n_kv_heads, gqa, head_dim, scale,
    BLOCK_D: tl.constexpr,
    BLOCK_BS: tl.constexpr,
):
    """Page-tiled variant of ``_oscar_bf16_tail_kernel``.

    Same online softmax and the same ``-inf``-on-empty contract, but one whole
    staging page (``block_sz`` slots) is scored per iteration as a
    ``[BLOCK_BS, BLOCK_D]`` tile.  The serial variant issues one ``tl.sum``
    reduction per *position*: at the shapes this backend actually runs
    (``n_pages`` around 20, ``block_sz`` 16) that is ~320 dependent reductions
    per program, which is why it can lose to the dense einsum it replaces even
    though it moves far less memory.  Which of the three paths is fastest is a
    measurement, not something this comment can settle.

    The running max starts at a finite sentinel rather than ``-inf``.  The
    serial kernel can use ``-inf`` because its ``if ok != 0`` guard skips the
    rescale entirely for invalid slots; here the rescale is unconditional, and
    ``-inf - (-inf)`` is NaN, which would poison ``acc`` for any all-invalid
    page.  The sentinel never reaches the output: emptiness is decided by
    ``l_i > 0``, exactly as in the eager path.
    """
    b = tl.program_id(0)
    kh = tl.program_id(1)
    gi = tl.program_id(2)

    d = tl.arange(0, BLOCK_D)
    dmask = d < head_dim
    o = tl.arange(0, BLOCK_BS)
    omask = o < block_sz

    qh_off = kh * gqa + gi
    q = tl.load(
        q_ptr + b * stride_qb + qh_off * stride_qh + d, mask=dmask, other=0.0
    ).to(tl.float32)

    m_i = -1.0e38
    l_i = 0.0
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)

    p = 0
    while p < n_pages:
        blk = tl.load(blk_ptr + b * stride_bb + p).to(tl.int64)
        row = blk % n_rows
        ok = tl.load(
            valid_ptr + b * stride_vb + p * stride_vp + o, mask=omask, other=0
        )
        base = (
            row * stride_sr + o[:, None] * stride_so + kh * stride_sh + d[None, :]
        )
        tile = omask[:, None] & dmask[None, :]
        k = tl.load(stage_k_ptr + base, mask=tile, other=0.0).to(tl.float32)
        v = tl.load(stage_v_ptr + base, mask=tile, other=0.0).to(tl.float32)

        keep = omask & (ok != 0)
        sc = tl.sum(q[None, :] * k, axis=1) * scale
        sc = tl.where(keep, sc, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(sc, axis=0))
        alpha = tl.exp(m_i - m_new)
        pexp = tl.where(keep, tl.exp(sc - m_new), 0.0)
        acc = acc * alpha + tl.sum(pexp[:, None] * v, axis=0)
        l_i = l_i * alpha + tl.sum(pexp, axis=0)
        m_i = m_new
        p += 1

    out = tl.where(l_i > 0, acc / tl.maximum(l_i, 1e-38), 0.0)
    tl.store(o_ptr + b * stride_ob + qh_off * stride_oh + d, out, mask=dmask)
    lse = tl.where(l_i > 0, m_i + tl.log(tl.maximum(l_i, 1e-38)), float("-inf"))
    tl.store(lse_ptr + b * stride_lb + qh_off, lse)


class OscarMetadataBuilder(AttentionMetadataBuilder[OscarMetadata]):
    # The BF16 sink/recent staging path uses data-dependent block allocation
    # and has only been validated in eager mode.
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.NEVER

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self._init_reorder_batch_threshold(1, supports_spec_as_decode=False)

    def build_for_cudagraph_capture(
        self, common_attn_metadata: CommonAttentionMetadata
    ) -> OscarMetadata:
        m = self.build(0, common_attn_metadata)
        m.seq_lens.fill_(1)
        return m

    def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        cam = common_attn_metadata
        assert self.reorder_batch_threshold is not None
        num_decodes, num_prefills, num_decode_tokens, _ = split_decodes_and_prefills(
            cam, decode_threshold=self.reorder_batch_threshold
        )
        has_context = False
        if cam.query_start_loc_cpu is not None and cam.seq_lens_cpu is not None:
            q_lens_cpu = cam.query_start_loc_cpu[1:] - cam.query_start_loc_cpu[:-1]
            has_context = bool((cam.seq_lens_cpu > q_lens_cpu).any())
        return OscarMetadata(
            seq_lens=cam.seq_lens,
            slot_mapping=cam.slot_mapping,
            block_table=cam.block_table_tensor,
            query_start_loc=cam.query_start_loc,
            num_actual_tokens=cam.num_actual_tokens,
            max_query_len=cam.max_query_len,
            max_seq_len=cam.max_seq_len,
            is_prefill=(cam.max_query_len > 1),
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            has_context=has_context,
            query_start_loc_cpu=cam.query_start_loc_cpu,
            seq_lens_cpu=cam.seq_lens_cpu_upper_bound,
        )


class OscarAttentionImpl(AttentionImpl["OscarMetadata"]):
    supports_quant_query_input: bool = False

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int | None = None,
        alibi_slopes: list[float] | None = None,
        sliding_window: int | None = None,
        kv_cache_dtype: str = "auto",
        logits_soft_cap: float | None = None,
        attn_type: str = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
        **kwargs,
    ):
        if sliding_window is not None:
            raise NotImplementedError(
                "OSCAR INT2 KV cache does not support sliding-window attention "
                "layers; add 'sliding_window' to kv_cache_dtype_skip_layers to "
                "keep them in the native dtype."
            )
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = scale
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.kv_cache_dtype = kv_cache_dtype

        self.cfg = OscarConfig.from_cache_dtype(kv_cache_dtype, head_size)
        self.fa_version = get_flash_attn_version(head_size=head_size)
        vllm_config = get_current_vllm_config()
        self.max_num_kv_splits = (
            vllm_config.attention_config.tq_max_kv_splits_for_cuda_graph
        )
        # BF16 sink/recent windows (see OscarConfig). Resolved per-layer at
        # first forward when the cache block size is known.
        self.window_enabled = (
            self.cfg.sink_tokens > 0 or self.cfg.recent_tokens > 0
        ) and self.cfg.staging_tokens > 0

    # ---- rotation setup (one-time per layer) ------------------------------
    def _ensure_rotations(self, layer: Any, device: torch.device):
        if getattr(layer, "_oscar_ready", False):
            return
        D = self.head_size
        rk = get_layer_rotation(
            self.cfg.k_rotation_path, layer.layer_name, D, device, torch.float32
        )
        rv = get_layer_rotation(
            self.cfg.v_rotation_path, layer.layer_name, D, device, torch.float32
        )
        layer._oscar_Rk = rk
        layer._oscar_RkT = rk.t().contiguous()
        layer._oscar_Rv = rv
        layer._oscar_RvT = rv.t().contiguous()
        layer._oscar_ready = True

    def _rotate_clip(
        self, x: torch.Tensor, R: torch.Tensor, clip_ratio: float
    ) -> torch.Tensor:
        """Rotate ``x`` (``[N, H, D]``) by ``R`` and optionally percentile-clip."""
        x_rot = torch.matmul(x.float(), R)
        if clip_ratio > 0.0:
            thr = torch.quantile(x_rot.abs(), clip_ratio, dim=-1, keepdim=True)
            x_rot = torch.clamp(x_rot, -thr, thr)
        return x_rot

    # ---- BF16 sink/recent staging ------------------------------------------
    # The reference OSCAR serving stack never attends the newest (and first,
    # "sink") tokens in INT2: they stay BF16 and are only demoted once they
    # age out of the recent window. Mirroring that here: sink/recent tokens
    # are dual-written into a per-layer BF16 staging arena keyed by cache
    # block id (owner-tagged per slot, so evictions/collisions degrade
    # gracefully to the INT2 copy), and attention reads them in BF16.

    def _ensure_staging(self, layer: Any, kv_cache: torch.Tensor):
        if getattr(layer, "_oscar_stage_ready", False):
            return
        bs = kv_cache.shape[1]
        cfg = self.cfg
        self.stage_block = bs
        self.sink_eff = (cfg.sink_tokens // bs) * bs
        self.sink_pages = self.sink_eff // bs
        # recent window can straddle one extra page
        self.tail_pages = (cfg.recent_tokens + bs - 1) // bs + 1
        rows = max(
            (cfg.staging_tokens + bs - 1) // bs,
            self.sink_pages + self.tail_pages + 2,
        )
        layer._oscar_stage_k = torch.zeros(
            rows,
            bs,
            self.num_kv_heads,
            self.head_size,
            dtype=torch.bfloat16,
            device=kv_cache.device,
        )
        layer._oscar_stage_v = torch.zeros_like(layer._oscar_stage_k)
        layer._oscar_slot_owner = torch.full(
            (rows, bs), -1, dtype=torch.int64, device=kv_cache.device
        )
        layer._oscar_stage_rows = rows
        layer._oscar_stage_ready = True

    def _staging_write(self, layer: Any, key, value, attn_metadata):
        """Write this step's sink/recent tokens (raw BF16 K/V) into staging.

        Fused into a single Triton kernel; see _oscar_staging_write_kernel for
        why.  The INT2 copy is still written by ``do_kv_cache_update``, so a
        token evicted from staging (row reuse or hash collision) falls back to
        its quantized copy exactly as before.
        """
        N = attn_metadata.num_actual_tokens
        if N == 0:
            return
        dev = key.device
        slot = attn_metadata.slot_mapping[:N].to(torch.int64)
        seq = attn_metadata.seq_lens.to(torch.int64)
        qsl = attn_metadata.query_start_loc.to(torch.int64)
        q_lens = qsl[1:] - qsl[:-1]
        req = torch.repeat_interleave(
            _oscar_arange(seq.shape[0], dev), q_lens
        )
        pos = seq[req] - qsl[req + 1] + _oscar_arange(N, dev)

        bs = self.stage_block
        sk, sv = layer._oscar_stage_k, layer._oscar_stage_v
        own = layer._oscar_slot_owner
        n_heads, head_dim = key.shape[1], key.shape[2]
        BLOCK_D = triton.next_power_of_2(head_dim)
        for _phase in (0, 1):
            _oscar_staging_write_kernel[(N, n_heads if _phase else 1)](
                key, value, sk, sv, own,
                slot, pos,
                key.stride(0), key.stride(1), sk.stride(0), sk.stride(1), sk.stride(2),
                own.stride(0), own.stride(1),
                seq, req,
                N, n_heads, head_dim, bs, layer._oscar_stage_rows,
                self.sink_eff, self.cfg.recent_tokens,
                BLOCK_D=BLOCK_D, PHASE=_phase,
            )

    def _stage_splice(self, layer: Any, bt_row, cached_len: int, k_cached, v_cached):
        """Replace INT2-reconstructed rows of a cached prefix with their exact
        BF16 staged values wherever the staging tag still matches."""
        bs = self.stage_block
        rows_total = layer._oscar_stage_rows
        dev = k_cached.device
        npg = (cached_len + bs - 1) // bs
        blk = bt_row[:npg].to(torch.int64)
        rows = blk % rows_total
        staged = layer._oscar_slot_owner[rows] == blk.unsqueeze(-1)  # [npg, bs]
        pos = (torch.arange(npg, device=dev) * bs).unsqueeze(-1) + torch.arange(
            bs, device=dev
        )
        staged = (staged & (pos < cached_len)).reshape(-1)[:cached_len]
        ks = layer._oscar_stage_k[rows].reshape(npg * bs, self.num_kv_heads, -1)
        vs = layer._oscar_stage_v[rows].reshape(npg * bs, self.num_kv_heads, -1)
        m = staged.view(-1, 1, 1)
        k_out = torch.where(m, ks[:cached_len].to(k_cached.dtype), k_cached)
        v_out = torch.where(m, vs[:cached_len].to(v_cached.dtype), v_cached)
        return k_out, v_out

    def do_kv_cache_update(
        self,
        layer: torch.nn.Module,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        N = slot_mapping.shape[0]
        if N <= 0:
            return
        self._ensure_rotations(layer, key.device)
        k = key[:N].view(N, self.num_kv_heads, self.head_size)
        v = value[:N].view(N, self.num_kv_heads, self.head_size)
        k_rot = self._rotate_clip(k, layer._oscar_Rk, self.cfg.k_clip_ratio)
        v_rot = self._rotate_clip(v, layer._oscar_Rv, self.cfg.v_clip_ratio)
        oscar_store(
            k_rot,
            v_rot,
            kv_cache,
            slot_mapping,
            key_levels=self.cfg.key_levels,
            value_levels=self.cfg.value_levels,
            key_packed_size=self.cfg.key_packed_size,
            data_bytes=self.cfg.key_data_bytes,
        )

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: "OscarMetadata",
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        num_tokens = query.shape[0]
        if output is None:
            output = torch.zeros(
                num_tokens,
                self.num_heads * self.head_size,
                dtype=query.dtype,
                device=query.device,
            )
        if attn_metadata is None:
            return output.fill_(0)
        N = attn_metadata.num_actual_tokens
        if N <= 0:
            return output.fill_(0)

        self._ensure_rotations(layer, query.device)
        q = query[:N].view(N, self.num_heads, self.head_size)

        if self.window_enabled and kv_cache.numel() > 0:
            self._ensure_staging(layer, kv_cache)
            self._staging_write(
                layer,
                key[:N].view(N, self.num_kv_heads, self.head_size),
                value[:N].view(N, self.num_kv_heads, self.head_size),
                attn_metadata,
            )

        num_decodes = attn_metadata.num_decodes
        num_decode_tokens = attn_metadata.num_decode_tokens

        if not attn_metadata.is_prefill:
            attn_out = self._decode_attention(q, kv_cache, attn_metadata, layer)
        elif num_decodes == 0:
            k = key[:N].view(N, self.num_kv_heads, self.head_size)
            v = value[:N].view(N, self.num_kv_heads, self.head_size)
            attn_out = self._prefill_attention(q, k, v, kv_cache, attn_metadata, layer)
        else:
            attn_out = torch.empty(
                N, self.num_heads, self.head_size, device=q.device, dtype=q.dtype
            )
            decode_meta = OscarMetadata(
                seq_lens=attn_metadata.seq_lens[:num_decodes],
                slot_mapping=attn_metadata.slot_mapping[:num_decode_tokens],
                block_table=attn_metadata.block_table[:num_decodes],
                query_start_loc=attn_metadata.query_start_loc[: num_decodes + 1],
                num_actual_tokens=num_decode_tokens,
                max_query_len=1,
                max_seq_len=attn_metadata.max_seq_len,
                is_prefill=False,
            )
            attn_out[:num_decode_tokens] = self._decode_attention(
                q[:num_decode_tokens], kv_cache, decode_meta, layer
            )
            prefill_seq_lens = attn_metadata.seq_lens[num_decodes:]
            if attn_metadata.seq_lens_cpu is not None:
                prefill_max_seq = int(attn_metadata.seq_lens_cpu[num_decodes:].max())
            else:
                prefill_max_seq = attn_metadata.max_seq_len
            prefill_qsl = (
                attn_metadata.query_start_loc[num_decodes:] - num_decode_tokens
            )
            prefill_qsl_cpu = None
            if attn_metadata.query_start_loc_cpu is not None:
                prefill_qsl_cpu = (
                    attn_metadata.query_start_loc_cpu[num_decodes:] - num_decode_tokens
                )
            prefill_meta = OscarMetadata(
                seq_lens=prefill_seq_lens,
                slot_mapping=attn_metadata.slot_mapping[num_decode_tokens:N],
                block_table=attn_metadata.block_table[num_decodes:],
                query_start_loc=prefill_qsl,
                num_actual_tokens=N - num_decode_tokens,
                max_query_len=attn_metadata.max_query_len,
                max_seq_len=prefill_max_seq,
                is_prefill=True,
                query_start_loc_cpu=prefill_qsl_cpu,
                seq_lens_cpu=attn_metadata.seq_lens_cpu[num_decodes:]
                if attn_metadata.seq_lens_cpu is not None
                else None,
            )
            k = key[:N].view(N, self.num_kv_heads, self.head_size)
            v = value[:N].view(N, self.num_kv_heads, self.head_size)
            attn_out[num_decode_tokens:] = self._prefill_attention(
                q[num_decode_tokens:],
                k[num_decode_tokens:],
                v[num_decode_tokens:],
                kv_cache,
                prefill_meta,
                layer,
            )

        if output.ndim == 3:
            output[:N] = attn_out.to(output.dtype)
        else:
            output[:N] = attn_out.reshape(N, -1).to(output.dtype)
        return output

    def _flash_attn_varlen(self, q, k, v, cu_q, cu_k, max_q, max_k):
        kwargs = dict(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu_q,
            cu_seqlens_k=cu_k,
            max_seqlen_q=max_q,
            max_seqlen_k=max_k,
            softmax_scale=self.scale,
            causal=True,
        )
        if self.fa_version is not None:
            kwargs["fa_version"] = self.fa_version
        return flash_attn_varlen_func(**kwargs)

    def _prefill_attention(self, query, key, value, kv_cache, attn_metadata, layer):
        N, Hq, D = query.shape
        Hk = key.shape[1]
        use_gqa = Hk < Hq

        # First-chunk prefill: all K/V are in the current batch — attend on the
        # raw (uncompressed) K/V exactly like the standard backend. Note that
        # max_query_len == max_seq_len alone does NOT imply this: a fresh
        # request can share the batch with a shorter continuation (prefix-cache
        # hit / chunked prefill), so also require that no request has prior
        # context.
        if (
            _HAS_FLASH_ATTN
            and attn_metadata.max_query_len == attn_metadata.max_seq_len
            and not attn_metadata.has_context
        ):
            return self._flash_attn_varlen(
                query,
                key,
                value,
                attn_metadata.query_start_loc,
                attn_metadata.query_start_loc,
                attn_metadata.max_query_len,
                attn_metadata.max_query_len,
            )

        # Continuation: dequant cached (rotated) INT2 K/V, inverse-rotate back to
        # the original space, concat with the current chunk, run flash attention.
        if attn_metadata.query_start_loc_cpu is not None:
            qsl = attn_metadata.query_start_loc_cpu.tolist()
        else:
            qsl = attn_metadata.query_start_loc.tolist()
        if attn_metadata.seq_lens_cpu is not None:
            seq_lens_list = attn_metadata.seq_lens_cpu.tolist()
        else:
            seq_lens_list = attn_metadata.seq_lens.tolist()

        output = torch.zeros(N, Hq, D, device=query.device, dtype=query.dtype)
        num_reqs = len(qsl) - 1
        for i in range(num_reqs):
            q_start, q_end = qsl[i], qsl[i + 1]
            q_len = q_end - q_start
            if q_len <= 0:
                continue
            seq_len = seq_lens_list[i]
            q_seq = query[q_start:q_end]
            k_seq = key[q_start:q_end]
            v_seq = value[q_start:q_end]
            cached_len = seq_len - q_len
            if cached_len <= 0:
                out = self._sdpa(q_seq, k_seq, v_seq, cached_len, seq_len, use_gqa)
            else:
                k_cached, v_cached = oscar_full_dequant_kv(
                    kv_cache,
                    attn_metadata.block_table[i : i + 1],
                    cached_len,
                    Hk,
                    D,
                    self.cfg.key_levels,
                    self.cfg.value_levels,
                    self.cfg.key_data_bytes,
                    self.cfg.key_packed_size,
                    self.cfg.value_data_bytes,
                )
                # inverse rotation back to original space
                k_cached = torch.matmul(k_cached.float(), layer._oscar_RkT).to(
                    query.dtype
                )
                v_cached = torch.matmul(v_cached.float(), layer._oscar_RvT).to(
                    query.dtype
                )
                if self.window_enabled and getattr(layer, "_oscar_stage_ready", False):
                    # exact BF16 values for any still-staged prefix positions
                    k_cached, v_cached = self._stage_splice(
                        layer,
                        attn_metadata.block_table[i],
                        cached_len,
                        k_cached,
                        v_cached,
                    )
                k_full = torch.cat([k_cached, k_seq], dim=0)
                v_full = torch.cat([v_cached, v_seq], dim=0)
                out = self._sdpa(q_seq, k_full, v_full, cached_len, seq_len, use_gqa)
            output[q_start:q_end] = out.to(query.dtype)
        return output

    def _sdpa(self, q_seq, k_full, v_full, cached_len, seq_len, use_gqa):
        q_len = q_seq.shape[0]
        q_t = q_seq.transpose(0, 1).unsqueeze(0)
        k_t = k_full.transpose(0, 1).unsqueeze(0)
        v_t = v_full.transpose(0, 1).unsqueeze(0)
        device = q_seq.device
        if cached_len <= 0:
            out = F.scaled_dot_product_attention(
                q_t, k_t, v_t, is_causal=True, scale=self.scale, enable_gqa=use_gqa
            )
        else:
            q_pos = torch.arange(q_len, device=device).unsqueeze(1) + cached_len
            k_pos = torch.arange(seq_len, device=device).unsqueeze(0)
            mask = k_pos <= q_pos
            out = F.scaled_dot_product_attention(
                q_t, k_t, v_t, attn_mask=mask, scale=self.scale, enable_gqa=use_gqa
            )
        return out[0].transpose(0, 1)

    def _decode_attention(self, query, kv_cache, attn_metadata, layer):
        if self.window_enabled and getattr(layer, "_oscar_stage_ready", False):
            return self._decode_attention_windowed(
                query, kv_cache, attn_metadata, layer
            )
        # Rotate the query into the same space as the rotated stored keys.
        q_rot = torch.matmul(query.float(), layer._oscar_Rk)
        out_rot = oscar_decode_attention(
            q_rot,
            kv_cache,
            attn_metadata.block_table,
            attn_metadata.seq_lens,
            self.scale,
            key_levels=self.cfg.key_levels,
            value_levels=self.cfg.value_levels,
            key_data_bytes=self.cfg.key_data_bytes,
            key_packed_size=self.cfg.key_packed_size,
            value_data_bytes=self.cfg.value_data_bytes,
            max_num_kv_splits=self.max_num_kv_splits,
        )
        # Map attention output (rotated-V space) back: out_true = out_rot @ R_v^T.
        out = torch.matmul(out_rot, layer._oscar_RvT)
        return out.to(query.dtype)

    def _decode_attention_windowed(self, query, kv_cache, attn_metadata, layer):
        """Decode attention with BF16 sink/recent windows.

        Splits every sequence into three ranges: ``[0, sink)`` and
        ``[cut, seq)`` are attended in BF16 from the staging arena, the
        middle ``[sink, cut)`` runs through the fused INT2 kernel (with the
        block table shifted past the sink pages so the kernel needs no
        changes), and the two partial results are LSE-merged. ``cut`` is the
        earliest position from which staging coverage is contiguous to the
        end of the sequence, so evicted staging entries transparently fall
        back to INT2.
        """
        B = query.shape[0]
        Hq = self.num_heads
        bs = self.stage_block
        R = layer._oscar_stage_rows
        dev = query.device
        owner = layer._oscar_slot_owner
        bt = attn_metadata.block_table
        seq = attn_metadata.seq_lens.to(torch.int64)
        maxpg = bt.shape[1]
        S_nb, TP, S_eff = self.sink_pages, self.tail_pages, self.sink_eff
        W = self.cfg.recent_tokens
        offs = _oscar_arange(bs, dev)

        # ---- window bounds (sink flag + BF16 tail cut) -------------------
        # Fused: see _oscar_window_bounds_kernel.  The eager form built
        # [B, TP, bs] and [B, S_nb, bs] booleans and ran flip/cumprod/flip to
        # reduce them to two [B] vectors; py-spy measured this enclosing
        # function at ~37% of the rollout worker.
        _cut = torch.empty(B, dtype=torch.int64, device=dev)
        _si = torch.empty(B, dtype=torch.int64, device=dev)
        _oscar_window_bounds_kernel[(B,)](
            owner, bt, seq, _cut, _si,
            owner.stride(0), owner.stride(1), bt.stride(0),
            B, maxpg, bs, R, S_nb, S_eff, W,
            BLOCK_BS=triton.next_power_of_2(bs),
        )
        cut, si = _cut, _si
        sink_active = si > 0

        # The per-slot validity masks are still needed by the BF16 branch, but
        # they are now derived from the already-reduced bounds instead of
        # driving the reduction.
        if S_nb > 0:
            sblk = bt[:, :S_nb].to(torch.int64)
            spos = (_oscar_arange(S_nb, dev) * bs).view(1, S_nb, 1) + offs.view(1, 1, bs)
            s_valid = (sink_active.view(B, 1, 1) & (spos < S_eff)).expand(B, S_nb, bs)
        else:
            sblk = torch.zeros(B, 0, dtype=torch.int64, device=dev)
            s_valid = torch.zeros(B, 0, bs, dtype=torch.bool, device=dev)

        last_page = (seq - 1) // bs
        pg = (last_page - (TP - 1)).unsqueeze(1) + _oscar_arange(TP, dev).unsqueeze(0)
        pg_ok = pg >= 0
        tblk = torch.gather(bt.to(torch.int64), 1, pg.clamp(0, maxpg - 1))
        town = owner[(tblk % R).unsqueeze(-1), offs.view(1, 1, bs)]
        t_staged = (town == tblk.unsqueeze(-1)) & pg_ok.unsqueeze(-1)
        pos = (pg * bs).unsqueeze(-1) + offs.view(1, 1, bs)
        inrange = (pos >= torch.maximum(seq - W, si * bs).view(B, 1, 1)) & (
            pos < seq.view(B, 1, 1)
        )
        t_valid = t_staged & inrange & (pos >= cut.view(B, 1, 1))

        # ---- INT2 kernel over [sink, cut): shift the block table by the
        # sink pages so positions stay page-aligned and the kernel is unchanged
        seq_eff = (cut - si * bs).to(torch.int32)
        gidx = (_oscar_arange(maxpg, dev).unsqueeze(0) + si.unsqueeze(1)).clamp(
            max=maxpg - 1
        )
        bt_eff = torch.gather(bt, 1, gidx)
        q_rot = torch.matmul(query.float(), layer._oscar_Rk)
        lse1 = torch.full((B, Hq), float("-inf"), dtype=torch.float32, device=dev)
        out1 = oscar_decode_attention(
            q_rot,
            kv_cache,
            bt_eff,
            seq_eff,
            self.scale,
            key_levels=self.cfg.key_levels,
            value_levels=self.cfg.value_levels,
            key_data_bytes=self.cfg.key_data_bytes,
            key_packed_size=self.cfg.key_packed_size,
            value_data_bytes=self.cfg.value_data_bytes,
            max_num_kv_splits=self.max_num_kv_splits,
            lse_buf=lse1,
        )
        o1 = torch.matmul(out1, layer._oscar_RvT)  # true space, normalized
        empty1 = (seq_eff <= 0).view(B, 1)
        lse1 = torch.where(
            empty1 | ~torch.isfinite(lse1),
            torch.full_like(lse1, float("-inf")),
            lse1,
        )
        o1 = torch.nan_to_num(o1)

        # ---- BF16 attention over the staged sink + tail segments ----
        all_blk = torch.cat([sblk, tblk], dim=1).contiguous()  # [B, P]
        valid = torch.cat([s_valid, t_valid], dim=1).contiguous()  # [B, P, bs]
        o2, lse2 = self._bf16_tail(query, layer, all_blk, valid)

        # ---- merge the two normalized partial attentions via LSE ----
        new_lse = torch.logaddexp(lse1, lse2)
        w1 = torch.exp(lse1 - new_lse).unsqueeze(-1)
        w2 = torch.exp(lse2 - new_lse).unsqueeze(-1)
        out = o1 * w1 + torch.nan_to_num(o2) * w2
        return out.to(query.dtype)

    def _bf16_tail(self, query, layer, all_blk, valid):
        """Staged-BF16 attention over ``all_blk``; see ``_BF16_TAIL_MODE``.

        Returns ``(o2, lse2)`` as fp32 ``[B, Hq, D]`` / ``[B, Hq]``.  ``o2`` is
        already softmax-normalized and ``lse2`` is ``-inf`` for rows with no
        valid slot, which is what the caller's ``logaddexp`` merge expects.
        """
        if _BF16_TAIL_MODE == 0:
            return self._bf16_tail_eager(query, layer, all_blk, valid)
        return self._bf16_tail_fused(
            query, layer, all_blk, valid, tiled=(_BF16_TAIL_MODE == 2)
        )

    def _bf16_tail_eager(self, query, layer, all_blk, valid):
        """Reference path: gather the selected pages, upcast, score densely.

        The gather is the expensive part -- it materialises
        ``[B, P, bs, Hk, D]`` in fp32 twice (K and V) per layer per decode
        step -- and it is what the fused kernels exist to avoid.
        """
        B = query.shape[0]
        Hq, Hk, D = self.num_heads, self.num_kv_heads, self.head_size
        g = Hq // Hk
        R = layer._oscar_stage_rows
        bs = self.stage_block
        P = all_blk.shape[1]
        L = P * bs
        rowsP = all_blk % R
        kseg = layer._oscar_stage_k[rowsP].reshape(B, L, Hk, D).float()
        vseg = layer._oscar_stage_v[rowsP].reshape(B, L, Hk, D).float()
        vmask = valid.reshape(B, L)
        qh = query.float().view(B, Hk, g, D)
        sc = torch.einsum("bkgd,blkd->bkgl", qh, kseg) * self.scale
        sc = sc.masked_fill(~vmask.view(B, 1, 1, L), float("-inf"))
        m2 = sc.amax(dim=-1)
        m2s = torch.where(torch.isfinite(m2), m2, torch.zeros_like(m2))
        p2 = torch.exp(sc - m2s.unsqueeze(-1))
        p2 = torch.where(vmask.view(B, 1, 1, L), p2, torch.zeros_like(p2))
        s2 = p2.sum(-1)
        o2 = torch.einsum("bkgl,blkd->bkgd", p2, vseg) / s2.clamp_min(1e-38).unsqueeze(
            -1
        )
        lse2 = torch.where(
            s2 > 0,
            m2s + torch.log(s2.clamp_min(1e-38)),
            torch.full_like(s2, float("-inf")),
        )
        return o2.reshape(B, Hq, D), lse2.reshape(B, Hq)

    def _bf16_tail_fused(self, query, layer, all_blk, valid, tiled: bool):
        """Fused path: one program per (batch, kv-head, gqa slot).

        ``all_blk`` and ``valid`` must be contiguous -- both kernels index the
        page and slot axes with an implicit stride of 1.
        """
        B = query.shape[0]
        Hq, Hk, D = self.num_heads, self.num_kv_heads, self.head_size
        g = Hq // Hk
        R = layer._oscar_stage_rows
        bs = self.stage_block
        P = all_blk.shape[1]
        dev = query.device
        o2 = torch.zeros(B, Hq, D, dtype=torch.float32, device=dev)
        lse2 = torch.full((B, Hq), float("-inf"), dtype=torch.float32, device=dev)
        if B == 0 or P == 0:
            return o2, lse2
        q = query.contiguous()
        # int8, not bool: both kernels do a plain load and compare against 0,
        # and the equivalence tests were written against int8.  Handing Triton
        # a torch.bool pointer would make it an i1 load -- probably fine, but
        # "probably" is not what the validated path was measured on.  The cast
        # is [B, P, bs] bytes, tens of KB.
        vi = valid.to(torch.int8).contiguous()
        sk, sv = layer._oscar_stage_k, layer._oscar_stage_v
        args = (
            q, sk, sv, all_blk, vi, o2, lse2,
            q.stride(0), q.stride(1), sk.stride(0), sk.stride(1), sk.stride(2),
            all_blk.stride(0), vi.stride(0), vi.stride(1),
            o2.stride(0), o2.stride(1), lse2.stride(0),
            P, bs, R, Hk, g, D, self.scale,
        )
        if tiled:
            _oscar_bf16_tail_tiled_kernel[(B, Hk, g)](
                *args,
                BLOCK_D=triton.next_power_of_2(D),
                BLOCK_BS=triton.next_power_of_2(bs),
            )
        else:
            _oscar_bf16_tail_kernel[(B, Hk, g)](
                *args, BLOCK_D=triton.next_power_of_2(D)
            )
        return o2, lse2
