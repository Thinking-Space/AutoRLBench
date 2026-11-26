"""Differentiable reference of StateQuant's deployed Mamba2 buffer arithmetic.

State/buffer round trips delegate to the pinned original codec. The forward
uses the current unrounded write for readout, rounded past writes, and an INT4
state fold after every eight response tokens. Gradients use an identity STE.
This block reference is for correctness and short training validation; its
Python loop over folds must be profiled before long-context production use.
"""
from dataclasses import dataclass
import os
import os
import torch


@dataclass(frozen=True)
class StateQuantMamba2Config:
    mode: str = "pace4"
    window: int = 8
    buffer_bits: int = 8

    def __post_init__(self):
        if self.mode not in ("pace4", "int4v"):
            raise ValueError("StateQuant training supports pace4 or int4v")
        if self.window != 8 or self.buffer_bits != 8:
            raise ValueError("Match the verified deployment: window=8, buffer_bits=8")


def _format(config, key_dim, value_dim):
    from statequant.quantizer import Quantizer
    axis, group = ("k", 64) if config.mode == "pace4" else ("v", 32)
    width = key_dim if axis == "k" else value_dim
    while group > 2 and width % group:
        group //= 2
    return Quantizer(4, axis, group, scale_dtype="fp16")


def state_roundtrip_ste(state, config):
    from statequant.kernels.fold import pack_vk, unpack_vk
    shape = state.shape
    fmt = _format(config, shape[-1], shape[-2])
    with torch.no_grad():
        codes, scales = pack_vk(state.detach(), fmt)
        rounded = unpack_vk(codes, scales, fmt, shape[-1]).reshape(shape)
    return rounded.detach() + (state - state.detach())


def buffer_roundtrip_ste(x, bits=8):
    from statequant.kernels.fold import _round_buf
    with torch.no_grad():
        rounded = _round_buf(x.detach(), bits)
    return rounded.detach() + (x - x.detach())


def _heads(x, heads):
    # [batch, time, groups, dim] -> [batch, heads, time, dim]
    b, t, groups, d = x.shape
    if heads % groups:
        raise ValueError("Mamba2 heads must be divisible by groups")
    return x[:, :, :, None, :].expand(b, t, groups, heads//groups, d).reshape(
        b, t, heads, d).transpose(1, 2)


def _scan_block(state, x, dt, B, C, A, D, quantize_buffer):
    """One parallel block, with state [B,H,V,K]; all arithmetic is FP32."""
    heads = state.shape[1]
    key, query = _heads(B, heads), _heads(C, heads)
    u = (dt[..., None] * x).transpose(1, 2)
    g = (dt * A).transpose(1, 2)
    cumulative = g.cumsum(-1)
    n = x.shape[1]
    weight = (cumulative[..., :, None] - cumulative[..., None, :]).clamp_max(0).exp()
    lower = torch.ones(n, n, dtype=torch.bool, device=x.device).tril(-1)
    # Preserve the diagonal's unrounded current write, as upstream step().
    if quantize_buffer:
        kbuf = _heads(buffer_roundtrip_ste(B), heads)
        ubuf = buffer_roundtrip_ste((dt[..., None] * x)).transpose(1, 2)
    else:
        kbuf, ubuf = key, u
    product = torch.matmul(query, kbuf.transpose(-1, -2))
    history = torch.matmul(torch.where(lower, product * weight, 0.0), ubuf)
    inherited = cumulative.exp()[..., None] * torch.matmul(query, state.transpose(-1, -2))
    own = (query * key).sum(-1, keepdim=True) * u
    out = inherited + history + own
    if D is not None:
        out = out + D[None, :, None, None] * x.transpose(1, 2)
    decay = (cumulative[..., -1:] - cumulative).exp()
    if quantize_buffer:
        # Preserve upstream fold accumulation order before the INT4 threshold.
        state = cumulative[..., -1, None, None].exp() * state
        if _OrderedFold is not None and os.environ.get("QAT_STATEQUANT_RESPONSE_FOLD_KERNEL", "0") == "1":
            state = _OrderedFold.apply(state, decay, ubuf, kbuf)
        else:
            state = _ordered_fold_torch(state, decay, ubuf, kbuf)
    else:
        state = cumulative[..., -1, None, None].exp() * state + torch.matmul(
            (ubuf * decay[..., None]).transpose(-1, -2), kbuf)
    return out.transpose(1, 2), state


def statequant_mamba2_scan(x, dt, B, C, A, D=None, *, dt_bias=None,
                          dt_softplus=True, dt_limit=(0.0, float("inf")),
                          prompt_width, attention_mask=None, config=None,
                          prefill_scan=None):
    """Teacher-forced full-sequence scan matching prompt-end + decode folds.

    x [B,T,H,V], dt [B,T,H], B/C [B,T,G,K], A/D/dt_bias [H].
    prompt_width includes left padding, so response starts at a common column.
    Optional prefill_scan is the original mamba_ssm scan with initial/final-state
    support; the fallback computes the same recurrence in 128-token blocks.
    """
    config = config or StateQuantMamba2Config()
    batch, length, heads, value_dim = x.shape
    if not 0 <= prompt_width <= length:
        raise ValueError("prompt_width must lie inside the sequence")
    if B.shape != C.shape or B.shape[:2] != (batch, length):
        raise ValueError("Mamba2 B/C must match batch and time")
    with torch.autocast(device_type=x.device.type, enabled=False):
        x, dt, B, C, A = x.float(), dt.float(), B.float(), C.float(), A.float()
        D = D.float() if D is not None else None
        if dt_bias is not None:
            dt = dt + dt_bias.float()
        if dt_softplus:
            dt = torch.where(dt <= 20.0, torch.log(1.0 + torch.exp(dt)), dt)
        dt = dt.clamp(*dt_limit)
        if attention_mask is not None:
            if attention_mask.shape != (batch, length):
                raise ValueError("attention_mask does not match the full sequence")
            dt = dt * attention_mask.to(dt.dtype)[..., None]
        state = x.new_zeros(batch, heads, value_dim, B.shape[-1])
        outputs = []
        if prompt_width and prefill_scan is not None:
            prefix, state = prefill_scan(
                x[:, :prompt_width], dt[:, :prompt_width], A,
                B[:, :prompt_width], C[:, :prompt_width], D=D,
                chunk_size=256, initial_states=None, return_final_states=True,
                dt_softplus=False)
            outputs.append(prefix.float())
        else:
            for start in range(0, prompt_width, 128):
                end = min(start+128, prompt_width)
                y, state = _scan_block(state, x[:, start:end], dt[:, start:end],
                    B[:, start:end], C[:, start:end], A, D, False)
                outputs.append(y)
            # Fallback reference uses the upstream recurrence order for the
            # prompt-end state. Small parallel-scan roundoff can cross an INT4
            # rounding boundary even when prefill outputs agree closely.
            if _OrderedPromptState is None or os.environ.get("QAT_STATEQUANT_PROMPT_KERNEL", "0") != "1":
                state = _ordered_prompt_state_torch(
                    x[:, :prompt_width], dt[:, :prompt_width],
                    B[:, :prompt_width], A)
            else:
                state = _OrderedPromptState.apply(
                    x[:, :prompt_width], dt[:, :prompt_width],
                    B[:, :prompt_width], A)
        state = state_roundtrip_ste(state, config)
        for start in range(prompt_width, length, config.window):
            end = min(start+config.window, length)
            y, state = _scan_block(state, x[:, start:end], dt[:, start:end],
                B[:, start:end], C[:, start:end], A, D, True)
            outputs.append(y)
            if end-start == config.window:
                state = state_roundtrip_ste(state, config)
        return torch.cat(outputs, dim=1), state

def _ordered_prompt_state_torch(x, dt, B, A):
    """Reference ordered recurrence retained for backward recomputation/parity."""
    batch, length, heads, value_dim = x.shape
    groups, key_dim = B.shape[2], B.shape[3]
    state = x.new_zeros(batch, heads, value_dim, key_dim)
    for index in range(length):
        key = B[:, index].repeat_interleave(heads // groups, 1)
        write = dt[:, index, :, None] * x[:, index]
        factor = torch.exp(dt[:, index] * A)
        state = state * factor[..., None, None] + write[..., :, None] * key[..., None, :]
    return state


try:
    import triton
    import triton.language as tl

    @triton.jit
    def _ordered_prompt_state_kernel(X, DT, BK, A, OUT,
                                     T: tl.constexpr, H: tl.constexpr,
                                     V: tl.constexpr, G: tl.constexpr,
                                     K: tl.constexpr, TILE_V: tl.constexpr,
                                     TILE_K: tl.constexpr):
        pid = tl.program_id(0)
        pid_v = tl.program_id(1)
        pid_k = tl.program_id(2)
        b = pid // H
        h = pid % H
        group = h // (H // G)
        vs = pid_v * TILE_V + tl.arange(0, TILE_V)
        ks = pid_k * TILE_K + tl.arange(0, TILE_K)
        state = tl.full((TILE_V, TILE_K), 0, tl.float32)
        a = tl.load(A + h)
        for t in range(T):
            d = tl.load(DT + (b * T + t) * H + h)
            xv = tl.load(X + ((b * T + t) * H + h) * V + vs, vs < V, 0)
            kv = tl.load(BK + ((b * T + t) * G + group) * K + ks, ks < K, 0)
            factor = tl.extra.cuda.libdevice.exp(d * a)
            write = d * xv
            state = state * factor + write[:, None] * kv[None, :]
        ptr = ((b * H + h) * V + vs[:, None]) * K + ks[None, :]
        tl.store(OUT + ptr, state, (vs[:, None] < V) & (ks[None, :] < K))


    class _OrderedPromptState(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, dt, B, A):
            if not x.is_cuda:
                ctx.save_for_backward(x, dt, B, A)
                return _ordered_prompt_state_torch(x, dt, B, A)
            x, dt, B, A = x.contiguous(), dt.contiguous(), B.contiguous(), A.contiguous()
            batch, length, heads, value_dim = x.shape
            groups, key_dim = B.shape[2], B.shape[3]
            out = torch.empty((batch, heads, value_dim, key_dim),
                              device=x.device, dtype=torch.float32)
            tile_v, tile_k = 4, 32
            _ordered_prompt_state_kernel[(batch * heads,
                                          triton.cdiv(value_dim, tile_v),
                                          triton.cdiv(key_dim, tile_k))](
                x, dt, B, A, out, length, heads, value_dim, groups, key_dim,
                tile_v, tile_k, num_warps=4, enable_fp_fusion=False)
            ctx.save_for_backward(x, dt, B, A)
            return out

        @staticmethod
        def backward(ctx, grad_out):
            x, dt, B, A = ctx.saved_tensors
            with torch.enable_grad():
                xr = x.detach().requires_grad_(True)
                dr = dt.detach().requires_grad_(True)
                br = B.detach().requires_grad_(True)
                ar = A.detach().requires_grad_(True)
                reference = _ordered_prompt_state_torch(xr, dr, br, ar)
                grads = torch.autograd.grad(reference, (xr, dr, br, ar), grad_out,
                                            allow_unused=True)
            return grads

except ImportError:
    _OrderedPromptState = None


def _ordered_fold_torch(state, decay, ubuf, kbuf):
    """Reference ordered eight-write accumulation retained for backward/parity."""
    for index in range(decay.shape[-1]):
        state = state + (decay[..., index, None] * ubuf[..., index, :])[..., :, None] * kbuf[..., index, None, :]
    return state


try:
    import triton
    import triton.language as tl

    @triton.jit
    def _ordered_fold_kernel(S, DECAY, U, K, OUT,
                             T: tl.constexpr, H: tl.constexpr,
                             V: tl.constexpr, KD: tl.constexpr,
                             TV: tl.constexpr, TK: tl.constexpr):
        pid = tl.program_id(0)
        pv = tl.program_id(1)
        pk = tl.program_id(2)
        b = pid // H
        h = pid % H
        vs = pv * TV + tl.arange(0, TV)
        ks = pk * TK + tl.arange(0, TK)
        s_ptr = ((b * H + h) * V + vs[:, None]) * KD + ks[None, :]
        state = tl.load(S + s_ptr, (vs[:, None] < V) & (ks[None, :] < KD), 0)
        for t in range(T):
            decay = tl.load(DECAY + (b * H + h) * T + t)
            uv = decay * tl.load(U + ((b * H + h) * T + t) * V + vs, vs < V, 0)
            kv = tl.load(K + ((b * H + h) * T + t) * KD + ks, ks < KD, 0)
            term = uv[:, None] * kv[None, :]
            state = state + term
        tl.store(OUT + s_ptr, state, (vs[:, None] < V) & (ks[None, :] < KD))


    class _OrderedFold(torch.autograd.Function):
        @staticmethod
        def forward(ctx, state, decay, ubuf, kbuf):
            state, decay = state.contiguous(), decay.contiguous()
            ubuf, kbuf = ubuf.contiguous(), kbuf.contiguous()
            b, h, v, kd = state.shape
            _, _, t, _ = ubuf.shape
            out = torch.empty_like(state, dtype=torch.float32)
            _ordered_fold_kernel[(b * h, triton.cdiv(v, 4), triton.cdiv(kd, 32))](
                state, decay, ubuf, kbuf, out, t, h, v, kd, 4, 32,
                num_warps=4, enable_fp_fusion=False)
            ctx.save_for_backward(state, decay, ubuf, kbuf)
            return out

        @staticmethod
        def backward(ctx, grad_out):
            state, decay, ubuf, kbuf = ctx.saved_tensors
            with torch.enable_grad():
                args = [t.detach().requires_grad_(True) for t in (state, decay, ubuf, kbuf)]
                ref = _ordered_fold_torch(*args)
                grads = torch.autograd.grad(ref, args, grad_out)
            return grads

except ImportError:
    _OrderedFold = None
