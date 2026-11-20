"""Observe actual attention arguments during a same-token scoring forward."""
import hashlib
import sys
import torch
import torch.nn.functional as functional


def tensor_info(value, padding):
    if value is None:
        return {"type": "none", "shape": None}
    if not torch.is_tensor(value):
        return {"type": type(value).__name__, "shape": None}
    result = {"type": "tensor", "shape": list(value.shape), "dtype": str(value.dtype)}
    if padding and value.ndim >= 2 and value.shape[-1] >= padding:
        # The last valid query is causal to every earlier key.
        keys = value[..., -1, :padding] if value.ndim >= 3 else value[..., :padding]
        if value.dtype == torch.bool:
            allowed = keys
        elif value.dtype.is_floating_point:
            allowed = keys >= 0
        else:
            allowed = keys != 0
        result.update(padding_keys_observed=keys.numel(), padding_keys_allowed=int(allowed.sum().item()))
    return result


def capture(model, forward, padding):
    """Run one unchanged forward and restore the hooks even on exceptions."""
    observed = {"model_input": [], "attention_entry": [], "sdpa": []}
    prior_profile = sys.getprofile()
    original_sdpa = functional.scaled_dot_product_attention
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else []

    def model_input(_module, _args, kwargs):
        m = kwargs.get("attention_mask")
        p = kwargs.get("position_ids")
        item = tensor_info(m, padding)
        if torch.is_tensor(p):
            item["position_shape"] = list(p.shape)
            item["valid_position_sha256"] = hashlib.sha256(p[..., padding:].detach().cpu().contiguous().numpy().tobytes()).hexdigest()
        observed["model_input"].append(item)

    def profiler(frame, event, arg):
        if prior_profile:
            prior_profile(frame, event, arg)
        if event == "call" and frame.f_code.co_name == "_tq_attention" and frame.f_code.co_filename.endswith("turboquant_attn_hook.py"):
            observed["attention_entry"].append(tensor_info(frame.f_locals.get("attention_mask"), padding))

    def sdpa(*args, **kwargs):
        mask = kwargs.get("attn_mask", args[3] if len(args) > 3 else None)
        item = tensor_info(mask, padding)
        item["is_causal"] = bool(kwargs.get("is_causal", args[5] if len(args) > 5 else False))
        observed["sdpa"].append(item)
        return original_sdpa(*args, **kwargs)

    hook = model.register_forward_pre_hook(model_input, with_kwargs=True)
    try:
        functional.scaled_dot_product_attention = sdpa
        sys.setprofile(profiler)
        with torch.no_grad():
            result = forward()
    finally:
        sys.setprofile(prior_profile)
        functional.scaled_dot_product_attention = original_sdpa
        hook.remove()
    cuda_after = torch.cuda.get_rng_state_all() if cuda_rng else []
    if not torch.equal(cpu_rng, torch.get_rng_state()) or len(cuda_rng) != len(cuda_after) or any(not torch.equal(a, b) for a, b in zip(cuda_rng, cuda_after)):
        raise RuntimeError("Observation unexpectedly changed RNG state")
    observed["rng_unchanged"] = True
    return result, observed
