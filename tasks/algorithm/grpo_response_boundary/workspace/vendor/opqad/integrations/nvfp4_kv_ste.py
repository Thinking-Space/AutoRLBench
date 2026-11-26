"""Transformers attention adapter for native-layout NVFP4 KV STE.

The callable registered here receives post-RoPE, un-repeated K/V (the same
seam at which vLLM's FlashInfer backend calls its NVFP4 cache writer). This
keeps the existing projection-level integer fake-quant hook untouched.
"""
from __future__ import annotations
import itertools
from dataclasses import dataclass
import torch
from opqad.quantization.nvfp4_kv import NVFP4KVConfig, nvfp4_kv_fake_quant_pair
from opqad.quantization.static_fp8_ste import static_fp8_quant_dequant_ste

_SEQ = itertools.count()

@dataclass
class NVFP4AttentionState:
    model: torch.nn.Module
    original_impl: str
    registered_impl: str
    config: NVFP4KVConfig
    num_layers: int
    quantize_query: bool = False
    q_scale: float = 1.0
    quantize_output: bool = False
    enabled: bool = True
    calls: int = 0
    layers: set | None = None
    def __post_init__(self):
        if self.layers is None: self.layers = set()
    def assert_all_layers(self):
        expected=set(range(self.num_layers))
        missing=expected-self.layers
        if missing:
            raise RuntimeError(f"NVFP4 STE missed layers {sorted(missing)}")
    def remove(self):
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        cfg=getattr(self.model.config,'text_config',self.model.config)
        if getattr(cfg,'_attn_implementation',None)==self.registered_impl:
            cfg._attn_implementation=self.original_impl
        # The registry is process-global; the unique key avoids clobbering other wrappers.
        try: del ALL_ATTENTION_FUNCTIONS[self.registered_impl]
        except Exception: pass

def install_nvfp4_kv_ste(model: torch.nn.Module, config: NVFP4KVConfig = NVFP4KVConfig(),
                          *, base_impl: str | None = None, quantize_query: bool = False,
                          q_scale: float = 1.0,
                          quantize_output: bool = False) -> NVFP4AttentionState:
    """Install native-layout KV STE plus optional TRTLLM FP8 Q/output STE.

    NVFP4 TRTLLM kernels emit FP8 E4M3 attention output even when model
    activations are BF16. Round the output before the model output projection
    when matching that backend; the ordinary checkpoint uses output scale 1.
    """
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    cfg=getattr(model.config,'text_config',model.config)
    num_layers=int(cfg.num_hidden_layers)
    current=getattr(cfg,'_attn_implementation','sdpa')
    if base_impl is None:
        base_impl=current if not current.startswith('opqad_nvfp4_') else 'sdpa'
    if base_impl.startswith('opqad_nvfp4_'):
        raise ValueError('base_impl cannot be this NVFP4 wrapper')
    base_fn=ALL_ATTENTION_FUNCTIONS[base_impl]
    name=f'opqad_nvfp4_{next(_SEQ)}'
    state=NVFP4AttentionState(model,base_impl,name,config,num_layers,
                              quantize_query,q_scale,quantize_output)
    def wrapped(module, query, key, value, attention_mask, **kwargs):
        layer_idx=getattr(module,'layer_idx',None)
        if state.enabled:
            if state.quantize_query:
                query=static_fp8_quant_dequant_ste(query,scale=state.q_scale)
            key,value=nvfp4_kv_fake_quant_pair(key,value,state.config)
            state.calls+=1
            if layer_idx is not None: state.layers.add(int(layer_idx))
        attn_output, attn_weights = base_fn(module,query,key,value,attention_mask,**kwargs)
        if state.enabled and state.quantize_output:
            attn_output = static_fp8_quant_dequant_ste(attn_output, scale=1.0)
        return attn_output, attn_weights
    ALL_ATTENTION_FUNCTIONS.register(name,wrapped)
    # Keep mask semantics associated with the selected base implementation.
    try:
        from opqad.integrations.turboquant_attn_hook import _register_mask_function
        _register_mask_function(name,base_impl)
    except Exception:
        pass
    cfg._attn_implementation=name
    model._qat_nvfp4_kv_state=state
    return state

def get_installed_nvfp4_state(model: torch.nn.Module):
    """Find the installed KV STE through FSDP/compile wrappers."""
    seen=set()
    cur=model
    for _ in range(8):
        if cur is None or id(cur) in seen: break
        seen.add(id(cur))
        state=getattr(cur, "_qat_nvfp4_kv_state", None)
        if state is not None: return state
        cur=(getattr(cur, "_fsdp_wrapped_module", None) or
             getattr(cur, "module", None) or getattr(cur, "_orig_mod", None))
    return None
