"""Install the StateQuant STE scan on Nemotron-H's existing HF mixers.

Projection, causal convolution, normalization and output projection stay the
model's own modules. Only the recurrent scan changes. The frozen teacher is
left untouched. Each actor microbatch supplies its exact prompt boundary.
"""
from dataclasses import dataclass, field
import types
import torch
from opqad.quantization.statequant_mamba2 import (
    StateQuantMamba2Config, statequant_mamba2_scan,
)


@dataclass
class StateQuantTrainingState:
    config: StateQuantMamba2Config
    layers: list = field(default_factory=list)
    prompt_width: int | None = None
    attention_mask: torch.Tensor | None = None
    calls: int = 0
    root_hooks: list = field(default_factory=list)

    @property
    def enabled(self):
        return True

    def prepare_batch(self, prompt_width, attention_mask):
        self.prompt_width = int(prompt_width)
        self.attention_mask = attention_mask

    def clear_batch(self):
        self.prompt_width, self.attention_mask = None, None


def get_statequant_training_state(model):
    for module in model.modules():
        state = getattr(module, "_opqad_statequant_training", None)
        if state is not None:
            return state
    return None


def _forward(self, hidden_states, cache_params=None, cache_position=None, attention_mask=None, **kwargs):
    state = self._opqad_statequant_training
    if state.prompt_width is None:
        import inspect
        parameters = inspect.signature(self._opqad_statequant_original_forward).parameters
        original_kwargs = dict(kwargs)
        for name, value in (("cache_params", cache_params), ("cache_position", cache_position),
                            ("attention_mask", attention_mask)):
            if name in parameters:
                original_kwargs[name] = value
        return self._opqad_statequant_original_forward(hidden_states, **original_kwargs)
    if cache_params is not None:
        raise ValueError("StateQuant training uses use_cache=False; rollout uses the packed vLLM plugin")
    mask = state.attention_mask
    if mask is not None:
        hidden_states = hidden_states * mask[..., None].to(hidden_states.dtype)
    projected = self.in_proj(hidden_states)
    batch, length, _ = projected.shape
    gate, conv_input, dt = projected.split(
        [self.intermediate_size, self.conv_dim, self.num_heads], dim=-1)
    # The model's causal grouped convolution already has kernel-1 padding.
    conv = self.act(self.conv1d(conv_input.transpose(1, 2))[..., :length]).transpose(1, 2)
    if mask is not None:
        conv = conv * mask[..., None].to(conv.dtype)
    x, B, C = conv.split(
        [self.intermediate_size, self.n_groups*self.ssm_state_size,
         self.n_groups*self.ssm_state_size], dim=-1)
    # FP32 reference prefill is used until native input/gradient parity passes.
    out, _ = statequant_mamba2_scan(
        x.reshape(batch, length, self.num_heads, self.head_dim), dt,
        B.reshape(batch, length, self.n_groups, self.ssm_state_size),
        C.reshape(batch, length, self.n_groups, self.ssm_state_size),
        -self.A_log.float().exp(), self.D, dt_bias=self.dt_bias,
        dt_limit=(0.0, float("inf")), prompt_width=state.prompt_width,
        attention_mask=mask, config=state.config)
    state.calls += 1
    return self.out_proj(self.norm(out.reshape(batch, length, self.intermediate_size).to(gate.dtype), gate))


def install_statequant_training(model, config=None):
    existing = get_statequant_training_state(model)
    if existing is not None:
        return existing
    state = StateQuantTrainingState(config or StateQuantMamba2Config())
    for name, module in model.named_modules():
        if type(module).__name__ != "NemotronHMamba2Mixer":
            continue
        module._opqad_statequant_original_forward = module.forward
        module._opqad_statequant_training = state
        module.forward = types.MethodType(_forward, module)
        state.layers.append(name)
    if not state.layers:
        raise ValueError("No NemotronHMamba2Mixer found for StateQuant training")
    model._opqad_statequant_training = state
    # VeRL passes this boundary from its response tensor. The root pre-hook
    # removes the training-only argument before the HF model sees it. State
    # remains bound through backward so checkpoint recomputation uses the same
    # boundary; the next forward always refreshes it.
    def prepare(module, args, kwargs):
        width = kwargs.pop("statequant_prompt_width", None)
        if width is None:
            state.clear_batch()
        else:
            state.prepare_batch(width, kwargs.get("attention_mask"))
        return args, kwargs
    state.root_hooks.append(model.register_forward_pre_hook(prepare, with_kwargs=True))
    return state
