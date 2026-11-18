"""Constrained autoregressive sampling and low-rank policy adaptation."""
import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoTokenizer, Qwen3_5ForCausalLM


class LowRankLinear(nn.Module):
    def __init__(self, linear, rank=16):
        super().__init__()
        self.base = linear
        device = linear.weight.device
        self.down = nn.Parameter(torch.empty(rank, linear.in_features, device=device))
        self.up = nn.Parameter(torch.zeros(linear.out_features, rank, device=device))
        nn.init.normal_(self.down, std=0.02)
        self.scale = 2.0

    def forward(self, states):
        update = F.linear(F.linear(states, self.down.to(states.dtype)), self.up.to(states.dtype))
        return self.base(states) + self.scale * update


class Policy(nn.Module):
    def __init__(self, model_dir, device):
        super().__init__()
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, padding_side="left")
        self.action_ids = []
        for letter in "ABC":
            tokens = self.tokenizer.encode(letter, add_special_tokens=False)
            assert len(tokens) == 1
            self.action_ids.append(tokens[0])
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        self.model = Qwen3_5ForCausalLM.from_pretrained(
            model_dir, local_files_only=True, dtype=dtype, attn_implementation="sdpa",
        ).to(device)
        self.model.requires_grad_(False)
        for layer in self.model.model.layers:
            if hasattr(layer, "self_attn"):
                for name in ("q_proj", "v_proj"):
                    attention = layer.self_attn
                    setattr(attention, name, LowRankLinear(getattr(attention, name)))
        self.model.eval()
        self.register_buffer("action_weights", self.model.lm_head.weight[self.action_ids].detach())

    def encode(self, prompt):
        return self.tokenizer.encode(prompt, add_special_tokens=False)

    def log_probs(self, prefixes, action_count, temperature):
        batch = self.tokenizer.pad(
            {"input_ids": prefixes}, padding=True, return_tensors="pt",
        ).to(self.device)
        positions = (batch.attention_mask.cumsum(dim=-1) - 1).clamp_min(0)
        states = self.model.model(**batch, position_ids=positions, use_cache=False).last_hidden_state
        logits = F.linear(states[:, -1], self.action_weights[:action_count]).float()
        return (logits / temperature).log_softmax(dim=-1)
