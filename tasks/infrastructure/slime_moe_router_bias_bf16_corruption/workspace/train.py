#!/usr/bin/env python3
"""Small GRPO run reproducing FP32 MoE correction-bias corruption on load."""

from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path

import torch
from torch import nn

from checkpoint import load_hf_weights


SEED = 20261127
STEPS = 12
GROUP_SIZE = 8
LEARNING_RATE = 0.01


class Router(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("expert_bias", torch.zeros(8, dtype=torch.float32))

    def _maintain_float32_expert_bias(self) -> None:
        self.expert_bias = self.expert_bias.float()


class TinyMoEPolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.router = Router()
        # Expert 0 initially chooses the wrong action; expert 7 chooses the
        # correct one. The small learning rate cannot erase the loading defect
        # over this short recorded run.
        initial = torch.tensor(
            [[2.0, -2.0]] + [[0.0, 0.0]] * 6 + [[-1.0, 2.0]],
            dtype=torch.float32,
        )
        self.expert_logits = nn.Parameter(initial)

    def selected_expert(self) -> int:
        return int(torch.argmax(self.router.expert_bias).item())

    def logits(self) -> torch.Tensor:
        return self.expert_logits[self.selected_expert()]


def tensor_sha256(tensor: torch.Tensor) -> str:
    raw = tensor.detach().cpu().contiguous().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def evaluate(model: TinyMoEPolicy) -> float:
    return float(torch.softmax(model.logits(), dim=-1)[1].item())


def main() -> None:
    random.seed(SEED)
    torch.manual_seed(SEED)
    out = Path(os.environ.get("TRAINING_RUN_DIR", "outputs"))
    out.mkdir(parents=True, exist_ok=True)
    metrics_path = out / "metrics.jsonl"

    model = TinyMoEPolicy()
    source_bias = torch.linspace(39.965, 40.035, 8, dtype=torch.float32)

    # Megatron's Float16Module converts buffers before Bridge loads HF weights.
    model.to(dtype=torch.bfloat16)
    load_hf_weights([model], source_bias)
    model.expert_logits.data = model.expert_logits.data.float()

    optimizer = torch.optim.Adam([model.expert_logits], lr=LEARNING_RATE)
    rows: list[dict[str, object]] = []

    for step in range(STEPS + 1):
        probs = torch.softmax(model.logits(), dim=-1)
        row: dict[str, object] = {
            "step": step,
            "eval/correct_action_probability": float(probs[1].item()),
            "router/selected_expert": model.selected_expert(),
            "router/bias_dtype": str(model.router.expert_bias.dtype),
            "router/bias_unique_values": int(torch.unique(model.router.expert_bias).numel()),
            "router/bias_sha256": tensor_sha256(model.router.expert_bias.float()),
            "source/bias_sha256": tensor_sha256(source_bias),
        }
        if step < STEPS:
            dist = torch.distributions.Categorical(logits=model.logits())
            actions = dist.sample((GROUP_SIZE,))
            rewards = (actions == 1).float()
            advantages = (rewards - rewards.mean()) / (rewards.std(unbiased=False) + 1e-6)
            loss = -(dist.log_prob(actions) * advantages).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = float(model.expert_logits.grad.norm().item())
            optimizer.step()
            row.update(
                {
                    "train/reward_mean": float(rewards.mean().item()),
                    "train/advantage_std": float(advantages.std(unbiased=False).item()),
                    "train/policy_loss": float(loss.item()),
                    "train/grad_norm": grad_norm,
                }
            )
        rows.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)

    metrics_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    summary = {
        "schema_version": 1,
        "seed": SEED,
        "steps": STEPS,
        "group_size": GROUP_SIZE,
        "learning_rate": LEARNING_RATE,
        "final_eval_correct_action_probability": evaluate(model),
        "selected_expert": model.selected_expert(),
        "bias_dtype": str(model.router.expert_bias.dtype),
        "bias_unique_values": int(torch.unique(model.router.expert_bias).numel()),
        "bias_sha256": tensor_sha256(model.router.expert_bias.float()),
        "source_bias_sha256": tensor_sha256(source_bias),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print("FINAL_SUMMARY " + json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
