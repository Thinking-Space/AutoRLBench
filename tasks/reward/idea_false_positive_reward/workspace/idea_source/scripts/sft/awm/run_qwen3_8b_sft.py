#!/usr/bin/env python3
"""Train Qwen3-8B on AWM candidate skills with Transformers FSDP.

This launcher intentionally owns both data alignment and training so it can be
run directly from a tmux window without depending on the older AWM runner.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments


def json_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def clip_text(text: str, limit: int = 18000) -> str:
    if len(text) <= limit:
        return text
    head = limit * 2 // 3
    return text[:head] + "\n[tool output truncated]\n" + text[-(limit - head) :]


def remove_think(text: str) -> str:
    text = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.IGNORECASE | re.DOTALL)
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    return text.strip()


def trajectory_context(rollout: dict[str, Any]) -> str:
    parts = [f"Task: {rollout.get('task_description') or rollout.get('task', '')}"]
    for message in rollout.get("messages") or []:
        if not isinstance(message, dict) or message.get("role") == "system":
            continue
        role = str(message.get("role", ""))
        content = clip_text(remove_think(json_text(message.get("content", ""))))
        if role == "assistant" and message.get("tool_calls"):
            content += "\ntool_calls=" + json.dumps(message["tool_calls"], ensure_ascii=False)
        parts.append(f"{role}: {content}")
    return "\n\n".join(parts)


def build_data(skills_path: Path, rollouts_path: Path, data_dir: Path, val_ratio: float, seed: int, turn_rollouts_path: Path | None = None) -> tuple[Path, Path]:
    data_dir.mkdir(parents=True, exist_ok=True)
    train_path = data_dir / "sft_episode_skill_train.parquet"
    val_path = data_dir / "sft_episode_skill_val.parquet"
    if train_path.exists() and val_path.exists() and turn_rollouts_path is None:
        return train_path, val_path

    skills: dict[str, dict[str, Any]] = {}
    with skills_path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                item = json.loads(line)
                if item.get("parse_ok", True) and item.get("task_id"):
                    skills[str(item["task_id"])] = item

    rows: list[dict[str, Any]] = []
    with rollouts_path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            rollout = json.loads(line)
            skill = skills.get(str(rollout.get("task_id", "")))
            if not skill:
                continue
            response = {
                "episode_summary": str(skill.get("episode_summary", "")),
                "episode_skill": str(skill.get("episode_skill", "")),
                "step_skills": skill.get("step_skills", []),
            }
            prompt = (
                "Analyze the task below and write a concise, reusable episode-level "
                "policy skill. Include critical step-specific skills when present. "
                "Return JSON with episode_summary, episode_skill, and step_skills.\n\n"
                + clip_text(trajectory_context(rollout), 30000)
            )
            rows.append({
                "prompt": prompt,
                "response": remove_think(json.dumps(response, ensure_ascii=False)),
                "task_id": str(rollout.get("task_id", "")),
                "task_type": rollout.get("scenario", "awm"),
            })

    if turn_rollouts_path:
        turn_count = 0
        with turn_rollouts_path.open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                item = json.loads(line)
                messages = item.get("messages") or []
                if not messages or messages[-1].get("role") != "assistant":
                    continue
                history = []
                for message in messages[:-1]:
                    history.append(f"{message.get('role', 'user')}: {remove_think(json_text(message.get('content', '')))}")
                response = remove_think(json_text(messages[-1].get("content", "")))
                if not response:
                    continue
                rows.append({
                    "prompt": "You are completing a multi-turn tool-use task. Continue the conversation and produce the next assistant action or final answer.\n\n" + "\n\n".join(history),
                    "response": response,
                    "task_id": f"turn_{turn_count}",
                    "task_type": "multi_turn_tool_use",
                })
                turn_count += 1

    if len(rows) < 2:
        raise RuntimeError(f"Need at least 2 aligned records, found {len(rows)}")
    random.Random(seed).shuffle(rows)
    val_size = max(1, min(len(rows) - 1, int(round(len(rows) * val_ratio))))
    pd.DataFrame(rows[val_size:]).to_parquet(train_path, index=False)
    pd.DataFrame(rows[:val_size]).to_parquet(val_path, index=False)
    pd.DataFrame(rows).to_parquet(data_dir / "sft_episode_skill_all.parquet", index=False)
    (data_dir / "build_summary.json").write_text(
        json.dumps({"total": len(rows), "train": len(rows) - val_size, "val": val_size}, indent=2),
        encoding="utf-8",
    )
    print(f"aligned={len(rows)} train={len(rows) - val_size} val={val_size}", flush=True)
    return train_path, val_path


class PromptResponseDataset(Dataset[dict[str, str]]):
    def __init__(self, path: Path):
        frame = pd.read_parquet(path)
        self.prompts = frame["prompt"].astype(str).tolist()
        self.responses = frame["response"].astype(str).tolist()

    def __len__(self) -> int:
        return len(self.prompts)

    def __getitem__(self, index: int) -> dict[str, str]:
        return {"prompt": self.prompts[index], "response": self.responses[index]}


class SFTCollator:
    def __init__(self, tokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, features: list[dict[str, str]]) -> dict[str, torch.Tensor]:
        prompts: list[str] = []
        full_texts: list[str] = []
        for item in features:
            prompt = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": item["prompt"]}],
                add_generation_prompt=True,
                tokenize=False,
            )
            prompts.append(prompt)
            full_texts.append(prompt + item["response"] + self.tokenizer.eos_token)
        batch = self.tokenizer(
            full_texts,
            padding="longest",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            add_special_tokens=False,
        )
        prompt_batch = self.tokenizer(
            prompts,
            padding=False,
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=False,
        )
        labels = batch["input_ids"].clone()
        labels[batch["attention_mask"] == 0] = -100
        for row, ids in enumerate(prompt_batch["input_ids"]):
            labels[row, : min(len(ids), self.max_length)] = -100
        batch["labels"] = labels
        return batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/local/user/Qwen/Qwen3-8B")
    parser.add_argument("--skills", required=True)
    parser.add_argument("--rollouts", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-length", type=int, default=12288)
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--turn-rollouts", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_path, val_path = build_data(
        Path(args.skills), Path(args.rollouts), Path(args.data_dir), args.val_ratio, args.seed,
        Path(args.turn_rollouts) if args.turn_rollouts else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
    )
    model.config.use_cache = False
    updates_per_epoch = max(
        1,
        (len(PromptResponseDataset(train_path)) + args.gradient_accumulation_steps - 1)
        // args.gradient_accumulation_steps,
    )
    warmup_steps = max(1, int(round(updates_per_epoch * max(1.0, args.epochs) * 0.1)))
    if args.max_steps > 0:
        warmup_steps = max(1, int(round(args.max_steps * 0.1)))

    layer_class = "Qwen3_5DecoderLayer" if "qwen3.5" in args.model.lower() or "qwen3_5" in args.model.lower() else "Qwen3DecoderLayer"
    training_args = TrainingArguments(
        output_dir=args.output,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=warmup_steps,
        optim="adamw_torch",
        adam_beta1=0.9,
        adam_beta2=0.95,
        weight_decay=0.01,
        max_grad_norm=1.0,
        bf16=True,
        gradient_checkpointing=False,
        logging_strategy="steps",
        logging_steps=1,
        logging_first_step=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        report_to=[],
        remove_unused_columns=False,
        fsdp="full_shard auto_wrap",
        fsdp_config={
            # Qwen3.5 uses the hybrid decoder layer; keeping this explicit
            # lets FSDP shard parameters and optimizer state per rank.
            "transformer_layer_cls_to_wrap": [layer_class],
            "use_orig_params": True,
            "limit_all_gathers": True,
            "activation_checkpointing": True,
        },
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        ddp_timeout=1800,
        seed=args.seed,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=PromptResponseDataset(train_path),
        eval_dataset=PromptResponseDataset(val_path),
        data_collator=SFTCollator(tokenizer, args.max_length),
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(args.output)
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(args.output)


if __name__ == "__main__":
    main()
