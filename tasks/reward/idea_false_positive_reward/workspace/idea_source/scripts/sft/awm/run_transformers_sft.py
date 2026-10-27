#!/usr/bin/env python3
"""Transformers-native FSDP SFT runner for Qwen3.5 AWM skill data."""

import argparse
import json
from typing import Any

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments


class PromptResponseDataset(Dataset):
    def __init__(self, path: str):
        frame = pd.read_parquet(path)
        self.prompts = frame["prompt"].astype(str).tolist()
        self.responses = frame["response"].astype(str).tolist()

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, index: int) -> dict[str, str]:
        return {"prompt": self.prompts[index], "response": self.responses[index]}


class SFTCollator:
    def __init__(self, tokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, features: list[dict[str, str]]) -> dict[str, torch.Tensor]:
        full_texts = []
        prompt_texts = []
        for item in features:
            prompt_text = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": item["prompt"]}],
                add_generation_prompt=True,
                tokenize=False,
            )
            prompt_texts.append(prompt_text)
            full_texts.append(prompt_text + item["response"] + self.tokenizer.eos_token)
        batch = self.tokenizer(
            full_texts,
            padding="longest",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            add_special_tokens=False,
        )
        prompt_batch = self.tokenizer(
            prompt_texts,
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


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--train", required=True)
    p.add_argument("--val", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--max-length", type=int, default=12288)
    p.add_argument("--epochs", type=float, default=3)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--gradient-accumulation-steps", type=int, default=1)
    p.add_argument("--learning-rate", type=float, default=5e-6)
    p.add_argument("--max-steps", type=int, default=-1)
    p.add_argument("--no-fsdp", action="store_true", help="Disable FSDP for a single-GPU smoke test.")
    args = p.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        trust_remote_code=True,
    )
    model.config.use_cache = False
    train_dataset = PromptResponseDataset(args.train)
    val_dataset = PromptResponseDataset(args.val)
    estimated_steps = max(1, (len(train_dataset) + (args.batch_size * args.gradient_accumulation_steps) - 1) // (args.batch_size * args.gradient_accumulation_steps)) * max(1, int(args.epochs))
    warmup_steps = max(0, int(round(estimated_steps * 0.1)))
    if args.max_steps > 0:
        warmup_steps = max(0, int(round(args.max_steps * 0.1)))
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
        gradient_checkpointing=True,
        logging_strategy="steps",
        logging_steps=1,
        logging_first_step=True,
        eval_strategy="no" if args.max_steps > 0 else "epoch",
        save_strategy="no" if args.max_steps > 0 else "epoch",
        save_total_limit=1,
        report_to=[],
        remove_unused_columns=False,
        # DDP is used for Qwen3.5 because the hybrid linear/full-attention
        # stack currently deadlocks during this environment's FSDP setup.
        fsdp=None if args.no_fsdp else "full_shard auto_wrap",
        fsdp_config=None if args.no_fsdp else {"transformer_layer_cls_to_wrap": ["Qwen3DecoderLayer"], "use_orig_params": True},
        dataloader_num_workers=0 if args.max_steps > 0 else 8,
        dataloader_pin_memory=True,
        ddp_timeout=1800,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=SFTCollator(tokenizer, args.max_length),
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)


if __name__ == "__main__":
    main()
