#!/usr/bin/env python3
"""LoRA GRPO on GSM8K train with TRL GRPOTrainer and colocated vLLM generation."""
from __future__ import annotations

import argparse, json, os, shutil, subprocess, sys
from pathlib import Path

import torch
from datasets import Dataset
import pandas as pd
from peft import LoraConfig
from transformers import AutoTokenizer, TrainerCallback
from trl import GRPOConfig, GRPOTrainer

from common import gold_answer, messages

LOG_DIR = Path(os.environ.get("LOG_DIR", "results"))


def completion_text(c) -> str:
    return c[-1]["content"] if isinstance(c, list) else c


CHECKER = Path(__file__).resolve().parent / "check_answer.py"


def check(text: str, answer: str) -> float:
    """Run the answer checker in a child process; anything other than a clean '1' scores 0."""
    try:
        result = subprocess.run([sys.executable, str(CHECKER)], input=json.dumps({"completion": text, "answer": answer}),
                                capture_output=True, text=True, timeout=10, cwd=CHECKER.parent)
    except subprocess.TimeoutExpired:
        return 0.0
    return 1.0 if result.returncode == 0 and result.stdout.strip() == "1" else 0.0


def correctness_reward(completions, answer, **kwargs):
    """1.0 if the final '#### <number>' line matches the gold answer, else 0.0."""
    return [check(completion_text(c), a) for c, a in zip(completions, answer)]


class Trainer(GRPOTrainer):
    """GRPOTrainer that also appends every fourth completion of each step, as logged by the trainer, to rollouts.jsonl."""

    def log(self, logs, start_time=None):
        super().log(logs, start_time)
        if not (self.accelerator.is_main_process and "loss" in logs and self._logs["prompt"]):  # skip the end-of-run summary
            return
        rewards = self._logs["rewards"]["correctness_reward"]
        with (LOG_DIR / "rollouts.jsonl").open("a") as f:
            for i, (p, c, r, a) in enumerate(zip(self._logs["prompt"], self._logs["completion"], rewards,
                                                 self._logs["advantages"])):
                if i % 4 == 0:
                    f.write(json.dumps({"step": self.state.global_step, "prompt": p, "completion": c,
                                        "reward": r, "advantage": a}) + "\n")


class MetricsJsonl(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        with (LOG_DIR / "metrics.jsonl").open("a") as f:
            f.write(json.dumps({"global_step": state.global_step, **(logs or {})}, default=float) + "\n")


def gsm8k_train(path: str) -> Dataset:
    df = pd.read_parquet(path)
    return Dataset.from_list([{"prompt": messages(q), "answer": gold_answer(s)}
                              for q, s in zip(df["question"], df["answer"])])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--data", default="data/gsm8k/main/train-00000-of-00001.parquet")
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--num-generations", type=int, default=8)
    p.add_argument("--seed", type=int, default=20261120)
    p.add_argument("--beta", type=float, default=0.0)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--lora-r", type=int, default=64)
    args = p.parse_args()
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    dataset = gsm8k_train(args.data)
    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    config = GRPOConfig(
        output_dir=args.output + "_trainer",
        max_steps=args.max_steps,
        per_device_train_batch_size=args.num_generations,
        gradient_accumulation_steps=4,
        num_generations=args.num_generations,
        max_prompt_length=512,
        max_completion_length=1024,
        temperature=args.temperature,
        top_p=0.95,
        learning_rate=args.learning_rate,
        lr_scheduler_type="constant_with_warmup",
        warmup_steps=10,
        weight_decay=0.0,
        beta=args.beta,
        loss_type="dr_grpo",
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        use_vllm=True,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=0.3,
        vllm_max_model_length=1536,
        reward_num_workers=32,  # the checker runs in a child process; score a step's 32 completions concurrently
        logging_steps=1,
        logging_first_step=True,
        save_strategy="no",
        report_to="none",
        model_init_kwargs={"dtype": torch.bfloat16, "attn_implementation": "flash_attention_2"},
        seed=args.seed,
        data_seed=args.seed,
    )
    peft = LoraConfig(task_type="CAUSAL_LM", r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.0,
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    trainer = Trainer(model=args.model, reward_funcs=correctness_reward, args=config, train_dataset=dataset,
                          processing_class=tokenizer, peft_config=peft, callbacks=[MetricsJsonl()])
    trainer.train()
    merged = trainer.accelerator.unwrap_model(trainer.model).merge_and_unload().to(dtype=torch.bfloat16)
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(out, safe_serialization=True)
    tokenizer.save_pretrained(out)
    for f in ("generation_config.json",):
        if (Path(args.model) / f).exists():
            shutil.copy2(Path(args.model) / f, out / f)
    (out / "grpo_recipe.json").write_text(json.dumps(vars(args), indent=2) + "\n")


if __name__ == "__main__":
    main()
