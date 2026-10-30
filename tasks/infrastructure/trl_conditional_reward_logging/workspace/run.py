import argparse
import json
import os
import re
from pathlib import Path
from decimal import Decimal

import numpy as np
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from transformers.trainer_utils import get_last_checkpoint
from trl import GRPOConfig, GRPOTrainer


def number(text):
    match = re.search(r"(?:####\s*|\\boxed\{)([-+]?\d[\d,]*(?:\.\d+)?)", text)
    return format(Decimal(match.group(1).replace(",", "")).normalize(), "f") if match else None


class RecordMetrics(TrainerCallback):
    def __init__(self, directory):
        self.path = directory / f"metrics_rank{int(os.environ.get('RANK', 0))}.jsonl"

    def on_log(self, args, state, control, logs=None, **kwargs):
        with self.path.open("a") as stream:
            stream.write(json.dumps({"step": state.global_step, **logs}) + "\n")


class RecordedGRPO(GRPOTrainer):
    def __init__(self, *args, record_directory, record_tokens, **kwargs):
        self.record_directory = record_directory
        self.record_tokens = record_tokens
        self.record_index = 0
        self.capture_logps = False
        super().__init__(*args, **kwargs)

    def _get_per_token_logps_and_entropies(self, *args, **kwargs):
        result = super()._get_per_token_logps_and_entropies(*args, **kwargs)
        if self.capture_logps:
            self.recorded_logps = result[0].detach()
        return result

    def _compute_loss(self, model, inputs):
        self.capture_logps = self.record_tokens and model.training
        loss = super()._compute_loss(model, inputs)
        if self.capture_logps:
            directory = self.record_directory / "token_records"
            directory.mkdir(exist_ok=True)
            logps = self.recorded_logps
            previous = inputs.get("old_per_token_logps")
            previous = logps if previous is None else previous
            filename = f"step{self.state.global_step:04d}_batch{self.record_index:05d}_rank{self.accelerator.process_index}.npz"
            np.savez_compressed(
                directory / filename,
                logps=logps.float().cpu().numpy(),
                old_logps=previous.detach().float().cpu().numpy(),
                advantages=inputs["advantages"].detach().float().cpu().numpy(),
                mask=inputs["completion_mask"].detach().cpu().numpy(),
                generation_token_count=float(inputs["num_items_in_batch"]),
                loss=float(loss.detach()),
                accumulation_steps=self.current_gradient_accumulation_steps,
                generation_steps=self.args.steps_per_generation,
                world_size=self.accelerator.num_processes,
                clip_low=self.epsilon_low,
                clip_high=self.epsilon_high,
            )
            self.record_index += 1
        self.capture_logps = False
        return loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", choices=["normalization", "logging"], required=True)
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rank = int(os.environ.get("RANK", 0))
    rollout_path = args.output / f"rollouts_rank{rank}.jsonl"
    dataset = Dataset.from_parquet(str(args.data / "train.parquet")).shuffle(seed=args.seed).select(range(256))
    validation = Dataset.from_parquet(str(args.data / "test.parquet")).select(range(64))

    def format_row(row):
        return {"prompt": [{"role": "user", "content": row["question"] + "\nSolve the problem and write the final numerical answer after ####."}], "answer": number(row["answer"])}

    dataset = dataset.map(format_row, remove_columns=dataset.column_names)
    validation = validation.map(format_row, remove_columns=validation.column_names)

    def correctness(completions, answer, prompts, log_metric, trainer_state, **kwargs):
        texts = [completion[0]["content"] for completion in completions]
        parsed = [number(text) for text in texts]
        rewards = [float(predicted == expected) for predicted, expected in zip(parsed, answer)]
        with rollout_path.open("a") as stream:
            for prompt, text, expected, predicted, reward in zip(prompts, texts, answer, parsed, rewards):
                stream.write(json.dumps({"step": trainer_state.global_step, "prompt": prompt, "completion": text, "answer": expected, "parsed": predicted, "reward": reward}) + "\n")
        if args.case == "logging" and rank == 0:
            log_metric("answer_parse_rate", sum(value is not None for value in parsed) / len(parsed))
        return rewards

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=True, dtype=torch.float32, attn_implementation="sdpa")
    configuration = GRPOConfig(
        output_dir=str(args.output / "checkpoints"),
        run_name=f"gsm8k_{args.case}_seed{args.seed}",
        max_steps=args.steps,
        seed=args.seed,
        data_seed=args.seed,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=2,
        steps_per_generation=4,
        num_generations=4,
        max_completion_length=256,
        temperature=0.8,
        learning_rate=1e-6,
        lr_scheduler_type="constant",
        bf16=True,
        beta=0.0,
        loss_type="dapo",
        scale_rewards="group",
        use_vllm=False,
        use_liger_kernel=False,
        gradient_checkpointing=False,
        logging_steps=1,
        save_strategy="steps",
        save_steps=6,
        save_total_limit=1,
        eval_strategy="no",
        report_to=["wandb"],
        ddp_timeout=90,
        ddp_find_unused_parameters=False,
    )
    trainer = RecordedGRPO(
        model=model,
        processing_class=tokenizer,
        args=configuration,
        train_dataset=dataset,
        eval_dataset=validation,
        reward_funcs=correctness,
        record_directory=args.output,
        record_tokens=args.case == "normalization",
        callbacks=[RecordMetrics(args.output)],
    )
    checkpoint = get_last_checkpoint(str(args.output / "checkpoints")) if args.resume else None
    result = trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_state()
    metrics = trainer.evaluate()
    if trainer.accelerator.is_main_process:
        (args.output / "final_metrics.json").write_text(json.dumps({"step": trainer.state.global_step, "training": result.metrics, "evaluation": metrics}, indent=2) + "\n")


if __name__ == "__main__":
    main()
