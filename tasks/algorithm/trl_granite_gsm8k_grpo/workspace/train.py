import argparse
import gzip
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import wandb
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback, set_seed
from trl import GRPOConfig, GRPOTrainer

from data_tools import extract_answer, score_answer


def read_rows(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def prompt(question):
    return [{"role": "user", "content": question + "\nSolve the problem step by step. End with '#### ' followed by the numerical answer."}]


def reconcile_evidence(output, checkpoint):
    step = json.loads((Path(checkpoint) / "trainer_state.json").read_text())["global_step"]
    for name in ["metrics.jsonl", "evaluation.jsonl"]:
        path = output / name
        rows = [row for row in read_rows(path) if row["step"] <= step]
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    for path in (output / "rollouts").glob("*.jsonl.gz"):
        if int(path.name.split(".")[0]) >= step:
            path.unlink()
    for path in (output / "evaluation").glob("*.jsonl.gz"):
        if int(path.name.split(".")[0]) > step:
            path.unlink()


class RunEvidence(TrainerCallback):
    def __init__(self, tokenizer, evaluation, config, output):
        self.tokenizer = tokenizer
        self.evaluation = evaluation
        self.config = config
        self.output = output
        self.started = time.monotonic()

    def on_log(self, args, state, control, logs=None, **kwargs):
        record = {"step": state.global_step, **logs}
        with (self.output / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
        wandb.log(record)

    def evaluate(self, model, step):
        python_rng, numpy_rng = random.getstate(), np.random.get_state()
        torch_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state_all()
        was_training = model.training
        model.eval()
        records = []
        batch_size = self.config["evaluation_batch_size"]
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for start in range(0, len(self.evaluation), batch_size):
                rows = self.evaluation[start:start + batch_size]
                texts = [self.tokenizer.apply_chat_template(prompt(row["question"]), tokenize=False, add_generation_prompt=True, thinking=False) for row in rows]
                inputs = self.tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
                generated = model.generate(**inputs, do_sample=False, max_new_tokens=self.config["max_completion_length"], use_cache=True, pad_token_id=self.tokenizer.pad_token_id)
                completions = generated[:, inputs.input_ids.shape[1]:]
                for row, tokens, text in zip(rows, completions.tolist(), self.tokenizer.batch_decode(completions, skip_special_tokens=True), strict=True):
                    records.append({"id": row["id"], "question": row["question"], "answer": row["answer"], "completion": text, "completion_ids": tokens, "reward": score_answer(text, row["answer"])})
        with gzip.open(self.output / "evaluation" / f"{step}.jsonl.gz", "wt") as stream:
            for row in records:
                stream.write(json.dumps(row) + "\n")
        metric = {
            "step": step,
            "eval/accuracy": sum(row["reward"] for row in records) / len(records),
            "eval/format_rate": sum(extract_answer(row["completion"]) is not None for row in records) / len(records),
            "eval/count": len(records),
        }
        with (self.output / "evaluation.jsonl").open("a") as stream:
            stream.write(json.dumps(metric) + "\n")
        print(json.dumps(metric), flush=True)
        wandb.log(metric)
        model.train(was_training)
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
        torch.set_rng_state(torch_rng)
        torch.cuda.set_rng_state_all(cuda_rng)

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        if state.global_step == 0:
            self.evaluate(model, 0)

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step % self.config["evaluation_interval"] == 0 or state.global_step == args.max_steps:
            self.evaluate(model, state.global_step)

    def on_train_end(self, args, state, control, **kwargs):
        (self.output / "completion.json").write_text(json.dumps({"completed_steps": state.global_step, "elapsed_seconds": time.monotonic() - self.started}, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--model", default="model")
    parser.add_argument("--data", default="data")
    parser.add_argument("--output", default="results/run")
    parser.add_argument("--checkpoints", default="checkpoints")
    parser.add_argument("--resume")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    output = Path(args.output)
    (output / "evaluation").mkdir(parents=True, exist_ok=True)
    (output / "rollouts").mkdir(exist_ok=True)
    if args.resume:
        reconcile_evidence(output, args.resume)
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    set_seed(config["seed"])
    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left", local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32, attn_implementation="sdpa", local_files_only=True)
    training = read_rows(Path(args.data) / "train.jsonl")
    evaluation = read_rows(Path(args.data) / "eval.jsonl")
    training = Dataset.from_list([{"prompt": prompt(row["question"]), "answer": row["answer"], "id": row["id"]} for row in training])
    wandb.init(project="rl-training-case", name="run", mode="offline", dir=str(output), config=config, settings=wandb.Settings(disable_code=True, disable_git=True, x_disable_stats=True, x_disable_meta=True, console="off"))

    def correctness_reward(prompts, completions, completion_ids, answer, id, trainer_state, **kwargs):
        texts = [completion[0]["content"] for completion in completions]
        scores = [score_answer(text, gold) for text, gold in zip(texts, answer, strict=True)]
        with gzip.open(output / "rollouts" / f"{trainer_state.global_step}.jsonl.gz", "at") as stream:
            for row_id, messages, text, tokens, gold, score in zip(id, prompts, texts, completion_ids, answer, scores, strict=True):
                stream.write(json.dumps({"id": row_id, "prompt": messages, "completion": text, "completion_ids": tokens, "answer": gold, "reward": score}) + "\n")
        return scores

    training_args = GRPOConfig(
        output_dir=args.checkpoints,
        max_steps=config["max_steps"],
        learning_rate=config["learning_rate"],
        lr_scheduler_type="constant",
        per_device_train_batch_size=config["per_device_train_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        num_generations=config["num_generations"],
        max_completion_length=config["max_completion_length"],
        num_iterations=config["num_iterations"],
        temperature=1.0,
        top_p=1.0,
        top_k=0,
        beta=config["beta"],
        loss_type="grpo",
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": True},
        use_vllm=False,
        seed=config["seed"],
        data_seed=config["seed"],
        shuffle_dataset=config["shuffle_dataset"],
        chat_template_kwargs={"thinking": False},
        logging_steps=1,
        save_steps=config["checkpoint_interval"],
        save_total_limit=config["checkpoint_limit"],
        report_to="none",
        disable_tqdm=True,
    )
    trainer = GRPOTrainer(model=model, args=training_args, processing_class=tokenizer, train_dataset=training, reward_funcs=correctness_reward, callbacks=[RunEvidence(tokenizer, evaluation, config, output)])
    trainer.train(resume_from_checkpoint=args.resume)
    wandb.finish()


if __name__ == "__main__":
    main()
