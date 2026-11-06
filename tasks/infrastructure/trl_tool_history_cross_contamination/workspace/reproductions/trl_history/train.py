"""Real vLLM server + full-parameter GRPO, with identical tracing in both arms."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request


def append(path, record):
    with Path(path).open("a") as file:
        file.write(json.dumps(record, default=str) + "\n")


def instrument(trainer, output):
    import functools
    import inspect
    counter = 0
    active = {}
    original_turn = trainer._generate_single_turn

    @functools.wraps(original_turn)
    def turn(*args, **kwargs):
        caller = inspect.currentframe().f_back
        indices = caller.f_locals.get("idxs_with_tool")
        active["rows"] = list(indices) if indices is not None else list(range(len(args[0])))
        active["phase"] = "continuation" if indices is not None else "initial"
        return original_turn(*args, **kwargs)

    trainer._generate_single_turn = turn
    backend = trainer.vllm_generation
    original_generate = backend.generate
    original_client = backend.vllm_client.generate

    @functools.wraps(original_client)
    def client(*args, **kwargs):
        record = active["record"]
        record["client_prompt_ids"] = kwargs.get("prompts", args[0] if args else None)
        record["client_n"] = kwargs.get("n", 1)
        result = original_client(*args, **kwargs)
        # These are echoed by the real server for the corresponding generated choices.
        record["server_prompt_ids"] = result["prompt_ids"]
        record["completion_ids"] = result["completion_ids"]
        return result

    backend.vllm_client.generate = client

    @functools.wraps(original_generate)
    def generate(prompts, images, num_generations, profiler=None):
        nonlocal counter
        record = {"request_index": counter, "step": trainer.state.global_step,
                  "mode": "train" if trainer.model.training else "eval", "phase": active["phase"],
                  "rows": active["rows"], "num_generations": num_generations, "expected_prompt_ids": prompts,
                  "trajectory_ids": [trainer.environments[i].trajectory_id for i in active["rows"]]}
        active["record"] = record
        counter += 1
        started = time.monotonic()
        try:
            return original_generate(prompts, images, num_generations, profiler)
        finally:
            record["seconds"] = time.monotonic()-started
            append(output / "request_traces.jsonl", record)

    backend.generate = generate
    original_sync = backend.sync_weights

    @functools.wraps(original_sync)
    def sync():
        samples = {}
        for name, parameter in trainer.model.named_parameters():
            if len(samples) < 2 or name.endswith("lm_head.weight"):
                samples[name] = parameter.detach().flatten()[:16].float().cpu().tolist()
        started = time.monotonic()
        result = original_sync()
        append(output / "weight_sync.jsonl", {"step": trainer.state.global_step,
                                               "seconds": time.monotonic()-started, "samples": samples})
        return result

    backend.sync_weights = sync
    original_rollout = trainer._generate_and_score_completions

    @functools.wraps(original_rollout)
    def rollout(inputs):
        result = original_rollout(inputs)
        record = {"step": trainer.state.global_step, "mode": "train" if trainer.model.training else "eval",
                  "trajectory_ids": [env.trajectory_id for env in trainer.environments]}
        for key in ["prompt_ids", "completion_ids", "completion_mask", "tool_mask", "advantages"]:
            if key in result:
                record[key] = result[key].detach().cpu().tolist()
        append(output / "rollouts.jsonl", record)
        return result

    trainer._generate_and_score_completions = rollout


def worker(args):
    import functools
    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback, set_seed
    from trl import GRPOConfig, GRPOTrainer
    from reproductions.trl_history.environment import RepairEnvironment, SYSTEM

    set_seed(args.seed)
    manifest = json.loads((args.corpus / "manifest.json").read_text())
    def dataset(split):
        return Dataset.from_list([{"case_id": case_id, "prompt": [
            {"role": "system", "content": SYSTEM}, {"role": "user", "content": "Repair this upstream Python defect."}]}
            for case_id in manifest["splits"][split]])
    config = GRPOConfig(
        output_dir=str(args.output / "checkpoints"), run_name=args.output.name,
        learning_rate=1e-6, max_steps=args.steps, per_device_train_batch_size=1,
        gradient_accumulation_steps=32, generation_batch_size=32,
        per_device_eval_batch_size=4, num_generations=4, num_generations_eval=4,
        max_completion_length=3072, generation_kwargs={"max_tokens": 512, "seed": args.seed},
        max_tool_calling_iterations=6, temperature=0.8, top_p=0.95,
        use_vllm=True, vllm_mode="server", vllm_server_host="127.0.0.1", vllm_server_port=8000,
        bf16=True, gradient_checkpointing=True, beta=0.0, optim="adamw_torch",
        logging_steps=1, save_strategy="no", eval_strategy="steps", eval_steps=1,
        report_to=["wandb"], seed=args.seed, data_seed=args.seed,
        chat_template_kwargs={"enable_thinking": False}, remove_unused_columns=False,
    )
    (args.output / "effective_config.json").write_text(config.to_json_string())
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, attn_implementation="sdpa")
    model.config.max_position_embeddings = 4096
    model.config.use_cache = False
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    class EvidenceCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            append(Path(args.output_dir).parent / "metrics.jsonl", {"step": state.global_step, **(logs or {}),
                "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated()})

    trainer = GRPOTrainer(
        model=model, args=config, train_dataset=dataset("train"), eval_dataset=dataset("dev"),
        processing_class=tokenizer,
        environment_factory=functools.partial(RepairEnvironment, args.corpus, args.test_python, args.output),
        callbacks=[EvidenceCallback()],
    )
    instrument(trainer, args.output)
    trainer.evaluate()
    trainer.train()
    trainer.save_state()
    import wandb
    wandb.finish()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, default=Path("/artifacts/corpus"))
    parser.add_argument("--test-python", default="/opt/case-venv/bin/python")
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.worker:
        worker(args)
        return
    devices = os.environ.get("CUDA_VISIBLE_DEVICES", "0,1").split(",")
    if len(devices) != 2:
        raise ValueError("This recipe requires exactly two visible GPUs")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=devices[1], VLLM_SERVER_DEV_MODE="1")
    command = [sys.executable, "-m", "vllm.entrypoints.cli.main", "serve", args.model,
               "--host", "127.0.0.1", "--port", "8000", "--tensor-parallel-size", "1",
               "--max-model-len", "4096", "--max-num-seqs", "32", "--gpu-memory-utilization", "0.85",
               "--enforce-eager", "--seed", str(args.seed), "--dtype", "bfloat16",
               "--weight-transfer-config", '{"backend":"nccl"}', "--logprobs-mode", "processed_logprobs",
               "--max-logprobs", "-1"]
    (args.output / "server_command.json").write_text(json.dumps(command, indent=2))
    with (args.output / "server.log").open("w") as log:
        server = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic()+600
            while time.monotonic() < deadline:
                if server.poll() is not None:
                    raise RuntimeError(f"vLLM exited with {server.returncode}; see server.log")
                try:
                    with urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=2):
                        break
                except OSError:
                    time.sleep(2)
            else:
                raise TimeoutError("vLLM startup exceeded ten minutes")
            worker_env = dict(os.environ, CUDA_VISIBLE_DEVICES=devices[0], WANDB_PROJECT="pc_autorlbench_trl")
            subprocess.run([sys.executable, "-m", "reproductions.trl_history.train", *sys.argv[1:], "--worker"],
                           env=worker_env, check=True)
        finally:
            server.terminate()
            try:
                server.wait(timeout=20)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()


if __name__ == "__main__":
    main()
