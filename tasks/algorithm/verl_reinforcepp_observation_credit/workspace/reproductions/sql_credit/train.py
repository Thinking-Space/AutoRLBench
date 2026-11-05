"""Launch the pinned veRL trainer, adding identical evidence capture to both arms."""
import argparse
import json
import os
from pathlib import Path


def install_evidence_capture(output):
    import functools
    import torch
    from verl.trainer.ppo import core_algos
    from verl.trainer.ppo import ray_trainer
    from verl.utils.tracking import Tracking

    original = core_algos.get_adv_estimator_fn("reinforce_plus_plus")
    call_index = 0
    active_batch = None
    original_compute = ray_trainer.compute_advantage

    @functools.wraps(original_compute)
    def compute(data, *args, **kwargs):
        nonlocal active_batch
        active_batch = data
        try:
            return original_compute(data, *args, **kwargs)
        finally:
            active_batch = None

    ray_trainer.compute_advantage = compute

    @functools.wraps(original)
    def traced(*args, **kwargs):
        nonlocal call_index
        advantages, returns = original(*args, **kwargs)
        rewards = kwargs.get("token_level_rewards", args[0] if args else None)
        mask = kwargs.get("response_mask", args[1] if len(args) > 1 else None)
        record = {"call_index": call_index, "shape": list(mask.shape), "rows": []}
        for row in range(len(mask)):
            item = {"row": row, "rewards": rewards[row].detach().cpu().tolist(),
                                   "mask": mask[row].detach().cpu().tolist(),
                                   "returns": returns[row].detach().cpu().tolist(),
                                   "advantages": advantages[row].detach().cpu().tolist()}
            if active_batch is not None:
                item["response_ids"] = active_batch.batch["responses"][row].detach().cpu().tolist()
                item["uid"] = str(active_batch.non_tensor_batch["uid"][row])
            record["rows"].append(item)
        with (output / "credit_traces.jsonl").open("a") as f:
            f.write(json.dumps(record) + "\n")
        call_index += 1
        return advantages, returns

    core_algos.ADV_ESTIMATOR_REGISTRY["reinforce_plus_plus"] = traced
    original_log = Tracking.log

    def log(self, data, step, backend=None):
        def encode(value):
            if isinstance(value, torch.Tensor):
                return value.detach().cpu().tolist()
            if hasattr(value, "item"):
                return value.item()
            return str(value)
        with (output / "metrics.jsonl").open("a") as f:
            f.write(json.dumps({"step": step, **data}, default=encode) + "\n")
        return original_log(self, data, step, backend)

    Tracking.log = log


def main():
    import datasets
    import ray
    import verl
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    from transformers import set_seed
    from reproductions.sql_credit.workload import SYSTEM, make_case

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=True, parents=True)
    os.environ["PC_RUN_DIR"] = str(args.output)
    set_seed(args.seed)
    # Development evidence only: sealed test cases are never used in pilots or recipe tuning.
    for split, count in [("train", 512), ("dev", 12 if args.steps <= 5 else 96)]:
        records = []
        for i in range(count):
            case = make_case(split, i)
            records.append({"data_source": "operations_sql", "agent_name": "tool_agent",
                            "prompt": [{"role": "system", "content": SYSTEM},
                                       {"role": "user", "content": case.question}],
                            "reward_model": {"style": "rule", "ground_truth": case.case_id},
                            "extra_info": {"index": i, "split": split, "need_tools_kwargs": True,
                                           "tools_kwargs": {"warehouse": {"create_kwargs": {"case_id": case.case_id}}}}})
        datasets.Dataset.from_list(records).to_parquet(str(args.output / f"{split}.parquet"))
    tool_path = args.output / "tools.json"
    tool_path.write_text(json.dumps({"tools": [{"class_name": "reproductions.sql_credit.verl_adapter.WarehouseTool",
                                               "config": {"type": "native"}}]}))
    overrides = [
        "algorithm.adv_estimator=reinforce_plus_plus", "algorithm.gamma=1.0", "algorithm.use_kl_in_reward=False",
        f"data.train_files={args.output}/train.parquet", f"data.val_files={args.output}/dev.parquet",
        "data.train_batch_size=16", "data.max_prompt_length=2048", "data.max_response_length=4096",
        "data.dataloader_num_workers=0", f"data.seed={args.seed}", "data.truncation=error",
        "+data.apply_chat_template_kwargs.enable_thinking=False", f"actor_rollout_ref.model.path={args.model}",
        "actor_rollout_ref.model.use_remove_padding=False", "actor_rollout_ref.model.enable_gradient_checkpointing=True",
        "+actor_rollout_ref.model.override_config.attn_implementation=sdpa",
        "actor_rollout_ref.actor.ppo_mini_batch_size=16", "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1",
        "actor_rollout_ref.actor.optim.lr=1e-6", "actor_rollout_ref.actor.use_kl_loss=False",
        "actor_rollout_ref.actor.entropy_coeff=0.0", "actor_rollout_ref.actor.use_dynamic_bsz=False",
        "actor_rollout_ref.actor.fsdp_config.param_offload=False",
        "actor_rollout_ref.actor.fsdp_config.optimizer_offload=False",
        "actor_rollout_ref.rollout.name=vllm", "actor_rollout_ref.rollout.mode=async",
        "actor_rollout_ref.rollout.tensor_model_parallel_size=1", "actor_rollout_ref.rollout.n=4",
        "actor_rollout_ref.rollout.gpu_memory_utilization=0.35", "actor_rollout_ref.rollout.enforce_eager=True",
        "actor_rollout_ref.rollout.max_model_len=8192", "actor_rollout_ref.rollout.max_num_seqs=16",
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1",
        "actor_rollout_ref.rollout.agent.num_workers=2", "actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent",
        "actor_rollout_ref.rollout.multi_turn.enable=True", "actor_rollout_ref.rollout.multi_turn.max_assistant_turns=6",
        "actor_rollout_ref.rollout.multi_turn.max_parallel_calls=1",
        f"actor_rollout_ref.rollout.multi_turn.tool_config_path={tool_path}", f"actor_rollout_ref.rollout.seed={args.seed}",
        "actor_rollout_ref.rollout.val_kwargs.n=1", "actor_rollout_ref.rollout.val_kwargs.do_sample=False",
        "reward.num_workers=2", "reward.custom_reward_function.path=/workspace/reproductions/sql_credit/verl_adapter.py",
        "reward.custom_reward_function.name=compute_score", "trainer.use_v1=False",
        "trainer.n_gpus_per_node=2", "trainer.nnodes=1", "trainer.val_before_train=True",
        "trainer.logger=[console,wandb]", "trainer.project_name=pc_autorlbench_sql",
        f"trainer.experiment_name={args.output.name}", "trainer.save_freq=-1", "trainer.test_freq=1",
        f"trainer.total_training_steps={args.steps}", "trainer.total_epochs=10",
        f"trainer.default_local_dir={args.output}/checkpoints", f"trainer.rollout_data_dir={args.output}/rollouts",
        f"trainer.validation_data_dir={args.output}/validation", "trainer.log_val_generations=4",
    ]
    with initialize_config_dir(config_dir=str(Path(verl.__file__).parent / "trainer/config"), version_base=None):
        config = compose(config_name="ppo_trainer", overrides=overrides)
    OmegaConf.save(config, args.output / "effective_config.yaml")

    @ray.remote
    class EvidenceRunner:
        def run(self, cfg):
            from verl.trainer.main_ppo_v0 import TaskRunner
            set_seed(args.seed)
            install_evidence_capture(args.output)
            # Delegate to the upstream runner; the wrapper only records metrics and returns.
            runner = TaskRunner.__ray_metadata__.modified_class()
            return runner.run(cfg)

    from verl.trainer.main_ppo import run_ppo
    run_ppo(config, task_runner_class=EvidenceRunner)
    ray.shutdown()


if __name__ == "__main__":
    main()
