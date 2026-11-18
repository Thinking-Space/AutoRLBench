"""Run synchronous JustRL math GRPO, or its matched critic/GAE PPO variant."""
from pathlib import Path
import sys

sys.path.insert(0, '/workspace')
if Path(__file__).with_name('launch.py').exists():
    from launch import main
else:
    from scripts.training.launch import main

if __name__ == '__main__':
    if '--ppo' in sys.argv:
        sys.argv.remove('--ppo')
        # Keep shared data, template, reward, sampling and actor clipping in config.yaml.
        # Explicit command-line overrides can still replace these variant defaults.
        sys.argv[1:1] = [
            'data.train_batch_size=64',
            'actor_rollout_ref.actor.ppo_mini_batch_size=16',
            # Use the available update-stage memory; algorithmic mini-batch stays 16.
            'actor_rollout_ref.actor.ppo_max_token_len_per_gpu=131072',
            'actor_rollout_ref.actor.fsdp_config.use_no_sync_for_gradient_accumulation=true',
            'actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=131072',
            'algorithm.adv_estimator=gae', 'algorithm.gamma=1.0', 'algorithm.lam=1.0',
            'critic.enable=true',
            'critic.model.path=/workspace/models/DeepSeek-R1-Distill-Qwen-1.5B',
            'critic.strategy=fsdp', 'critic.ppo_mini_batch_size=16',
            'critic.ppo_micro_batch_size_per_gpu=1',
            'critic.use_dynamic_bsz=true', 'critic.ppo_max_token_len_per_gpu=262144',
            'critic.fsdp.use_no_sync_for_gradient_accumulation=true',
            '+critic.ppo_infer_max_token_len_per_gpu=262144',
            'critic.optim.lr=1e-5', 'critic.optim.lr_warmup_steps=0',
            'critic.optim.lr_warmup_steps_ratio=0.0',
            'critic.optim.lr_scheduler_type=constant',
            'critic.ppo_epochs=1', 'critic.shuffle=false',
            'critic.cliprange_value=0.5', 'critic.loss_agg_mode=token-mean',
            'trainer.critic_warmup=0', 'trainer.max_critic_ckpt_to_keep=2',
        ]
    for size in ('7B', '32B'):
        flag = f'--skywork-{size.lower()}'
        if flag not in sys.argv:
            continue
        sys.argv.remove(flag)
        # Skywork-OR1 math data (filtered by this model's difficulty) and raw-question template on the
        # R1 distill of that size; the rest stays as in config.yaml.
        data = f'/workspace/experiments/justrl_math/inputs/skywork_{size.lower()}'
        sys.argv[1:1] = [
            f'actor_rollout_ref.model.path=/workspace/models/DeepSeek-R1-Distill-Qwen-{size}',
            f'data.train_files={data}/train.parquet',
            f'data.val_files=[{data}/AIME24.parquet,{data}/AIME25.parquet,{data}/AMC23.parquet]',
            'trainer.total_epochs=2',
        ] + ([
            # 7B actor + optimizer stay resident on each GPU, so vLLM gets less of the 275 GB.
            'actor_rollout_ref.rollout.gpu_memory_utilization=0.6',
        ] if size == '7B' else [
            # The frozen 32B reference only runs one forward pass per step; keep it on CPU between uses.
            'actor_rollout_ref.ref.fsdp_config.param_offload=true',
        ])
    main('justrl_math')
