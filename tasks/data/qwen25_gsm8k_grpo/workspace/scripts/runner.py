"""Native verl GRPO entry with shared configuration and separate workspaces."""
import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

from runtime import ROOT, MODEL, configure, atomic_json


def verify_resume(run):
    checkpoints = run / 'checkpoints'
    candidates = sorted(checkpoints.glob('global_step_*'),
                        key=lambda p: int(p.name.split('_')[-1]), reverse=True)
    for folder in candidates:
        manifest = folder / 'complete.json'
        if not manifest.is_file():
            continue
        data = json.loads(manifest.read_text())
        valid = True
        for item in data['files']:
            path = folder / item['path']
            if not path.is_file() or path.stat().st_size != item['bytes']:
                valid = False
                break
            digest = hashlib.sha256()
            with path.open('rb') as stream:
                for block in iter(lambda: stream.read(8 * 1024**2), b''):
                    digest.update(block)
            if digest.hexdigest() != item['sha256']:
                valid = False
                break
        if valid:
            (checkpoints / 'latest_checkpointed_iteration.txt').write_text(str(data['step']))
            return data['step']
    latest = checkpoints / 'latest_checkpointed_iteration.txt'
    if latest.exists():
        latest.unlink()  # This run's invalid/incomplete pointer only.
    return 0


def overrides(args, workspace, run):
    probe = args.mode == 'probe'
    return [
        'trainer.use_v1=false', 'algorithm.adv_estimator=grpo',
        'algorithm.use_kl_in_reward=false', 'algorithm.filter_groups.enable=false',
        'algorithm.rollout_correction.bypass_mode=false',
        'algorithm.rollout_correction.rollout_is=null', 'algorithm.rollout_correction.rollout_rs=null',
        f'data.train_files={workspace}/data/preprocessed/train.parquet',
        f'data.val_files={ROOT}/eval/' + ('probe.parquet' if probe else 'validation.parquet'),
        'data.train_batch_size=128', 'data.val_batch_size=64',
        'data.max_prompt_length=512', 'data.max_response_length=512',
        'data.filter_overlong_prompts=true', 'data.truncation=error',
        'data.seed=42', 'data.dataloader_num_workers=0', 'data.validation_shuffle=false',
        f'actor_rollout_ref.model.path={MODEL}',
        'actor_rollout_ref.model.use_remove_padding=true',
        'actor_rollout_ref.model.enable_gradient_checkpointing=true',
        'actor_rollout_ref.model.use_fused_kernels=false',
        'actor_rollout_ref.actor.strategy=fsdp', 'actor_rollout_ref.actor.fsdp_config.strategy=fsdp',
        'actor_rollout_ref.actor.fsdp_config.model_dtype=fp32',
        'actor_rollout_ref.actor.fsdp_config.dtype=bfloat16',
        'actor_rollout_ref.actor.fsdp_config.param_offload=false',
        'actor_rollout_ref.actor.fsdp_config.optimizer_offload=false',
        'actor_rollout_ref.actor.fsdp_config.use_torch_compile=false',
        'actor_rollout_ref.actor.fsdp_config.seed=42',
        'actor_rollout_ref.actor.ppo_mini_batch_size=32',
        'actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1',
        'actor_rollout_ref.actor.ppo_epochs=1', 'actor_rollout_ref.actor.use_dynamic_bsz=true',
        'actor_rollout_ref.actor.ppo_max_token_len_per_gpu=12288',
        'actor_rollout_ref.actor.optim.lr=1e-6', 'actor_rollout_ref.actor.optim.weight_decay=0.0',
        'actor_rollout_ref.actor.optim.lr_scheduler_type=constant',
        'actor_rollout_ref.actor.entropy_coeff=0.0', 'actor_rollout_ref.actor.use_kl_loss=true',
        'actor_rollout_ref.actor.kl_loss_coef=0.001', 'actor_rollout_ref.actor.kl_loss_type=low_var_kl',
        'actor_rollout_ref.actor.loss_agg_mode=token-mean', 'actor_rollout_ref.actor.data_loader_seed=42',
        'actor_rollout_ref.actor.checkpoint.save_contents=[model,optimizer,extra]',
        'actor_rollout_ref.rollout.name=vllm', 'actor_rollout_ref.rollout.mode=async',
        'actor_rollout_ref.rollout.dtype=bfloat16', 'actor_rollout_ref.rollout.n=8',
        'actor_rollout_ref.rollout.tensor_model_parallel_size=1',
        'actor_rollout_ref.rollout.temperature=1.0', 'actor_rollout_ref.rollout.top_p=1.0',
        'actor_rollout_ref.rollout.top_k=-1', 'actor_rollout_ref.rollout.seed=42',
        'actor_rollout_ref.rollout.calculate_log_probs=true',
        'actor_rollout_ref.rollout.logprobs_mode=raw_logprobs',
        'actor_rollout_ref.rollout.enforce_eager=true',
        'actor_rollout_ref.rollout.gpu_memory_utilization=0.4',
        'actor_rollout_ref.rollout.enable_prefix_caching=false',
        'actor_rollout_ref.rollout.enable_chunked_prefill=true',
        'actor_rollout_ref.rollout.max_model_len=1024',
        'actor_rollout_ref.rollout.max_num_batched_tokens=4096',
        'actor_rollout_ref.rollout.max_num_seqs=128',
        'actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=true',
        'actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=12288',
        'actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1',
        'actor_rollout_ref.rollout.agent.num_workers=4',
        'actor_rollout_ref.rollout.val_kwargs.temperature=' + ('1.0' if probe else '0.0'),
        'actor_rollout_ref.rollout.val_kwargs.do_sample=' + ('true' if probe else 'false'),
        'actor_rollout_ref.rollout.val_kwargs.n=' + ('8' if probe else '1'),
        'actor_rollout_ref.ref.log_prob_use_dynamic_bsz=true',
        'actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=12288',
        'actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1',
        'actor_rollout_ref.ref.fsdp_config.strategy=fsdp',
        'actor_rollout_ref.ref.fsdp_config.use_torch_compile=false',
        'reward.custom_reward_function.path=null', 'reward.reward_manager.name=naive', 'reward.num_workers=4',
        f'trainer.n_gpus_per_node={args.gpus}', 'trainer.nnodes=1',
        f'trainer.total_training_steps={args.steps}', 'trainer.total_epochs=4',
        'trainer.test_freq=' + ('3' if args.mode == 'smoke' else '20'),
        'trainer.save_freq=' + ('-1' if probe else ('3' if args.mode == 'smoke' else '10')),
        'trainer.max_actor_ckpt_to_keep=2', 'trainer.val_before_train=true',
        'trainer.val_only=' + ('true' if probe else 'false'),
        'trainer.logger=[console,file,wandb]', 'trainer.project_name=qwen25_gsm8k_grpo',
        f'trainer.experiment_name={args.run}', f'trainer.default_local_dir={run}/checkpoints',
        'trainer.resume_mode=auto' if not probe else 'trainer.resume_mode=disable',
        f'trainer.rollout_data_dir={run}/rollouts', f'trainer.validation_data_dir={run}/validation',
        f'ray_kwargs.ray_init.num_cpus={12 if args.gpus == 1 else 20}',
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', default='.')
    parser.add_argument('--run', required=True)
    parser.add_argument('--mode', choices=['probe', 'smoke', 'train'], required=True)
    parser.add_argument('--steps', type=int, default=40)
    parser.add_argument('--gpus', type=int, choices=[1, 4], default=4)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--input-config', action='store_true')
    args = parser.parse_args()
    assert 1 <= args.steps <= 60
    workspace = ROOT
    run = configure(args.run, workspace)
    os.environ['GSM_MODE'] = args.mode
    os.environ['GSM_ATTEMPT'] = time.strftime('%Y%m%dT%H%M%S') + '-' + str(os.getpid())
    sys.path[:0] = [str(ROOT / 'scripts'), str(workspace / 'verl')]
    lock = (run / 'writer.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (run / 'complete.json').is_file() and not args.dry_run:
        complete = json.loads((run / 'complete.json').read_text())
        if complete['steps'] >= (0 if args.mode == 'probe' else args.steps):
            print('RUN_ALREADY_COMPLETE', args.run, flush=True)
            return
    from hydra import initialize_config_dir, compose
    from omegaconf import OmegaConf
    with initialize_config_dir(config_dir=str(workspace / 'verl/verl/trainer/config'), version_base=None):
        config = compose(config_name='ppo_trainer', overrides=overrides(args, workspace, run))
    OmegaConf.resolve(config)
    if args.input_config:
        (run / 'resolved-config.yaml').write_text(OmegaConf.to_yaml(config))
        print('INPUT_CONFIG_READY', args.run, flush=True)
        return
    from verl.trainer.ppo.utils import need_critic, need_reference_policy
    from verl.utils.config import validate_config
    validate_config(config=config, use_reference_policy=need_reference_policy(config), use_critic=need_critic(config))
    assert not config.algorithm.filter_groups.enable
    assert config.reward.custom_reward_function.path is None
    assert config.data.max_response_length == 512 and config.actor_rollout_ref.rollout.n == 8
    (run / 'resolved-config.yaml').write_text(OmegaConf.to_yaml(config))
    atomic_json(run / 'launch.json', {'workspace': str(workspace), 'run': args.run, 'mode': args.mode,
                                     'steps': args.steps, 'gpus': args.gpus, 'argv': sys.argv, 'attempt': os.environ['GSM_ATTEMPT'],
                                     'packages': {n: importlib.metadata.version(n) for n in
                                                  ['torch', 'vllm', 'transformers', 'ray', 'verl']}})
    if args.dry_run:
        print('CONFIG_VALIDATED', args.run, flush=True)
        return
    restored_step = 0 if args.mode == 'probe' else verify_resume(run)
    atomic_json(run / 'resume.json', {'step': restored_step, 'attempt': os.environ['GSM_ATTEMPT']})
    import ray
    from verl.trainer.constants_ppo import get_ppo_ray_runtime_env
    from verl.trainer import main_ppo_v0
    from verl.trainer.main_ppo import main as train_main
    from observed_runner import ObservedTaskRunner
    main_ppo_v0.TaskRunner = ObservedTaskRunner
    runtime_env = get_ppo_ray_runtime_env(config)
    names = [key for key in os.environ if key.startswith(('HF_', 'XDG_', 'TORCH', 'TRITON', 'VLLM_',
                                                          'WANDB_', 'CUDA_CACHE', 'FLASHINFER', 'VERL_FILE_', 'GSM_'))]
    names += ['RUN_DIR', 'PYTHONPATH', 'TMPDIR', 'TOKENIZERS_PARALLELISM', 'OMP_NUM_THREADS',
              'PYTHONDONTWRITEBYTECODE', 'TRANSFORMERS_OFFLINE', 'RAY_USAGE_STATS_ENABLED']
    runtime_env['env_vars'].update({key: os.environ[key] for key in names if key in os.environ})
    runtime_env['py_executable'] = sys.executable
    runtime_env['worker_process_setup_hook'] = 'observer.worker_setup'
    backing = ROOT / 'r' / (args.run + '-' + str(os.getpid()))
    backing.mkdir(parents=True, exist_ok=True)
    temp = Path('/tmp') / ('gs_' + args.run[-12:] + '_' + str(os.getpid()))
    if not temp.exists():
        temp.symlink_to(backing, target_is_directory=True)
    started = time.time()
    success = False
    try:
        ray.init(address='local', num_cpus=12 if args.gpus == 1 else 20, num_gpus=args.gpus, include_dashboard=False,
                 object_store_memory=4 * 1024**3, _temp_dir=str(temp), runtime_env=runtime_env)
        train_main(config)
        success = True
    finally:
        ray.shutdown()
        atomic_json(run / ('execution-' + os.environ['GSM_ATTEMPT'] + '.json'),
                    {'success': success, 'elapsed_s': time.time() - started, 'mode': args.mode,
                     'steps': args.steps, 'restored_step': restored_step})
    if success:
        atomic_json(run / 'complete.json', {'steps': 0 if args.mode == 'probe' else args.steps,
                                           'mode': args.mode, 'time': time.time()})
        print('RUN_COMPLETE', args.run, flush=True)


if __name__ == '__main__':
    main()
