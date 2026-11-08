"""Shared observation hooks. Do not alter reward, gradients, losses, or sampling math."""
import hashlib
import json
import os
from pathlib import Path
import time


def append(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + '\n')


def local_tensor(tensor):
    return tensor.to_local() if hasattr(tensor, 'to_local') else tensor


def sample_indices(size, device):
    import torch
    count = min(64, size)
    return torch.arange(count, device=device, dtype=torch.long) * (size - 1) // max(1, count - 1)


def worker_setup():
    """Ray setup hook: record actual AdamW updates on worker-local parameters."""
    import torch
    if getattr(torch.optim.AdamW.step, '_gsm_observed', False):
        return
    original = torch.optim.AdamW.step

    def observed_step(optimizer, *args, **kwargs):
        snapshots = []
        for group in optimizer.param_groups:
            for parameter in group['params']:
                if parameter.grad is None:
                    continue
                tensor = local_tensor(parameter.detach()).reshape(-1)
                if tensor.numel() == 0:
                    continue
                indices = sample_indices(tensor.numel(), tensor.device)
                snapshots.append((parameter, indices, tensor[indices].float().cpu()))
                if len(snapshots) >= 8:
                    break
            if len(snapshots) >= 8:
                break
        result = original(optimizer, *args, **kwargs)
        delta = 0.0
        finite = True
        steps = []
        for parameter, indices, before in snapshots:
            after = local_tensor(parameter.detach()).reshape(-1)[indices].float().cpu()
            delta = max(delta, (after - before).abs().max().item())
            finite = finite and bool(torch.isfinite(after).all())
            step = optimizer.state.get(parameter, {}).get('step')
            if step is not None:
                steps.append(int(step.item() if hasattr(step, 'item') else step))
        append(Path(os.environ['RUN_DIR']) / f'optimizer_{os.getpid()}.jsonl', {
            'time': time.time(), 'pid': os.getpid(), 'rank': os.environ.get('RANK'),
            'adam_step_min': min(steps) if steps else None,
            'adam_step_max': max(steps) if steps else None, 'sample_max_parameter_delta': delta,
            'finite_parameter_samples': finite, 'sampled_parameters': len(snapshots),
            'attempt': os.environ.get('GSM_ATTEMPT'),
        })
        return result
    observed_step._gsm_observed = True
    torch.optim.AdamW.step = observed_step


def install_trainer_hooks():
    from runtime import ROOT, atomic_json
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer
    if getattr(RayPPOTrainer, '_gsm_observed', False):
        return
    RayPPOTrainer._gsm_observed = True
    original_init = RayPPOTrainer.init_workers
    original_validate = RayPPOTrainer._validate
    original_save = RayPPOTrainer._save_checkpoint

    def init_workers(trainer, *args, **kwargs):
        result = original_init(trainer, *args, **kwargs)
        manager = trainer.async_rollout_manager
        original_generate = manager.generate_sequences

        def generate(prompts):
            output = original_generate(prompts)
            phase = getattr(trainer, '_gsm_phase', 'validation' if prompts.meta_info.get('validate') else 'train')
            step = int(prompts.meta_info.get('global_steps', trainer.global_steps))
            path = Path(os.environ['RUN_DIR']) / 'observations' / phase / f'{step}.jsonl'
            masks = output.batch.get('response_mask')
            responses = output.batch['responses']
            infos = prompts.non_tensor_batch.get('extra_info', [{}] * len(prompts))
            labels = prompts.non_tensor_batch.get('reward_model', [{}] * len(prompts))
            for i in range(len(output)):
                token_ids = responses[i].detach().cpu().tolist()
                length = int(masks[i].sum().item()) if masks is not None else None
                append(path, {'step': step, 'index': infos[i].get('original_index', infos[i].get('index')),
                              'ground_truth': labels[i].get('ground_truth'),
                              'response_length': length, 'response_ids': token_ids[:length] if length is not None else token_ids,
                              'at_response_cap': length == int(trainer.config.data.max_response_length),
                              'padded_batch_observation': True, 'attempt': os.environ.get('GSM_ATTEMPT')})
            return output
        manager.generate_sequences = generate
        return result

    def validate(trainer, *args, **kwargs):
        from torchdata.stateful_dataloader import StatefulDataLoader
        from verl.trainer.ppo.utils import create_rl_dataset
        from verl.utils.dataset.rl_dataset import collate_fn
        trainer._gsm_phase = 'validation'
        result = original_validate(trainer, *args, **kwargs)
        if os.environ.get('GSM_MODE') != 'train':
            trainer._gsm_phase = 'train'
            return result
        saved_loader = trainer.val_dataloader
        saved_dump = trainer.config.trainer.validation_data_dir
        settings = trainer.config.actor_rollout_ref.rollout.val_kwargs
        saved_settings = dict(settings)
        try:
            # Generation workers keep their initialization-time val_kwargs (temperature0).
            # Keep the validation declaration consistent with worker settings.
            panels = []
            if trainer.global_steps >= trainer.total_training_steps:
                panels.append(('heldout', ROOT / 'eval/test.parquet', False, 1))
            for phase, file, sample, count in panels:
                dataset = create_rl_dataset([str(file)], trainer.config.data, trainer.tokenizer,
                                            trainer.processor, is_train=False)
                trainer.val_dataloader = StatefulDataLoader(dataset, batch_size=64, shuffle=False,
                                                           num_workers=0, collate_fn=collate_fn, drop_last=False)
                trainer.config.trainer.validation_data_dir = str(Path(os.environ['RUN_DIR']) / phase)
                settings.do_sample, settings.temperature, settings.n = sample, (1.0 if sample else 0.0), count
                trainer._gsm_phase = phase
                metrics = original_validate(trainer, *args, **kwargs)
                result.update({phase + '/' + key: value for key, value in metrics.items()})
        finally:
            trainer.val_dataloader = saved_loader
            trainer.config.trainer.validation_data_dir = saved_dump
            for key, value in saved_settings.items():
                settings[key] = value
            trainer._gsm_phase = 'train'
        return result

    def save(trainer, *args, **kwargs):
        result = original_save(trainer, *args, **kwargs)
        folder = Path(trainer.config.trainer.default_local_dir) / f'global_step_{trainer.global_steps}'
        files = []
        for path in sorted(folder.rglob('*')):
            if not path.is_file() or path.name in ['complete.json', 'complete.json.partial']:
                continue
            digest = hashlib.sha256()
            with path.open('rb') as stream:
                for block in iter(lambda: stream.read(8 * 1024**2), b''):
                    digest.update(block)
            files.append({'path': str(path.relative_to(folder)), 'bytes': path.stat().st_size,
                          'sha256': digest.hexdigest()})
        world = int(trainer.config.trainer.n_gpus_per_node) * int(trainer.config.trainer.nnodes)
        assert len(files) >= 3 * world + 1, 'Missing distributed model/optimizer/RNG checkpoint files'
        assert any('optim' in item['path'] for item in files)
        atomic_json(folder / 'complete.json', {'step': int(trainer.global_steps), 'files': files,
                                              'time': time.time(), 'attempt': os.environ.get('GSM_ATTEMPT')})
        print('CHECKPOINT_COMPLETE', trainer.global_steps, flush=True)
        return result

    RayPPOTrainer.init_workers = init_workers
    RayPPOTrainer._validate = validate
    RayPPOTrainer._save_checkpoint = save
