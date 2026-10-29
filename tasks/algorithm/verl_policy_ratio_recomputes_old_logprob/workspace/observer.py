"""Read-only observations of actual framework calls, tensors and generations."""
import json
import os
from pathlib import Path
import re
from decimal import Decimal, InvalidOperation


def emit(stream, record):
    root = os.environ.get('RUN_OUTPUT')
    if root:
        record = dict(record, stage=os.environ.get('RUN_STAGE', 'full'),
                      trainer_step=int(os.environ.get('EVIDENCE_STEP', '-1')))
        with (Path(root) / (stream + '.jsonl')).open('a') as output:
            output.write(json.dumps(record) + '\n')


def exact(text, answer):
    # Qwen math responses use a boxed final numerical answer.
    matches = re.findall(r"\\boxed\{([^{}]+)\}", text)
    if not matches:
        return 0.0
    try:
        predicted = Decimal(matches[-1].replace(',', '').replace('$', '').strip())
        expected = Decimal(str(answer).replace(',', ''))
        return float(predicted.is_finite() and expected.is_finite() and predicted == expected)
    except InvalidOperation:
        return 0.0


def validation_score(data_source, solution_str, ground_truth, extra_info=None):
    answer = str(extra_info['canonical_answer']) if extra_info else str(ground_truth)
    score = exact(solution_str, answer)
    emit('validation', {'response': solution_str, 'ground_truth': answer, 'score': score,
                        'index': int(extra_info['index']) if extra_info else None})
    return score


def install():
    import torch
    from verl.trainer.ppo import core_algos
    if getattr(core_algos, '_autorlbench_observed', False):
        return
    core_algos._autorlbench_observed = True
    original_loss = core_algos.compute_policy_loss

    def loss(*args, **kwargs):
        import inspect
        values = inspect.signature(original_loss).bind(*args, **kwargs).arguments
        old, current = values['old_log_prob'], values['log_prob']
        mask, advantages, clip = values['eos_mask'], values['advantages'], values['cliprange']
        result = original_loss(*args, **kwargs)
        with torch.no_grad():
            ratio = (current - old).exp()
            expected = torch.maximum(-advantages * ratio, -advantages * ratio.clamp(1-clip, 1+clip))
            expected_loss = (expected * mask).sum() / mask.sum()
            expected_kl = (-(current-old)*mask).sum() / mask.sum()
            emit('objective', {'loss': float(result[0]), 'behavior_objective': float(expected_loss),
                'reported_kl': float(result[2]), 'behavior_kl': float(expected_kl),
                'clip_fraction': float(result[1]),
                'logp_delta_max': float(((current-old).abs()*mask).max()),
                'advantage_abs_max': float((advantages.abs()*mask).max()),
                'valid_tokens': int(mask.sum())})
        return result
    core_algos.compute_policy_loss = loss

    from verl.utils.dataset.rl_dataset import RLHFDataset
    original_read = RLHFDataset._read_files_and_tokenize

    def read(self):
        original_read(self)
        for _, row in self.dataframe.iterrows():
            emit('dataset', {'index': int(row['extra_info']['index']),
                 'split': row['extra_info']['split'],
                 'canonical_answer': str(row['extra_info']['canonical_answer']),
                 'training_answer': str(row['reward_model']['ground_truth'])})
    RLHFDataset._read_files_and_tokenize = read

    from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager
    original_load = FSDPCheckpointManager.load_checkpoint

    def load(self, local_path, *args, **kwargs):
        result = original_load(self, local_path, *args, **kwargs)
        if local_path:
            emit('checkpoint', {'path': str(local_path),
                'scheduler_epoch': self.lr_scheduler.last_epoch,
                'optimizer_steps': sorted(set(float(v['step']) for v in self.optimizer.state.values() if 'step' in v))})
        return result
    FSDPCheckpointManager.load_checkpoint = load

    from verl.workers.sharding_manager.fsdp_vllm import FSDPVLLMShardingManager
    original_enter = FSDPVLLMShardingManager.__enter__

    def enter(self):
        result = original_enter(self)
        if os.environ.get('BENCHMARK_KIND') == 'infrastructure':
            model = self.inference_engine.llm_engine.model_executor.driver_worker.worker.model_runner.model
            params = self.module.state_dict()
            errors = []
            with torch.no_grad():
                for name, dest in model.named_parameters():
                    if name not in params:
                        continue
                    source = params[name]
                    if hasattr(source, 'full_tensor'):
                        source = source.full_tensor()
                    if source.shape != dest.shape:
                        continue
                    stride = max(1, dest.numel() // 1024)
                    expected = source.flatten()[::stride].to(dest.dtype)
                    actual = dest.flatten()[::stride]
                    errors.append(float((expected.float()-actual.float()).abs().max()))
            emit('weight_transfer', {'compared_parameters': len(errors),
                 'max_difference': max(errors) if errors else None})
        return result
    FSDPVLLMShardingManager.__enter__ = enter
