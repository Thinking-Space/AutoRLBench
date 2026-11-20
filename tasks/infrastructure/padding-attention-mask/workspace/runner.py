"""Teacher-free mathematical GRPO for the attention and NVFP4 lifecycle cases.

Imports GPU libraries only after CLI/config preflight. CPU probes use tiny random
Qwen2 models, not a surrogate attention implementation. No GPU run is implied by
the presence of this runner; execution status lives in the task evidence files.
"""
from __future__ import annotations
import argparse
import contextlib
import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import re
import sys
import subprocess
import time
for _candidate in [Path(__file__).parent / 'vendor', Path(__file__).parent / 'qat_cache_mask_vendor']:
    if _candidate.is_dir():
        sys.path.insert(0, str(_candidate))

def read_json(path):
    return json.loads(Path(path).read_text())

def versions():
    result = {'python': sys.version.split()[0]}
    for name in ['torch', 'transformers', 'vllm', 'nvidia-modelopt', 'wandb']:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    if 'torch' in sys.modules:
        result['torch_cuda'] = sys.modules['torch'].version.cuda
    return result

def prompt_text(problem):
    from qat_cache_mask_data import format_problem
    return format_problem(problem)

def encode_prompt(tokenizer, problem):
    from qat_cache_mask_data import encode_problem
    return encode_problem(tokenizer, problem)

def reward_math(text, answer, finish_reason):
    if finish_reason != 'stop':
        return -1.0
    matches = re.findall('\\\\boxed\\{\\s*([-+]?\\d+)\\s*\\}', text)
    return 1.0 if matches and int(matches[-1]) == int(answer) else -1.0

def math_data(seed=42, count=256):
    """Rebuildable exact-answer math dataset, with no model/teacher data source."""
    rng = random.Random(seed)
    rows = []
    for index in range(count):
        a, b, c = [rng.randrange(2, 31) for _ in range(3)]
        rows.append({'id': str(index), 'problem': f'Compute ({a} + {b}) * {c}.', 'answer': str((a + b) * c)})
    return rows

def install_attention(model, config):
    from qat_kv.turboquant_attn_hook import install_turboquant_fake_quant
    return install_turboquant_fake_quant(model, config.get('quantizer_preset', 'turboquant_k3v4_nc'), skip_layers=config.get('skip_layers'))

def position_ids(mask):
    return (mask.cumsum(-1) - 1).clamp_min(0)

class ValidVocabularyMask:
    """Use the same tokenizer-valid support in generation and training softmax."""

    def __init__(self, size):
        self.size = int(size)

    def __call__(self, input_ids, scores):
        scores[..., self.size:] = float('-inf')
        return scores

def response_log_probs(model, prompt, response, prompt_width, pad_id, device):
    import torch
    width = max(len(prompt), prompt_width)
    prefix = [pad_id] * (width - len(prompt)) + list(prompt)
    ids = torch.tensor([prefix + list(response)], dtype=torch.long, device=device)
    mask = torch.tensor([[0] * (width - len(prompt)) + [1] * (len(prompt) + len(response))], device=device)
    context = torch.autocast('cuda', dtype=torch.bfloat16) if str(device).startswith('cuda') else contextlib.nullcontext()
    with context:
        output = model(input_ids=ids, attention_mask=mask, position_ids=position_ids(mask), use_cache=False)
    valid_vocab = int(getattr(model, '_autorl_valid_vocab_size', output.logits.shape[-1]))
    if any((token < 0 or token >= valid_vocab for token in response)):
        raise ValueError('Response contains a tokenizer-invalid token')
    logits = output.logits[0, width - 1:width + len(response) - 1, :valid_vocab].float()
    targets = torch.tensor(response, dtype=torch.long, device=device)
    return logits.log_softmax(-1).gather(-1, targets[:, None]).squeeze(-1)

def group_advantages(rewards, n):
    import torch
    values = torch.tensor(rewards, dtype=torch.float32).reshape(-1, n)
    centered = values - values.mean(-1, keepdim=True)
    return (centered / (values.std(-1, keepdim=True, unbiased=False) + 1e-06)).flatten().tolist()

def ppo_loss(current, old, advantage, denominator, low=0.2, high=0.28):
    import torch
    ratio = (current - old).clamp(-20, 20).exp()
    return torch.maximum(-advantage * ratio, -advantage * ratio.clamp(1 - low, 1 + high)).sum() / denominator

class HFBackend:

    def __init__(self, actor, tokenizer, config):
        self.actor, self.tokenizer, self.config = (actor, tokenizer, config)

    def sync(self, actor):
        pass

    def release(self):
        pass

    def evaluate(self, examples, n=32):
        return evaluate_backend(self, examples, n)

    def sample(self, prompts):
        import torch
        rows = []
        model = self.actor
        model.eval()
        for ids in prompts:
            n = self.config['samples_per_prompt']
            tensor = torch.tensor([ids] * n, device=self.config['device'])
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                output = model.generate(input_ids=tensor, attention_mask=torch.ones_like(tensor), do_sample=True, temperature=self.config.get('temperature', 1.0), top_p=self.config.get('top_p', 1.0), top_k=0, max_new_tokens=self.config['max_new_tokens'], logits_processor=[ValidVocabularyMask(len(self.tokenizer))], pad_token_id=self.tokenizer.pad_token_id, eos_token_id=self.tokenizer.eos_token_id, return_dict_in_generate=True, output_scores=True, use_cache=True)
            for draw in range(n):
                response = output.sequences[draw, len(ids):].tolist()
                if self.tokenizer.eos_token_id in response:
                    response = response[:response.index(self.tokenizer.eos_token_id) + 1]
                logps = [float(logits[draw].float().log_softmax(-1)[token]) for logits, token in zip(output.scores[:len(response)], response, strict=True)]
                stopped = bool(response and response[-1] == self.tokenizer.eos_token_id)
                rows.append(dict(prompt_ids=ids, response_ids=response, rollout_log_probs=logps, finish_reason='stop' if stopped else 'length'))
        return rows

def inspect_native_tq(worker):
    config = worker.vllm_config
    context = config.compilation_config.static_forward_context
    return dict(cache_dtype=str(config.cache_config.cache_dtype), attention_classes=[type(layer.impl).__name__ for layer in context.values() if hasattr(layer, 'impl')], skip_layers=sorted((str(index) for index in config.cache_config.kv_cache_dtype_skip_layers)))

class NativeTQBackend:
    """One real native-KV vLLM sampler, with live BF16 actor-weight synchronization."""

    def __init__(self, actor, tokenizer, config):
        self.actor, self.tokenizer, self.config = (actor, tokenizer, config)
        os.environ.setdefault('VLLM_ENABLE_V1_MULTIPROCESSING', '0')
        inference = Path(__file__).parent / 'inference'
        if not inference.exists():
            inference = Path(__file__).parent / 'qat_probability_inference'
        runtime = Path(os.environ.get('OPQAD_VLLM_OVERLAY', str(Path.cwd() / 'runtime/vllm')))
        if not runtime.exists():
            subprocess.run([sys.executable, str(inference / 'install_overlay.py'), '--output', str(runtime)], check=True)
        sys.path.insert(0, str(runtime))
        os.environ['PYTHONPATH'] = str(runtime) + os.pathsep + os.environ.get('PYTHONPATH', '')
        from vllm import LLM
        state = getattr(actor, '_qat_turboquant_state')
        self.skip_layers = sorted((str(index) for index in state.skip_layers))
        self.llm = LLM(model=str(Path(config['model']).resolve()), dtype='bfloat16', kv_cache_dtype=config['quantizer_preset'], kv_cache_dtype_skip_layers=self.skip_layers, max_model_len=config['max_prompt_length'] + config['max_new_tokens'], max_num_seqs=config.get('max_num_seqs', 64), max_num_batched_tokens=32768, enable_chunked_prefill=True, enable_prefix_caching=False, enforce_eager=True, gpu_memory_utilization=config.get('gpu_memory_utilization', 0.4), logprobs_mode='processed_logprobs', seed=config['seed'])
        self.inspection = self.llm.collective_rpc(inspect_native_tq)
        for row in self.inspection:
            if row['cache_dtype'] != config['quantizer_preset'] or 'TurboQuantAttentionImpl' not in row['attention_classes']:
                raise RuntimeError('The requested real native TurboQuant backend is inactive')
            if row['skip_layers'] != self.skip_layers:
                raise RuntimeError('Actor and rollout quantize different layers')

    def sync(self, actor):
        source = {name: tensor.detach().to('cpu') for name, tensor in actor.state_dict().items()}

        def apply(model):
            consumed = model.load_weights(source.items())
            return {'source_tensors': len(source), 'loaded_tensors': len(consumed) if consumed is not None else -1}
        self.last_sync = self.llm.apply_model(apply)
        del source
        gc.collect()

    def release(self):
        self.llm.reset_prefix_cache()

    def evaluate(self, examples, n=32):
        return evaluate_backend(self, examples, n)

    def sample(self, prompts):
        from vllm import SamplingParams
        parameters = SamplingParams(n=self.config['samples_per_prompt'], temperature=self.config.get('temperature', 1.0), top_p=self.config.get('top_p', 1.0), top_k=-1, max_tokens=self.config['max_new_tokens'], logprobs=1, seed=self.config['seed'], allowed_token_ids=list(range(len(self.tokenizer))))
        outputs = self.llm.generate([dict(prompt_token_ids=ids) for ids in prompts], parameters, use_tqdm=False)
        rows = []
        for ids, output in zip(prompts, outputs, strict=True):
            for draw in output.outputs:
                tokens = list(draw.token_ids)
                rows.append(dict(prompt_ids=ids, response_ids=tokens, rollout_log_probs=[float(row[token].logprob) for row, token in zip(draw.logprobs, tokens, strict=True)], finish_reason=draw.finish_reason))
        return rows

def evaluate_backend(backend, examples, n=32):
    """Fixed-question avg@n entry, suitable for prepared AIME exact-integer rows."""
    if n != 32 or len(examples) != 30 or len({row['id'] for row in examples}) != 30:
        raise ValueError('Formal evaluation is fixed to all30 unique AIME24 questions with32 samples each')
    original = backend.config
    backend.config = dict(original, samples_per_prompt=n, temperature=0.7, top_p=0.9)
    try:
        backend.sync(backend.actor)
        prompts = [encode_prompt(backend.tokenizer, row['problem']) for row in examples]
        samples = backend.sample(prompts)
        if len(samples) != len(examples) * n:
            raise RuntimeError('Evaluation lost a question or sampled response')
        for index, sample in enumerate(samples):
            text = backend.tokenizer.decode(sample['response_ids'], skip_special_tokens=False)
            sample['correct'] = int(reward_math(text, examples[index // n]['answer'], sample['finish_reason']) > 0)
        return ({'eval/correct': sum((s['correct'] for s in samples)), 'eval/responses': len(samples), 'eval/avg32': sum((s['correct'] for s in samples)) / len(samples), 'eval/clip_ratio': sum((s['finish_reason'] == 'length' for s in samples)) / len(samples)}, samples)
    finally:
        backend.release()
        backend.config = original

def save_checkpoint(actor, optimizer, step, log_dir, config):
    import torch
    path = log_dir / 'checkpoints' / f'global_step_{step}'
    path.mkdir(parents=True, exist_ok=False)

    def host(value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu()
        if isinstance(value, dict):
            return {key: host(item) for key, item in value.items()}
        if isinstance(value, list):
            return [host(item) for item in value]
        if isinstance(value, tuple):
            return tuple((host(item) for item in value))
        return value
    if hasattr(actor, '_fsdp_wrapped_module'):
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, FullStateDictConfig, StateDictType
        context = FSDP.state_dict_type(actor, StateDictType.FULL_STATE_DICT, FullStateDictConfig(offload_to_cpu=True, rank0_only=True))
    else:
        context = contextlib.nullcontext()
    with context:
        torch.save(host(actor.state_dict()), path / 'model.pt')
    torch.save(host(optimizer.state_dict()), path / 'optimizer.pt')
    torch.save(dict(torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all(), python=random.getstate()), path / 'rng.pt')
    (path / 'state.json').write_text(json.dumps(dict(step=step, config=config, question_offset=step * config['batch_prompts'], complete=True), indent=2) + '\n')
    temporary = log_dir / 'checkpoints/latest.tmp'
    temporary.write_text(str(step) + '\n')
    temporary.replace(log_dir / 'checkpoints/latest.txt')

def restore_checkpoint(actor, optimizer, path, config):
    import torch
    path = Path(path)
    state = read_json(path / 'state.json')
    prior = dict(state['config'])
    current = dict(config)
    prior.pop('resume_from', None)
    current.pop('resume_from', None)
    if prior != current:
        raise ValueError('Checkpoint scientific configuration differs from this arm')
    if not state.get('complete'):
        raise ValueError('Incomplete training checkpoint')
    if hasattr(actor, '_fsdp_wrapped_module'):
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, FullStateDictConfig, StateDictType
        context = FSDP.state_dict_type(actor, StateDictType.FULL_STATE_DICT, FullStateDictConfig(offload_to_cpu=True, rank0_only=True))
    else:
        context = contextlib.nullcontext()
    with context:
        actor.load_state_dict(torch.load(path / 'model.pt', map_location='cpu', weights_only=True))
    optimizer.load_state_dict(torch.load(path / 'optimizer.pt', map_location='cpu', weights_only=True))
    rng = torch.load(path / 'rng.pt', map_location='cpu', weights_only=False)
    torch.set_rng_state(rng['torch'])
    torch.cuda.set_rng_state_all(rng['cuda'])
    random.setstate(rng['python'])
    return int(state['step'])

def train(config, log_dir):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if not torch.cuda.is_available():
        raise RuntimeError('GPU training requires a CUDA allocation; use --cpu-contract for the CPU proof')
    if config['case'] == 'nvfp4_cache' and torch.cuda.get_device_capability(config['device'])[0] < 10:
        raise RuntimeError('The pinned FlashInfer/CUTLASS NVFP4 path requires a supported Blackwell GPU')
    log_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(config['model'], local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    torch.manual_seed(config['seed'])
    random.seed(config['seed'])
    actor = AutoModelForCausalLM.from_pretrained(config['model'], local_files_only=True, dtype=torch.float32, attn_implementation='sdpa').to(config['device'])
    actor._autorl_valid_vocab_size = len(tokenizer)
    install_attention(actor, config)
    actor.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    actor.config.use_cache = False
    backend = NativeTQBackend(actor, tokenizer, config) if config.get('rollout_backend') == 'native_vllm' else HFBackend(actor, tokenizer, config)
    optimizer = torch.optim.AdamW(actor.parameters(), lr=config['learning_rate'], betas=(0.9, 0.999), weight_decay=0.1)
    first_step = restore_checkpoint(actor, optimizer, config['resume_from'], config) if config.get('resume_from') else 0
    from qat_cache_mask_data import load_examples
    if not config.get('data'):
        raise ValueError('Formal RL requires a prepared real mathematical dataset')
    rows = load_examples(config['data'])
    (log_dir / 'run_identity.json').write_text(json.dumps(dict(versions=versions(), actor_master_dtype=str(next(actor.parameters()).dtype), forward_dtype='bfloat16_autocast', tokenizer_valid_rows=len(tokenizer), model_output_rows=int(actor.config.vocab_size), train_data_sha256=hashlib.sha256(Path(config['data']).read_bytes()).hexdigest(), eval_data_sha256=hashlib.sha256(Path(config['eval_data']).read_bytes()).hexdigest() if config.get('eval_data') else None, objective='binary exact-integer mathematical task reward, group-relative advantages, PPO clip', teacher_calls=0), indent=2) + '\n')
    rollout_file = (log_dir / 'rollouts.jsonl').open('a')
    from experiment_logging import ExperimentLogger
    logger = ExperimentLogger(log_dir, config, name=config['case'], enable_wandb=config.get('wandb', True))
    try:
        if config.get('eval_data') and first_step == 0:
            initial, generated = backend.evaluate(load_examples(config['eval_data']), n=32)
            (log_dir / 'evaluation_0000.json').write_text(json.dumps(generated) + '\n')
            logger.log(0, initial)
        for step in range(first_step + 1, config['steps'] + 1):
            started = time.time()
            batch = [rows[((step - 1) * config['batch_prompts'] + i) % len(rows)] for i in range(config['batch_prompts'])]
            prompts = [encode_prompt(tokenizer, row['problem']) for row in batch]
            if any((len(ids) > config['max_prompt_length'] for ids in prompts)):
                raise ValueError('Prompt exceeds fixed budget')
            backend.sync(actor)
            samples = backend.sample(prompts)
            backend.release()
            if len(samples) != len(prompts) * config['samples_per_prompt'] or any((not s['response_ids'] for s in samples)):
                raise RuntimeError('GRPO requires every original prompt group and all nonempty sampled responses')
            rewards = []
            for index, sample in enumerate(samples):
                answer = batch[index // config['samples_per_prompt']]['answer']
                sample['text'] = tokenizer.decode(sample['response_ids'], skip_special_tokens=False)
                sample['reward'] = reward_math(sample['text'], answer, sample['finish_reason'])
                rewards.append(sample['reward'])
                rollout_file.write(json.dumps(dict(step=step, **sample)) + '\n')
            advantages = group_advantages(rewards, config['samples_per_prompt'])
            n = config['samples_per_prompt']
            groups = [rewards[i:i + n] for i in range(0, len(rewards), n)]
            (log_dir / f'rollout_identity_{step:04d}.json').write_text(json.dumps(dict(step=step, question_ids=[row['id'] for row in batch], response_token_sha256=[hashlib.sha256(json.dumps(s['response_ids']).encode()).hexdigest() for s in samples]), indent=2) + '\n')
            actor.eval()
            old = []
            error_sum = 0.0
            error_tokens = 0
            row_error_means = []
            raw_backend_error_sum = 0.0
            padding_effect_sum = 0.0
            raw_backend_rows = []
            padding_effect_rows = []
            with torch.no_grad():
                for sample in samples:
                    logp = response_log_probs(actor, sample['prompt_ids'], sample['response_ids'], config['max_prompt_length'], tokenizer.pad_token_id, config['device'])
                    old.append(logp.detach().cpu())
                    error = (logp.cpu() - torch.tensor(sample['rollout_log_probs'])).abs()
                    error_sum += float(error.sum())
                    error_tokens += error.numel()
                    row_error_means.append(float(error.mean()))
                    if config['case'] == 'padding_attention':
                        raw = response_log_probs(actor, sample['prompt_ids'], sample['response_ids'], len(sample['prompt_ids']), tokenizer.pad_token_id, config['device']).cpu()
                        numeric = (raw - torch.tensor(sample['rollout_log_probs'])).abs()
                        padding = (logp.cpu() - raw).abs()
                        raw_backend_error_sum += float(numeric.sum())
                        padding_effect_sum += float(padding.sum())
                        raw_backend_rows.append(float(numeric.mean()))
                        padding_effect_rows.append(float(padding.mean()))
            actor.train()
            mini = config.get('mini_batch_responses', config['samples_per_prompt'])
            loss_values = []
            grad_values = []
            for start in range(0, len(samples), mini):
                part = samples[start:start + mini]
                denominator = sum((len(s['response_ids']) for s in part))
                optimizer.zero_grad(set_to_none=True)
                for index, sample in enumerate(part, start):
                    current = response_log_probs(actor, sample['prompt_ids'], sample['response_ids'], config['max_prompt_length'], tokenizer.pad_token_id, config['device'])
                    loss = ppo_loss(current, old[index].to(current.device), advantages[index], denominator, config['clip_low'], config['clip_high'])
                    loss_values.append(float(loss.detach()))
                    loss.backward()
                if hasattr(actor, 'clip_grad_norm_'):
                    grad = actor.clip_grad_norm_(1.0)
                else:
                    grad = torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
                optimizer.step()
                grad_values.append(float(grad))
            metrics = dict(step=step, accuracy=sum((r > 0 for r in rewards)) / len(rewards), response_length_mean=sum((len(s['response_ids']) for s in samples)) / len(samples), clip_ratio=sum((s['finish_reason'] == 'length' for s in samples)) / len(samples), responses=len(samples), correct=sum((r > 0 for r in rewards)), teacher_calls=0, zero_variance_fraction=sum((len(set(g)) == 1 for g in groups)) / len(groups), rollout_train_logprob_mae=error_sum / max(1, error_tokens), rollout_train_logprob_mae_trajectory=sum(row_error_means) / len(row_error_means), actor_pg_loss=sum(loss_values) / max(1, len(grad_values)), grad_norm=max(grad_values), step_seconds=time.time() - started)
            if hasattr(backend, 'lifecycle'):
                (log_dir / 'lifecycle.json').write_text(json.dumps(backend.lifecycle, indent=2) + '\n')
                (log_dir / 'last_weight_sync.json').write_text(json.dumps(backend.last_sync, indent=2) + '\n')
            if hasattr(backend, 'inspection'):
                (log_dir / 'native_tq_inspection.json').write_text(json.dumps(backend.inspection, indent=2) + '\n')
                (log_dir / 'last_weight_sync.json').write_text(json.dumps(backend.last_sync, indent=2) + '\n')
            if raw_backend_rows:
                metrics.update(native_vs_hf_unpadded_logprob_mae=raw_backend_error_sum / max(1, error_tokens), hf_padding_logprob_effect_mae=padding_effect_sum / max(1, error_tokens), native_vs_hf_unpadded_logprob_mae_trajectory=sum(raw_backend_rows) / len(raw_backend_rows), hf_padding_logprob_effect_mae_trajectory=sum(padding_effect_rows) / len(padding_effect_rows))
            if config.get('eval_every', 0) and (step % config['eval_every'] == 0 or step == config['steps']):
                if not config.get('eval_data'):
                    raise ValueError('eval_data must name frozen exact-answer evaluation rows')
                evaluation, generated = backend.evaluate(load_examples(config['eval_data']), n=32)
                metrics.update(evaluation)
                (log_dir / f'evaluation_{step:04d}.json').write_text(json.dumps(generated) + '\n')
            rollout_file.flush()
            logger.log(step, metrics)
            if config.get('save_every', 0) and (step % config['save_every'] == 0 or step == config['steps']):
                save_checkpoint(actor, optimizer, step, log_dir, config)
        (log_dir / 'complete.json').write_text(json.dumps(dict(training='pure_math_grpo', steps=config['steps'], versions=versions()), indent=2) + '\n')
    finally:
        rollout_file.close()
        logger.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=Path('config.json'))
    parser.add_argument('--log-dir', type=Path, default=Path(os.environ.get('LOG_DIR', 'experiment_logs')))
    args = parser.parse_args()
    train(read_json(args.config), args.log_dir)
