"""Run the original GRPO loop with the shared-image native TQ backend."""
import argparse
import os
import pathlib

import runner
from qat_cache_mask_data import format_problem


def encode_prompt(tokenizer, problem):
    messages = [{'role': 'user', 'content': format_problem(problem)}]
    ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=False)
    valid_vocab = len(tokenizer)
    if not isinstance(ids, list) or not ids or any(type(x) is not int or x < 0 or x >= valid_vocab for x in ids):
        raise ValueError('Chat template must produce valid integer token IDs')
    return ids


class NativeTQBackend(runner.NativeTQBackend):
    def __init__(self, actor, tokenizer, config):
        from vllm import LLM
        self.actor, self.tokenizer, self.config = actor, tokenizer, config
        os.environ.setdefault('VLLM_ENABLE_V1_MULTIPROCESSING', '0')
        state = actor._qat_turboquant_state
        self.skip_layers = sorted(str(x) for x in state.skip_layers)
        self.llm = LLM(
            model=str(pathlib.Path(config['model']).resolve()), dtype='bfloat16',
            kv_cache_dtype=config['quantizer_preset'], kv_cache_dtype_skip_layers=self.skip_layers,
            max_model_len=config['max_prompt_length'] + config['max_new_tokens'],
            max_num_seqs=config.get('max_num_seqs', 64), max_num_batched_tokens=32768,
            enable_chunked_prefill=True, enable_prefix_caching=False, enforce_eager=True,
            gpu_memory_utilization=config.get('gpu_memory_utilization', 0.4),
            logprobs_mode='processed_logprobs', seed=config['seed'])
        self.inspection = self.llm.collective_rpc(runner.inspect_native_tq)
        for row in self.inspection:
            if (row['cache_dtype'] != config['quantizer_preset']
                    or 'TurboQuantAttentionImpl' not in row['attention_classes']
                    or row['skip_layers'] != self.skip_layers):
                raise RuntimeError('Native TurboQuant configuration differs from the requested configuration')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=pathlib.Path, default=pathlib.Path('config.shared.json'))
    parser.add_argument('--log-dir', type=pathlib.Path, default=pathlib.Path(os.environ.get('LOG_DIR', 'experiment_logs')))
    args = parser.parse_args()
    runner.encode_prompt = encode_prompt
    runner.NativeTQBackend = NativeTQBackend
    runner.train(runner.read_json(args.config), args.log_dir)


if __name__ == '__main__':
    main()
