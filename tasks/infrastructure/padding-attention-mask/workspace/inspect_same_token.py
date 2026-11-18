"""Inspect the task actor on two layouts of identical tokens, on CPU."""
import argparse
import importlib.util
import json
import pathlib
import sys
import torch
from transformers.models.qwen2 import Qwen2Config, Qwen2ForCausalLM
from attention_observer import capture

parser = argparse.ArgumentParser()
parser.add_argument('--workspace', required=True, type=pathlib.Path)
parser.add_argument('--output', required=True, type=pathlib.Path)
args = parser.parse_args()
sys.path.insert(0, str(args.workspace.resolve()))
spec = importlib.util.spec_from_file_location('task_runner', args.workspace / 'runner.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
torch.manual_seed(42)
model = Qwen2ForCausalLM(Qwen2Config(vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256, attn_implementation='sdpa')).eval()
runner.install_attention(model, {'quantizer_preset': 'turboquant_k3v4_nc', 'skip_layers': []})
prompt, response = [2, 3, 4], [5, 6, 1]
outputs = {}
scores = {}
for name, width in [('natural', len(prompt)), ('fixed', 84)]:
    values, observation = capture(model, lambda: runner.response_log_probs(model, prompt, response, width, 0, 'cpu'), width-len(prompt))
    if len(observation['model_input']) != 1 or len(observation['attention_entry']) != 2 or len(observation['sdpa']) != 2:
        raise RuntimeError('Expected attention observations were not collected')
    outputs[name], scores[name] = observation, values
result = {'prompt_ids': prompt, 'response_ids': response, 'quantizer': 'active', 'scores': {k: v.tolist() for k, v in scores.items()}, 'score_difference': (scores['natural']-scores['fixed']).abs().tolist(), 'observations': outputs}
args.output.write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps({'output': str(args.output), 'status': 'complete'}))
