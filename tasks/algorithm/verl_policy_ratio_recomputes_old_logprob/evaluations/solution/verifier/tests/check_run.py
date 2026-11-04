"""Evaluate mechanism invariants from a fresh native-verl execution."""
import argparse
from collections import Counter
import json
import math
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path

try:
    from .profiles import experiment_profile
except ImportError:
    from profiles import experiment_profile


def rows(root, name):
    path = root / (name + '.jsonl')
    if not path.is_file():
        raise AssertionError('missing actual runtime evidence: ' + name)
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def score_answer(text, answer):
    matches = re.findall(r"\\boxed\{([^{}]+)\}", text)
    if not matches:
        return 0.0
    try:
        predicted = Decimal(matches[-1].replace(',', '').replace('$', '').strip())
        expected = Decimal(str(answer).replace(',', ''))
        return float(predicted.is_finite() and expected.is_finite() and predicted == expected)
    except InvalidOperation:
        return 0.0


def source_rows(data, split):
    records = [json.loads(line) for line in (data/(split+'.jsonl')).read_text().splitlines()]
    return {int(row['extra_info']['index']): row for row in records}


def final_evaluation(root, data, step):
    expected = source_rows(data, 'test')
    validation = rows(root, 'validation')
    selected = [row for row in validation if row.get('trainer_step') == step]
    assert len(selected) == len(expected), 'missing or duplicated final-step held-out responses'
    assert {int(row['index']) for row in selected} == set(expected), 'final evaluation does not cover the fixed held-out split'
    scores = [score_answer(row['response'], expected[int(row['index'])]['reward_model']['ground_truth']) for row in selected]
    return sum(scores)/len(scores), len(scores)


def evaluate(root, kind, steps=None, data_root=None):
    profile = experiment_profile(kind, steps)
    steps = profile['steps']
    assert data_root is not None, 'trusted GSM8K data is required'
    data_root = Path(data_root)
    test_size = len(source_rows(data_root, 'test'))
    metrics = rows(root, 'metrics')
    updates = [row for row in metrics if 'actor/optimizer_step_min' in row]
    assert [row['step'] for row in updates] == list(range(1, steps+1)), 'incomplete trainer budget'
    epochs = profile['optimizer_passes']
    assert all(row['actor/optimizer_step_min'] == row['actor/optimizer_step_max'] == epochs*row['step'] for row in updates), 'missing optimizer updates'
    assert all(math.isfinite(row['actor/grad_norm']) for row in updates), 'nonfinite gradients'
    assert max(row['actor/parameter_probe_change'] for row in updates) > 1e-8, 'actor parameters did not change'
    samples = rows(root, 'rewards')
    assert len(samples) == steps*profile['train_batch_size']*profile['group_size'], 'missing freely generated training completions'
    if kind == 'algorithm':
        assert test_size == profile['held_out_rows'], 'held-out set size changed'
        assert {r.get('trainer_step') for r in samples} == set(range(1, steps+1)), 'training response steps changed'
        for step in range(1, steps+1):
            groups = Counter(int(r['index']) for r in samples if r.get('trainer_step') == step)
            assert len(groups) == profile['train_batch_size'] and set(groups.values()) == {profile['group_size']}, 'training batch or group size changed'
    validation = rows(root, 'validation')
    assert len(validation) == test_size*len(profile['evaluation_steps']), 'missing independent held-out evaluation'
    assert {r.get('trainer_step') for r in validation} == set(profile['evaluation_steps']), 'evaluation schedule changed'
    assert all(sum(r.get('trainer_step') == step for r in validation) == test_size for step in profile['evaluation_steps']), 'incomplete evaluation points'
    worker = rows(root, 'worker_events')
    assert len(worker) == steps and all(sum(row['optimizer_state_bytes'].values()) >= 3_900_000_000 for row in worker), 'missing full-parameter Adam state'
    objective = rows(root, 'objective')
    assert len(objective) >= steps*epochs*(profile['train_batch_size']*profile['group_size']//profile['actor_micro_batch_size']), 'missing actual loss calls'
    # A correct mandatory-tool policy can already score every completion equally.
    # Its zero GRPO advantage is legitimate; entropy still drives real updates.
    if kind != 'environment':
        assert max(row['advantage_abs_max'] for row in objective) > 0, 'no nonzero group-relative advantage'
    result = {'completed_steps': len(updates), 'optimizer_updates': epochs*steps,
              'training_completions': len(samples), 'held_out_completions': len(validation),
              'initial_accuracy': final_evaluation(root, data_root, 0)[0],
              'final_accuracy': final_evaluation(root, data_root, steps)[0],
              'max_parameter_change': max(row['actor/parameter_probe_change'] for row in updates),
              'max_grad_norm': max(row['actor/grad_norm'] for row in updates)}
    result['max_advantage'] = max(row['advantage_abs_max'] for row in objective)
    if kind == 'reward':
        train = source_rows(data_root, 'train')
        mismatches = sum(row['score'] != score_answer(row['response'], train[int(row['index'])]['reward_model']['ground_truth']) for row in samples)
        result['reward_mismatches'] = mismatches
        assert mismatches == 0, f'{mismatches} generated responses receive an incorrect reward'
    elif kind == 'data':
        dataset = rows(root, 'dataset')
        train = source_rows(data_root, 'train')
        assert {int(r['index']) for r in dataset if r['split'] == 'train'} == set(train), 'training-label evidence does not cover the source split'
        mismatches = sum(row['training_answer'] != train[int(row['index'])]['reward_model']['ground_truth'] for row in dataset if row['split'] == 'train')
        result['label_mismatches'] = mismatches
        assert mismatches == 0, f'{mismatches} labels differ from the immutable source records'
    elif kind == 'environment':
        tools = rows(root, 'tools')
        try:
            from .python_tool_evidence import evaluate_python_tools
        except ImportError:
            from python_tool_evidence import evaluate_python_tools
        result.update(evaluate_python_tools(tools, data_root,
            steps*profile['train_batch_size']+test_size*len(profile['evaluation_steps']), training_samples=samples))
    elif kind == 'configuration':
        checkpoint = rows(root, 'checkpoint')
        assert len(checkpoint) == 1 and checkpoint[0]['scheduler_epoch'] == 4, 'checkpoint scheduler epoch was not restored'
        assert checkpoint[0]['optimizer_steps'] == [4.0], 'checkpoint optimizer state was not restored'
        assert all(row['actor/scheduler_epoch'] == row['step'] for row in updates), 'scheduler epoch diverges after resume'
        assert all(abs(row['actor/lr_used'] - 1e-5*min((row['step']-1)/6, 1)) < 1e-12 for row in updates), 'resumed learning-rate trajectory differs from uninterrupted warmup'
        result['restored_scheduler_epoch'] = checkpoint[0]['scheduler_epoch']
    elif kind == 'algorithm':
        loss_error = max(abs(row['loss']-row['behavior_objective']) for row in objective)
        kl_error = max(abs(row['reported_kl']-row['behavior_kl']) for row in objective)
        result.update(max_loss_error=loss_error, max_kl_error=kl_error,
                      max_clip_fraction=max(row['clip_fraction'] for row in objective),
                      max_logp_delta=max(row['logp_delta_max'] for row in objective))
        assert loss_error < 1e-5 and kl_error < 1e-5, 'loss ignores the frozen rollout behavior policy'
        assert result['max_logp_delta'] > 0.01 and result['max_clip_fraction'] > 0.001, 'rollout reuse did not exercise clipping'
    elif kind == 'infrastructure':
        transfers = rows(root, 'weight_transfer')
        assert len(transfers) >= steps+4, 'missing actor-to-vLLM weight transfers'
        assert all(row['compared_parameters'] > 20 for row in transfers), 'insufficient actual vLLM parameter coverage'
        result['max_weight_difference'] = max(row['max_difference'] for row in transfers)
        assert result['max_weight_difference'] == 0, 'vLLM parameters differ from the current actor after synchronization'
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    parser.add_argument('kind')
    parser.add_argument('--data', type=Path, required=True)
    args = parser.parse_args()
    try:
        print(json.dumps({'passed': True, 'metrics': evaluate(args.run, args.kind, data_root=args.data)}, indent=2))
    except Exception as error:
        print(json.dumps({'passed': False, 'error': str(error)}, indent=2))
        raise SystemExit(1)
