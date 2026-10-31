"""Isolated synthetic data-fault plugin; generation/rewards/PPO stay native.

Fault injection is deliberately confined to a prepared backing store. Missing
files become native ABORTED samples; the native check_no_aborted filter drops
their groups and resamples. Other I/O and content errors are fatal.
"""
import json
import math
import os
from pathlib import Path
import time

_SOURCE = None


def write_evaluation(directory, rollout_id, last_training_step, data):
    phase = 'before_train' if last_training_step < 0 else f'after_step_{rollout_id:07d}'
    path = Path(directory) / f'eval_{phase}.json'
    serialized = {}
    for name, values in data.items():
        serialized[name] = {
            'rewards': values['rewards'], 'truncated': values['truncated'],
            'samples': [{'index': s.index, 'prompt': s.prompt, 'label': s.label,
                         'response': s.response, 'response_length': s.response_length,
                         'reward': s.reward, 'status': s.status.value, 'metadata': s.metadata}
                        for s in values['samples']],
        }
    with path.open('x') as stream:
        json.dump(serialized, stream, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
    return path


def summarize(reads, admitted):
    if any(r['outcome'] not in ('ok', 'missing') for r in reads):
        raise ValueError('Unknown file-read outcome')
    if any(r['difficulty'] not in ('easy', 'hard') for r in [*reads, *admitted]):
        raise ValueError('Unknown difficulty')
    rewards = [float(row['reward']) for row in admitted]
    if any(not math.isfinite(r) or not 0 <= r <= 1 for r in rewards):
        raise ValueError('Expected finite, unmodified 0/1 math rewards')
    ok = sum(r['outcome'] == 'ok' for r in reads)
    aborted = len(reads) - ok
    if not admitted or ok < len(admitted):
        raise ValueError('Admitted samples must have successful file reads')
    result = {
        'data/read_attempts': len(reads),
        'data/abort_count': aborted,
        'data/abort_rate': aborted / len(reads),
        'data/admitted_count': len(admitted),
        'data/read_ok_not_admitted': ok - len(admitted),
        'data/admitted_hard_fraction': sum(r['difficulty'] == 'hard' for r in admitted) / len(admitted),
        'data/reward_sum': sum(rewards),
        'data/survivor_reward_mean': sum(rewards) / len(rewards),
        # Explicit diagnostic denominator: extra completed/cancelled reads are
        # not graded as failures. This is reward admitted per read attempt.
        'data/admitted_reward_per_read_attempt': sum(rewards) / len(reads),
    }
    for group in ['easy', 'hard']:
        rr = [float(r['reward']) for r in admitted if r['difficulty'] == group]
        result[f'data/admitted_{group}'] = len(rr)
        result[f'data/attempted_{group}'] = sum(r['difficulty'] == group for r in reads)
        result[f'data/abort_{group}'] = sum(r['difficulty'] == group and r['outcome'] == 'missing' for r in reads)
        if rr:
            result[f'data/reward_{group}_mean'] = sum(rr) / len(rr)
    return result


def terminal_metrics(reads, terminals, admitted_indices, requested_indices):
    outcomes = {row['index']: row for row in terminals}
    if len(outcomes) != len(terminals) or set(outcomes) != set(requested_indices):
        raise ValueError('Missing or duplicate terminal outcomes')
    by_read = {row['index']: row for row in reads}
    if len(by_read) != len(reads) or not set(by_read) <= set(outcomes):
        raise ValueError('Read attempts lack unique terminal outcomes')
    admitted = set(admitted_indices)
    if not admitted <= set(by_read):
        raise ValueError('Admitted samples lack read attempts')
    scores = []
    generation_aborts = pre_read_cancelled = surplus_scored = truncated = 0
    for index, row in outcomes.items():
        status, reward = row['status'], row['reward']
        if status in ('completed', 'truncated'):
            if index not in by_read or by_read[index]['outcome'] != 'ok':
                raise ValueError('Scored sample lacks successful read')
            if reward not in (0, 1):
                raise ValueError('Expected raw binary math reward')
            scores.append(float(reward))
            surplus_scored += index not in admitted
            truncated += status == 'truncated'
        elif status in ('aborted', 'pending'):
            if reward is not None or index in admitted:
                raise ValueError('Aborted/pending sample was graded or admitted')
            if index not in by_read:
                pre_read_cancelled += 1
            elif by_read[index]['outcome'] == 'ok':
                generation_aborts += 1
        else:
            raise ValueError('Unexpected generation terminal status: ' + status)
    if not reads:
        raise ValueError('No file reads')
    return {'data/terminal_count': len(outcomes), 'data/requested_count': len(requested_indices),
            'data/generation_abort_count': generation_aborts,
            'data/scheduler_cancel_before_read': pre_read_cancelled,
            'data/scored_count': len(scores), 'data/scored_not_admitted': surplus_scored,
            'data/truncated_count': truncated, 'data/all_scored_reward_sum': sum(scores),
            'data/reward_per_read_attempt': sum(scores) / len(reads)}


def install_outcome_observer():
    """Observe the native group result after grading, including drained tasks."""
    from miles.rollout import sglang_rollout as native
    original = native.generate_and_rm_group
    if getattr(original, '_relocation_observer', False):
        return

    async def observed(args, group, sampling_params, evaluation=False):
        result = await original(args, group, sampling_params, evaluation=evaluation)
        if not evaluation:
            if _SOURCE is None:
                raise RuntimeError('Missing source for terminal observation')
            for sample in result:
                record = {'id': sample.metadata['backing_ref']['id'], 'index': sample.index,
                          'group_index': sample.group_index, 'difficulty': sample.metadata['difficulty'],
                          'status': sample.status.value, 'reward': sample.reward,
                          'response_length': sample.response_length}
                _SOURCE.terminals.append(_SOURCE.emit('terminal', **record,
                                                      response=sample.response, label=sample.label))
        return result
    observed._relocation_observer = True
    native.generate_and_rm_group = observed


class BackedMathSource:
    """Native datasource contract, storing references instead of prompt bodies."""
    def __init__(self, args):
        from storage import BackingStore
        global _SOURCE
        self.args = args
        self.store = BackingStore(Path(os.environ['RELOCATION_DATA']))
        self.output = Path(os.environ['RELOCATION_OUTPUT'])
        self.output.mkdir(parents=True, exist_ok=True)
        self.arm = os.environ['RELOCATION_ARM']
        if self.arm not in ('healthy', 'faulty', 'repaired'):
            raise ValueError('Unknown experiment arm')
        self.at = int(os.environ['RELOCATION_AT'])
        if self.at <= 0:
            raise ValueError('Migration must occur after training has begun')
        self.dataset = self.store.rows
        self.cursor = self.sample_index = self.group_index = 0
        self.step = -1
        self.requests = 0
        self.reads = []
        self.terminals = []
        self.requested_indices = set()
        self.metadata = {}
        _SOURCE = self

    def emit(self, kind, **fields):
        event = {'kind': kind, 'step': self.step, 'arm': self.arm,
                 'wall_time': time.time(), **fields}
        with (self.output / 'events.jsonl').open('a') as f:
            f.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + '\n')
        return event

    def begin(self, step):
        self.step, self.requests, self.reads = step, 0, []
        self.terminals, self.requested_indices = [], set()
        if step >= self.at and self.arm != 'healthy':
            receipt = self.store.relocate(repair=self.arm == 'repaired')
            self.dataset = self.store.rows
            self.emit('relocation', receipt=receipt)
        self.emit('rollout_begin', cursor=self.cursor)

    def get_samples(self, num_samples):
        from miles.utils.types import Sample
        self.requests += num_samples
        # Bound resampling even if every backing file is unavailable.
        if self.requests > 16 * self.args.rollout_batch_size:
            raise RuntimeError('Backing-file retry budget exhausted')
        groups = []
        for _ in range(num_samples):
            row = dict(self.store.rows[self.cursor % len(self.store.rows)])
            self.cursor += 1
            group = []
            for _ in range(self.args.n_samples_per_prompt):
                group.append(Sample(group_index=self.group_index, index=self.sample_index,
                                    metadata={'backing_ref': row, 'difficulty': row['difficulty']}))
                self.requested_indices.add(self.sample_index)
                self.sample_index += 1
            self.group_index += 1
            groups.append(group)
        return groups

    def add_samples(self, samples):
        if samples:
            raise RuntimeError('Partial-rollout buffers must be disabled for this case')

    def save(self, rollout_id):
        target = Path(self.args.save) / 'rollout' / f'data_state_{rollout_id}.json'
        target.parent.mkdir(parents=True, exist_ok=True)
        state = {'cursor': self.cursor, 'sample_index': self.sample_index,
                 'group_index': self.group_index, 'step': self.step, 'arm': self.arm}
        temp = target.with_suffix('.tmp')
        temp.write_text(json.dumps(state) + '\n')
        temp.replace(target)

    def load(self, rollout_id=None):
        if not self.args.load or rollout_id is None:
            return
        target = Path(self.args.load) / 'rollout' / f'data_state_{rollout_id}.json'
        if not target.exists():
            if rollout_id >= 0 and Path(self.args.load, 'latest_checkpointed_iteration.txt').exists():
                raise RuntimeError('Checkpoint is missing matching datasource state')
            return
        state = json.loads(target.read_text())
        if state['arm'] != self.arm:
            raise ValueError('Cannot resume another experiment arm')
        self.cursor, self.sample_index, self.group_index = (state[k] for k in ('cursor', 'sample_index', 'group_index'))

    def get_metadata(self):
        return self.metadata

    def update_metadata(self, metadata):
        self.metadata.update(metadata)

    def get_buffer_length(self):
        return 0


async def generate(args, sample, sampling_params, evaluation=False):
    from miles.rollout.sglang_rollout import GenerateState, generate as native_generate
    from miles.utils.types import Sample
    reference = sample.metadata.get('backing_ref')
    if reference is None:
        # Fixed evaluation data is inline, outside the migrated backing store.
        if not evaluation:
            raise ValueError('Training samples require a backing reference')
        return await native_generate(args, sample, sampling_params)
    source = _SOURCE
    if source is None or evaluation:
        raise RuntimeError('Invalid lazy datasource context')
    fields = {'id': reference['id'], 'index': sample.index,
              'group_index': sample.group_index, 'difficulty': reference['difficulty'],
              'path': reference['path']}
    try:
        payload = source.store.read(reference)
    except FileNotFoundError as error:
        source.reads.append(source.emit('read', outcome='missing', errno=error.errno, **fields))
        sample.status = Sample.Status.ABORTED
        sample.metadata['env_error'] = 'backing_file_unavailable'
        return sample
    source.reads.append(source.emit('read', outcome='ok', sha256=reference['sha256'], **fields))
    state = GenerateState(args)
    sample.prompt = state.tokenizer.apply_chat_template(
        payload['prompt'], tokenize=False, add_generation_prompt=True,
        **(args.apply_chat_template_kwargs or {}))
    sample.label = payload['label']
    sample.metadata.update(payload.get('metadata', {}))
    return await native_generate(args, sample, sampling_params)


def generate_rollout(args, rollout_id, data_source, evaluation=False):
    from miles.rollout.sglang_rollout import generate_rollout as native_rollout
    if evaluation:
        output = native_rollout(args, rollout_id, data_source, evaluation=True)
        # Full native evaluation result is retained in the private run directory.
        write_evaluation(data_source.output, rollout_id, data_source.step, output.data)
        return output
    if args.partial_rollout or args.over_sampling_batch_size != args.rollout_batch_size:
        raise ValueError('This case requires no partial buffers and equal initial/target batch sizes')
    if args.dynamic_sampling_filter_path != 'miles.rollout.filter_hub.dynamic_sampling_filters.check_no_aborted':
        raise ValueError('Use the native aborted-group filter without reward filtering')
    data_source.begin(rollout_id)
    install_outcome_observer()
    output = native_rollout(args, rollout_id, data_source, evaluation=False)
    admitted = []
    for group in output.samples:
        for sample in group:
            if sample.status.name not in ('COMPLETED', 'TRUNCATED'):
                raise ValueError('Non-generated sample reached training')
            record = {'id': sample.metadata['backing_ref']['id'], 'index': sample.index,
                      'group_index': sample.group_index, 'difficulty': sample.metadata['difficulty'],
                      'reward': sample.reward, 'status': sample.status.value,
                      'response_length': sample.response_length}
            admitted.append(record)
            data_source.emit('admitted', **record, response=sample.response, label=sample.label)
    metrics = summarize(data_source.reads, admitted)
    metrics.update(terminal_metrics(data_source.reads, data_source.terminals,
                                   [row['index'] for row in admitted], data_source.requested_indices))
    output.metrics = {**(output.metrics or {}), **metrics}
    data_source.emit('rollout_end', metrics=metrics)
    print('DATA_RELOCATION_METRICS ' + json.dumps({'step': rollout_id, **metrics}), flush=True)
    return output
