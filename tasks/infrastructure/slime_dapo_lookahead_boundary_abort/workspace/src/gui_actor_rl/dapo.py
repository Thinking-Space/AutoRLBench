"""GUI adapter for slime's DAPO dynamic-sampling hook.

Variance is computed over independent episodes, never over GUI step children.
The original GUI environment reward and GRPO group normalization are retained.
"""
from collections import Counter
from dataclasses import dataclass, field
import json
import logging
import math
import os
from pathlib import Path
import statistics

from slime.rollout.filter_hub.base_types import DynamicFilterOutput
from slime.utils.types import Sample

logger = logging.getLogger(__name__)


def episode_rewards(args, group):
    """Reject incomplete or inconsistent groups; one reward per trajectory."""
    if len(group) != args.n_samples_per_prompt:
        return None, 'incomplete_group', None
    source = getattr(args, 'gui_reward_source', None) or os.getenv('GUI_REWARD_SOURCE', 'judge')
    rewards, indexes, group_ids, task_ids = [], set(), set(), set()
    for trajectory in group:
        children = trajectory if isinstance(trajectory, list) else [trajectory]
        if not children:
            return None, 'empty_trajectory', None
        first = children[0]
        if first.index in indexes:
            return None, 'duplicate_trajectory', None
        indexes.add(first.index)
        group_ids.add(first.group_index)
        values = []
        tokens = 0
        for child in children:
            if child.index != first.index or child.group_index != first.group_index:
                return None, 'inconsistent_identity', None
            task_ids.add((child.metadata.get('domain'), child.metadata.get('example_id')))
            if child.remove_sample or child.status not in (Sample.Status.COMPLETED, Sample.Status.TRUNCATED):
                return None, 'invalid_episode', None
            try:
                score = float(child.get_reward_value(args))
            except (TypeError, ValueError, KeyError):
                return None, 'invalid_reward', None
            if not math.isfinite(score) or not 0 <= score <= 1:
                return None, 'invalid_reward', None
            if source == 'env':
                try:
                    env_score = float(child.metadata['gui_score'])
                except (TypeError, ValueError, KeyError):
                    return None, 'invalid_reward', None
                if not math.isfinite(env_score) or not 0 <= env_score <= 1:
                    return None, 'invalid_reward', None
                if abs(score - env_score) > 1e-6:
                    return None, 'inconsistent_reward', None
            values.append(score)
            tokens += child.effective_response_length
        if tokens <= 0:
            return None, 'empty_loss_mask', None
        if max(values) - min(values) > 1e-9:
            return None, 'inconsistent_reward', None
        rewards.append(values[0])
    if len(group_ids) != 1 or None in group_ids or len(task_ids) != 1:
        return None, 'inconsistent_identity', None
    task = next(iter(task_ids))
    if not all(task):
        return None, 'missing_task_identity', None
    return rewards, None, task


@dataclass
class SamplingAudit:
    std_min: float = 1e-6
    accepted_tasks: set = field(default_factory=set)
    rows: list = field(default_factory=list)
    scheduler_metrics: dict = field(default_factory=dict)

    def qualify(self, args, group):
        """Inspect a completed group without assigning it to a training batch."""
        rewards, reason, task = episode_rewards(args, group)
        std = statistics.stdev(rewards) if rewards is not None else None
        if reason is None and std <= self.std_min:
            reason = 'zero_or_low_std'
        first = group[0][0] if group and isinstance(group[0], list) and group[0] else (group[0] if group else None)
        row = dict(task=task, group_index=getattr(first, 'group_index', None),
                   rewards=rewards, std=std, qualified=reason is None, keep=False,
                   reason=reason, disposition='qualified' if reason is None else 'rejected')
        self.rows.append(row)
        return row

    def assign(self, row):
        """Called in candidate order; qualification alone never consumes a task."""
        reason = row['reason']
        if reason is None and row['task'] in self.accepted_tasks:
            reason = 'duplicate_task'
        keep = reason is None
        if keep:
            self.accepted_tasks.add(row['task'])
        row.update(keep=keep, reason=reason, disposition='accepted' if keep else 'rejected')
        return DynamicFilterOutput(keep=keep, reason=reason)

    def check(self, args, group):
        return self.assign(self.qualify(args, group))

    def metrics(self):
        usable = [r for r in self.rows if r['rewards'] is not None]
        rewards = [v for r in usable for v in r['rewards']]
        accepted = [r for r in self.rows if r['keep']]
        prefix = 'rollout/dapo/'
        result = {prefix+'groups_examined':len(self.rows), prefix+'groups_accepted':len(accepted),
                  prefix+'acceptance_rate':len(accepted)/len(self.rows) if self.rows else 0,
                  prefix+'candidate_reward_mean':statistics.mean(rewards) if rewards else 0,
                  prefix+'candidate_full_success_rate':sum(v >= 1-1e-6 for v in rewards)/len(rewards) if rewards else 0,
                  prefix+'accepted_std_mean':statistics.mean(r['std'] for r in accepted) if accepted else 0}
        for reason, count in Counter(r['reason'] for r in self.rows if not r['keep'] and r['reason']).items():
            result[prefix+'rejected_'+reason] = count
        for app, count in Counter(r['task'][0] for r in accepted).items():
            result[prefix+'accepted_app_'+app] = count
        result.update(self.scheduler_metrics)
        return result


def check_gui_reward_nonzero_std(args, group, **kwargs):
    """slime --dynamic-sampling-filter-path contract, with rollout-local audit."""
    return args._gui_dapo_audit.check(args, group)


def configure(args):
    if args.n_samples_per_prompt < 2:
        raise ValueError('DAPO requires at least two independent episodes per task')
    if args.partial_rollout:
        raise ValueError('GUI DAPO does not support resuming partially executed desktops')
    if args.advantage_estimator != 'grpo' or not args.calculate_per_token_loss:
        raise ValueError('GUI DAPO requires GRPO advantages and per-token loss')
    std_min = float(os.getenv('GUI_DAPO_STD_MIN', '0.000001'))
    budget = int(os.getenv('GUI_DAPO_MAX_GROUPS', str(args.rollout_batch_size*3)))
    deadline = float(os.getenv('GUI_DAPO_TIMEOUT_S', '6000'))
    lookahead = int(os.getenv('GUI_DAPO_LOOKAHEAD_GROUPS', '0'))
    pool = int(os.getenv('GUI_POOL_MAX_ENVS', str(args.rollout_batch_size * args.n_samples_per_prompt)))
    concurrency = int(os.getenv('GUI_TRAJECTORY_CONCURRENCY', str(pool)))
    active = min(pool, concurrency, (args.rollout_batch_size + lookahead) * args.n_samples_per_prompt)
    cleanup = float(os.getenv('GUI_DAPO_CLEANUP_TIMEOUT_S', '180'))
    if not math.isfinite(std_min) or std_min < 0 or not math.isfinite(deadline) or deadline <= 0 or budget < args.rollout_batch_size:
        raise ValueError('Invalid GUI DAPO sampling bounds')
    if not 0 <= lookahead <= min(32, budget - args.rollout_batch_size):
        raise ValueError('Invalid GUI DAPO lookahead bounds')
    if active < args.n_samples_per_prompt or not math.isfinite(cleanup) or cleanup <= 0:
        raise ValueError('Invalid GUI DAPO capacity/cleanup bounds')
    args.dynamic_sampling_filter_path = 'gui_actor_rl.dapo.check_gui_reward_nonzero_std'
    args.rollout_max_groups = budget
    args.rollout_timeout_s = deadline
    args.rollout_lookahead_groups = lookahead
    args.rollout_max_active_episodes = active
    args.rollout_cleanup_timeout_s = cleanup
    args._gui_dapo_audit = SamplingAudit(std_min)
    logger.info('GUI DAPO configured: target_groups=%s samples_per_group=%s lookahead_groups=%s '
                'max_active_episodes=%s max_groups=%s batch_timeout_s=%s cleanup_timeout_s=%s',
                args.rollout_batch_size, args.n_samples_per_prompt, lookahead, active, budget, deadline, cleanup)
    return args._gui_dapo_audit


def write_audit(args, rollout_id, audit, error=None):
    directory = getattr(args, 'gui_timing_log_dir', None)
    if directory:
        path = Path(directory)/f'rollout_{rollout_id:04d}_dapo.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(dict(rollout_id=rollout_id, error=error, metrics=audit.metrics(), groups=audit.rows),indent=2)+'\n')
