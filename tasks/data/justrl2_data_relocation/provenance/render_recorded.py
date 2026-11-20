"""Render unchanged historical observations; not part of preparation or grading."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    logs = Path(__file__).resolve().parents[1] / 'workspace/logs/recorded'
    rows = json.loads((logs / 'metrics.json').read_text())
    evaluation = json.loads((logs / 'eval.json').read_text())
    steps = [row['step'] for row in rows]
    plt.rcParams.update({'font.size': 10, 'svg.hashsalt': 'justrl2-recorded', 'axes.spines.top': False,
                         'axes.spines.right': False})
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 3.5), layout='constrained')
    series = lambda name: [row['metrics'][name] for row in rows]
    axes[0].plot(steps, series('data/survivor_reward_mean'), 'o-', label='Admitted-sample reward')
    axes[0].plot(steps, series('data/reward_per_read_attempt'), 's-', label='Reward / read attempt')
    measured = [row for row in evaluation if row['rollout_id'] is not None]
    axes[0].plot([row['rollout_id'] for row in measured], [row['reward_mean'] for row in measured],
                 'D--', color='#64748b', label='Fixed evaluation (16)')
    axes[0].set(title='Reward denominators', ylabel='Reward', ylim=(0, 1))
    axes[1].plot(steps, series('data/attempted_hard'), 'o-', label='Attempted hard samples')
    axes[1].plot(steps, series('data/admitted_hard'), 's-', label='Admitted hard samples')
    axes[1].set(title='Hard-sample coverage', ylabel='Count', ylim=(-0.5, 20))
    axes[2].plot(steps, series('data/abort_count'), 'o-', label='Read aborts')
    axes[2].plot(steps, series('data/generation_abort_count'), 's-', label='Generation aborts')
    axes[2].set(title='Abort accounting', ylabel='Count', ylim=(-0.5, 20))
    for axis in axes:
        axis.set(xlabel='Recorded step', xticks=steps)
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8, loc='upper left')
    fig.suptitle('Historical observations — earlier eight-step run', fontsize=13)
    output = logs / 'curves.svg'
    fig.savefig(output, metadata={'Date': None})
    output.write_text('\n'.join(line.rstrip() for line in output.read_text().splitlines()) + '\n')
    plt.close(fig)


if __name__ == '__main__':
    main()
