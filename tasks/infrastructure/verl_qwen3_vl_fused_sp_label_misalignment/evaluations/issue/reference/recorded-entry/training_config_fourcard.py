"""Four-card execution, with the approved global training recipe unchanged."""
from training_config import overrides as original_overrides


def overrides(root, run):
    changes = {
        'trainer.n_gpus_per_node': '4',
        'trainer.resume_mode': 'disable',
        'trainer.experiment_name': 'quality_fourcard',
        'ray_kwargs.ray_init.num_cpus': '16',
    }
    result = []
    for value in original_overrides(root, run, False):
        key = value.split('=', 1)[0]
        result.append(f'{key}={changes[key]}' if key in changes else value)
    result.append('actor_rollout_ref.actor.fsdp_config.fsdp_size=4')
    return result
