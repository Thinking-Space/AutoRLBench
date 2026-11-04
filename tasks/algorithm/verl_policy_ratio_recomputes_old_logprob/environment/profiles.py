"""Fixed task-specific experiment budgets, shared by builders and graders."""


def experiment_profile(kind, steps=None):
    algorithm = kind == 'algorithm'
    total = steps if steps is not None else (30 if algorithm else 12)
    frequency = 10 if algorithm else 4
    return {'steps': total, 'train_rows': 256, 'held_out_rows': 100 if algorithm else 64,
            'train_batch_size': 16 if algorithm else 4,
            'group_size': 8 if algorithm else 4,
            'actor_micro_batch_size': 8 if algorithm else 2,
            'optimizer_passes': 4 if algorithm else 1,
            'eval_every': frequency,
            'evaluation_steps': sorted({0, total, *range(frequency, total+1, frequency)}),
            'validation_sample_seed': 42 if algorithm else None,
            'training_timeout_s': 7200 if algorithm else 1800}
