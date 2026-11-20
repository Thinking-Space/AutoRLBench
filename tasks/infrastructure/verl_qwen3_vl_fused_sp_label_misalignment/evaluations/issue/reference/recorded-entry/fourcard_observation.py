"""Observe actual FSDP mesh and optimizer allocation; preserve all computations."""
import functools
import json
import os
from pathlib import Path
import time


def emit(row):
    import torch.distributed as dist
    directory = Path(os.environ['BENCH_FOURCARD_DIR'])
    directory.mkdir(parents=True, exist_ok=True)
    row.update(pid=os.getpid(), global_rank=dist.get_rank(), time=time.time())
    with (directory / f'runtime-{os.getpid()}.jsonl').open('a') as stream:
        stream.write(json.dumps(row) + '\n')


def install(engine):
    if getattr(engine, '_fourcard_observed', False):
        return
    engine._fourcard_observed = True
    import torch
    import torch.distributed as dist
    assert dist.get_world_size() == 4
    if not hasattr(engine, 'device_mesh'):
        emit(dict(kind='numerical_probe', world_size=4, not_fsdp_training=True))
        return
    assert tuple(engine.device_mesh.shape) == (4,)
    assert tuple(engine.ulysses_device_mesh.shape) == (2, 2)
    assert engine.get_data_parallel_size() == 2
    assert engine.ulysses_sequence_parallel_size == 2
    strategy = str(engine.module.sharding_strategy)
    assert strategy.endswith('FULL_SHARD'), strategy
    emit(dict(kind='mesh', world_size=4, fsdp_mesh=engine.device_mesh.mesh.tolist(),
              ulysses_mesh=engine.ulysses_device_mesh.mesh.tolist(), dp=2, sp=2,
              sharding_strategy=strategy, optimizer=type(engine.optimizer).__name__))
    if engine.optimizer is None:
        return
    original = engine.optimizer_step
    ordinal = 0

    @functools.wraps(original)
    def observed_step(*args, **kwargs):
        nonlocal ordinal
        ordinal += 1
        before = torch.cuda.memory_allocated()
        try:
            result = original(*args, **kwargs)
        except BaseException as exc:
            emit(dict(kind='optimizer_failure', ordinal=ordinal, error=repr(exc),
                      allocated_before=before, allocated_after=torch.cuda.memory_allocated(),
                      peak_allocated=torch.cuda.max_memory_allocated(),
                      peak_reserved=torch.cuda.max_memory_reserved()))
            raise
        emit(dict(kind='optimizer_step', ordinal=ordinal, allocated_before=before,
                  allocated_after=torch.cuda.memory_allocated(),
                  peak_allocated=torch.cuda.max_memory_allocated(),
                  peak_reserved=torch.cuda.max_memory_reserved(),
                  free_bytes=torch.cuda.mem_get_info()[0],
                  peak_scope='cumulative CUDA allocator peak, no stats reset'))
        return result

    engine.optimizer_step = observed_step
