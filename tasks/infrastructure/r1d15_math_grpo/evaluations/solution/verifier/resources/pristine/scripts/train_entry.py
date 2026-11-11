"""Entrypoint with a bounded, task-owned Ray temporary directory."""
import os
from pathlib import Path
import ray

original_init = ray.init
def init(*args, **kwargs):
    kwargs.setdefault('_temp_dir', os.environ['RAY_TMPDIR'])
    kwargs.setdefault('object_store_memory', 4 * 1024**3)
    kwargs.setdefault('include_dashboard', False)
    return original_init(*args, **kwargs)
ray.init = init

from verl.trainer.main_ppo import main
try:
    main()
finally:
    ray.shutdown()
