"""Record real checkpoint phases and tensor placement without allocating tensors."""
import time
import json
import os
from pathlib import Path
import torch


def record_checkpoint(phase, step, optimizer):
    sizes = {}
    steps = []
    if optimizer is not None:
        for state in optimizer.state.values():
            for value in state.values():
                if torch.is_tensor(value):
                    device = value.device.type
                    sizes[device] = sizes.get(device, 0) + value.numel() * value.element_size()
            if 'step' in state:
                steps.append(float(state['step']))
    row = {'phase': phase, 'step': step, 'optimizer_state_bytes': sizes,
           'optimizer_step_min': min(steps) if steps else 0,
           'optimizer_step_max': max(steps) if steps else 0,
           'cuda_allocated_bytes': torch.cuda.memory_allocated(),
           'cuda_reserved_bytes': torch.cuda.memory_reserved(),
           'cuda_peak_allocated_bytes': torch.cuda.max_memory_allocated(),
           'time': time.time(),
           'host_memory_current_bytes': int(Path('/sys/fs/cgroup/memory.current').read_text()),
           'host_memory_limit_bytes': int(Path('/sys/fs/cgroup/memory.max').read_text()),
           'host_memory_stat': Path('/sys/fs/cgroup/memory.stat').read_text()}
    with (Path(os.environ['RUN_OUTPUT'])/'checkpoint_events.jsonl').open('a') as stream:
        stream.write(json.dumps(row)+'\n')
