"""Wait for shared-GPU capacity without changing model values or loss math."""
import functools
import json
import math
import os
import sys
import time
from pathlib import Path

GIB = 1024 ** 3


def wait_cuda_capacity(required_bytes, phase, *, coordinate=False):
    import torch
    if not torch.cuda.is_available():
        return
    torch.cuda.empty_cache()
    needed = int(required_bytes) + 4 * GIB
    group = None
    mpu = sys.modules.get("megatron.core.parallel_state")
    if coordinate and mpu is not None and torch.distributed.is_initialized():
        try:
            group = mpu.get_data_parallel_group_gloo(with_context_parallel=True)
        except AssertionError:
            group = None
    waited = False
    while True:
        free, total = torch.cuda.mem_get_info()
        if needed > total:
            raise RuntimeError(f"{phase}: allocation plus headroom ({needed}) exceeds GPU capacity ({total})")
        ready = free >= needed
        if group is not None:
            flag = torch.tensor([int(ready)], dtype=torch.int32, device="cpu")
            torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MIN, group=group)
            ready = bool(flag.item())
        if ready:
            if waited:
                print("CUDA_CAPACITY_READY " + json.dumps(dict(pid=os.getpid(), phase=phase,
                      free_bytes=free, needed_bytes=needed)), flush=True)
            return
        waited = True
        print("CUDA_CAPACITY_WAIT " + json.dumps(dict(pid=os.getpid(), phase=phase,
              free_bytes=free, needed_bytes=needed, coordinated=group is not None,
              visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"))), flush=True)
        time.sleep(30)


def reload_bytes(models, load_grad=True, load_frozen_params=True):
    required = 0
    for chunk in models:
        buffers = getattr(chunk, "buffers", None)
        if isinstance(buffers, list):
            for buffer in buffers + list(getattr(chunk, "expert_parallel_buffers", [])):
                if buffer.param_data.untyped_storage().nbytes() == 0:
                    required += buffer.param_data.numel() * buffer.param_data.element_size()
                if load_grad and hasattr(buffer, "grad_data_size") and buffer.grad_data.untyped_storage().nbytes() == 0:
                    required += buffer.grad_data_size * buffer.grad_data.element_size()
            if load_frozen_params:
                required += sum(p.numel() * p.element_size() for p in chunk.module.parameters()
                                if not p.requires_grad and p.device.type == "cpu")
        else:
            for p in chunk.parameters():
                if p.device.type == "cpu":
                    required += p.numel() * p.element_size()
                if load_grad and p.grad is not None and p.grad.device.type == "cpu":
                    required += p.grad.numel() * p.grad.element_size()
    return required


def patch_engine(module):
    original = module.load_megatron_model_to_gpu
    if getattr(original, "_runtime_capacity_wait", False):
        return

    @functools.wraps(original)
    def load(models, load_grad=True, load_frozen_params=True):
        needed = reload_bytes(models, load_grad, load_frozen_params)
        wait_cuda_capacity(needed, "megatron-reload-grad" if load_grad else "megatron-reload-weights",
                           coordinate=True)
        return original(models, load_grad=load_grad, load_frozen_params=load_frozen_params)
    load._runtime_capacity_wait = True
    module.load_megatron_model_to_gpu = load
    build_original = module.MegatronEngine._build_megatron_module

    @functools.wraps(build_original)
    def build(self, *args, **kwargs):
        needed = model_build_bytes(self)
        if needed:
            wait_cuda_capacity(needed, "megatron-model-build", coordinate=True)
        return build_original(self, *args, **kwargs)
    build._runtime_capacity_wait = True
    module.MegatronEngine._build_megatron_module = build


def model_build_bytes(engine):
    """Use the hash-verified full model's recorded count, not a smaller model."""
    config = engine.model_config.hf_config
    if not (getattr(config, "model_type", None) == "qwen2_moe"
            and getattr(config, "hidden_size", None) == 2048
            and getattr(config, "num_hidden_layers", None) == 24
            and getattr(config, "num_experts", None) == 60):
        return 0
    import torch
    ready = json.loads((Path(os.environ["W"]) / "metadata/REAL_READY.json").read_text())
    assert ready["model_id"] == "Qwen/Qwen1.5-MoE-A2.7B-Chat"
    assert ready["parameters"] == 14315784192
    ep = getattr(getattr(engine, "engine_config", None), "expert_model_parallel_size", 1)
    assert config.num_experts % ep == 0
    expert_parameters = (config.num_hidden_layers * config.num_experts * config.hidden_size
                         * config.moe_intermediate_size * 3)
    # Only the routed experts are sharded by EP. All dense/shared/gate weights
    # remain present on each CP rank; globally this is the same full model.
    per_rank_parameters = ready["parameters"] - expert_parameters + expert_parameters // ep
    return per_rank_parameters * torch.empty((), dtype=engine.param_dtype, device="cpu").element_size()


def patch_vllm_worker(module):
    original = module.request_memory
    if getattr(original, "_runtime_capacity_wait", False):
        return

    @functools.wraps(original)
    def request(snapshot, config):
        needed = math.ceil(snapshot.total_memory * config.gpu_memory_utilization)
        while snapshot.free_memory < needed:
            print("CUDA_CAPACITY_WAIT " + json.dumps(dict(pid=os.getpid(), phase="vllm-startup",
                  free_bytes=snapshot.free_memory, needed_bytes=needed,
                  visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"))), flush=True)
            time.sleep(30)
            snapshot.measure()
        return original(snapshot, config)
    request._runtime_capacity_wait = True
    module.request_memory = request
    original_wake = module.Worker.wake_up

    @functools.wraps(original_wake)
    def wake(self, tags=None):
        from vllm.device_allocator.cumem import CuMemAllocator
        allocator = CuMemAllocator.get_instance()
        needed = sum(data.handle[1] for data in allocator.pointer_to_data.values()
                     if tags is None or data.tag in tags)
        wait_cuda_capacity(needed, "vllm-wake-" + str(tags))
        return original_wake(self, tags)
    module.Worker.wake_up = wake
