"""Lower MCore 0.16 BF16 DDP initialization peak; do not alter loss or gradients.

MCore allocates both flat buffers before remapping individual parameters. Staging
the old values on CPU avoids simultaneously retaining three full GPU copies.
The original constructor still owns the layout, value copy and main-grad mapping.
Installed per Ray worker, only for this experiment, without editing the image.
"""
import functools
import importlib.abc
import importlib.machinery
import importlib.metadata
import json
import os
import sys

_expert_communication_ready = False


def _install_collective_diagnostics(torch):
    original = torch.distributed.all_gather
    if getattr(original, "_runtime_diagnostics", False):
        return
    calls = 0

    @functools.wraps(original)
    def gather(output, tensor, *args, **kwargs):
        nonlocal calls
        calls += 1
        if tensor.is_cuda and calls <= 4:
            print("COLLECTIVE_MEMORY " + json.dumps(dict(
                pid=os.getpid(), call=calls, shape=list(tensor.shape),
                free_bytes=torch.cuda.mem_get_info()[0],
                allocated_bytes=torch.cuda.memory_allocated(),
                reserved_bytes=torch.cuda.memory_reserved(),
                allocator=torch.cuda.memory.get_allocator_backend(),
                visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
            )), flush=True)
        try:
            return original(output, tensor, *args, **kwargs)
        except Exception:
            if tensor.is_cuda:
                print("COLLECTIVE_FAILURE_MEMORY " + json.dumps(dict(
                    pid=os.getpid(), free_bytes=torch.cuda.mem_get_info()[0],
                    allocated_bytes=torch.cuda.memory_allocated(), reserved_bytes=torch.cuda.memory_reserved(),
                )), flush=True)
            raise

    gather._runtime_diagnostics = True
    torch.distributed.all_gather = gather


def _warm_expert_communication(torch):
    global _expert_communication_ready
    if _expert_communication_ready or not torch.distributed.is_initialized():
        return
    from megatron.core import parallel_state
    group = parallel_state.get_expert_model_parallel_group(check_initialized=False)
    if group is None:
        return
    value = torch.ones(1, device=torch.cuda.current_device(), dtype=torch.float32)
    gathered = [torch.empty_like(value) for _ in range(torch.distributed.get_world_size(group))]
    torch.distributed.all_gather(gathered, value, group=group)
    torch.cuda.synchronize()
    _expert_communication_ready = True
    print("EXPERT_NCCL_WARMUP " + json.dumps(dict(
        pid=os.getpid(), group_size=len(gathered), free_bytes=torch.cuda.mem_get_info()[0],
    )), flush=True)


def _patch_buffer(module):
    import torch
    _install_collective_diagnostics(torch)
    _ParamAndGradBuffer = module._ParamAndGradBuffer

    if getattr(_ParamAndGradBuffer.__init__, "_runtime_cpu_staging", False):
        return
    if importlib.metadata.version("megatron-core") != "0.16.0":
        raise RuntimeError("This initialization workaround was scoped to Megatron-Core 0.16.0")
    original = _ParamAndGradBuffer.__init__

    @functools.wraps(original)
    def initialize(self, ddp_config, param_dtype, grad_dtype, params, *args, **kwargs):
        stage = (
            ddp_config.use_distributed_optimizer
            and param_dtype == torch.bfloat16
            and grad_dtype == torch.bfloat16
            and not ddp_config.reuse_grad_buf_for_mxfp8_param_ag
        )
        staged_bytes = 0
        if stage:
            # Preserve Parameter identity and attributes. Do not retain old GPU
            # tensor references: they must be released before the flat allocation.
            for param in params:
                if param.is_cuda:
                    staged_bytes += param.numel() * param.element_size()
                param.data = param.data.to(device="cpu")
            if staged_bytes:
                torch.cuda.empty_cache()
                from runtime_memory_wait import wait_cuda_capacity
                # Native construction allocates both BF16 flat buffers.
                wait_cuda_capacity(sum(p.numel() * p.element_size() for p in params) * 2,
                                   "ddp-flat-buffers", coordinate=True)
                # Initialize the communicator used by mbridge export while
                # temporary model values are on CPU, before vLLM is resident.
                _warm_expert_communication(torch)
                print("CPU_STAGED_DDP " + json.dumps(dict(
                    pid=os.getpid(), staged_bytes=staged_bytes,
                    cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                    logical_cuda_device=str(torch.cuda.current_device()),
                )), flush=True)
        original(self, ddp_config, param_dtype, grad_dtype, params, *args, **kwargs)

    initialize._runtime_cpu_staging = True
    _ParamAndGradBuffer.__init__ = initialize


class _BufferLoader(importlib.abc.Loader):
    def __init__(self, original):
        self.original = original

    def create_module(self, spec):
        return self.original.create_module(spec)

    def exec_module(self, module):
        self.original.exec_module(module)
        _patch_module(module)


def _patch_tracking(module):
    cls = module.Tracking
    if getattr(cls.log, "_runtime_explicit_wandb_finish", False):
        return
    old_init, old_log = cls.__init__, cls.log

    @functools.wraps(old_init)
    def initialize(self, *args, **kwargs):
        old_init(self, *args, **kwargs)
        config = kwargs.get("config") or {}
        self._runtime_final_logging_step = config.get("trainer", {}).get("total_training_steps")

    @functools.wraps(old_log)
    def log(self, data, step, backend=None):
        old_log(self, data=data, step=step, backend=backend)
        final_step = getattr(self, "_runtime_final_logging_step", None)
        if final_step is not None and step >= int(final_step) and "wandb" in self.logger:
            # Flush while the Ray task and its W&B service are still alive.
            # This does not change model computation, optimizer state or metrics.
            self.logger["wandb"].finish(exit_code=0)
            self.logger.pop("wandb")  # Already finished; avoid a second destructor call.
            print("WANDB_FINAL_STEP_FINISHED " + json.dumps(dict(pid=os.getpid(), step=step)), flush=True)

    log._runtime_explicit_wandb_finish = True
    cls.__init__, cls.log = initialize, log
    print("WANDB_LIFECYCLE_HOOK_INSTALLED " + str(os.getpid()), flush=True)


_TARGETS = (
    "megatron.core.distributed.param_and_grad_buffer",
    "verl.workers.rollout.vllm_rollout.vllm_rollout",
    "verl.workers.engine.megatron.transformer_impl",
    "vllm.v1.worker.gpu_worker",
    "verl.utils.tracking",
)


def _patch_module(module):
    if module.__name__ == _TARGETS[0]:
        _patch_buffer(module)
    elif module.__name__ == _TARGETS[1]:
        # Native CUDA IPC export independently fails even for a 1 MiB tensor
        # on this host. Select verl's existing shared-memory transfer path.
        def no_cuda_ipc():
            return False
        no_cuda_ipc._runtime_shm_fallback = True
        module.is_support_ipc = no_cuda_ipc
    elif module.__name__ == _TARGETS[2]:
        from runtime_memory_wait import patch_engine
        patch_engine(module)
    elif module.__name__ == _TARGETS[3]:
        from runtime_memory_wait import patch_vllm_worker
        patch_vllm_worker(module)
    else:
        _patch_tracking(module)


class _BufferFinder(importlib.abc.MetaPathFinder):
    _runtime_deferred_staging = True

    def find_spec(self, fullname, path=None, target=None):
        if fullname not in _TARGETS:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or spec.loader is None:
            return None
        spec.loader = _BufferLoader(spec.loader)
        return spec


def install():
    # Ray executes this hook before assigning the next actor's accelerator mask.
    # Importing MCore here can initialize CUDA under the parent's 4,5 mask,
    # making both later single-GPU actors retain physical GPU 4 as cuda:0.
    # Wait for the actor's own natural MCore import; never import torch/MCore
    # or call a CUDA API from worker-process setup.
    for name in _TARGETS:
        module = sys.modules.get(name)
        if module is not None:
            _patch_module(module)
    if not any(getattr(finder, "_runtime_deferred_staging", False) for finder in sys.meta_path):
        sys.meta_path.insert(0, _BufferFinder())

