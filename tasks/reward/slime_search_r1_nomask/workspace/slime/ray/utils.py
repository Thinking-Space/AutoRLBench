# Adapted from https://github.com/OpenRLHF/OpenRLHF/blob/10c733694ed9fbb78a0a2ff6a05efc7401584d46/openrlhf/trainer/ray/utils.py#L1
import os
import sys
from pathlib import Path

import ray
import torch
from slime.ray.ray_actor import RayActor

# Refer to
# https://github.com/ray-project/ray/blob/161849364a784442cc659fb9780f1a6adee85fce/python/ray/_private/accelerators/nvidia_gpu.py#L95-L96
# https://github.com/ray-project/ray/blob/161849364a784442cc659fb9780f1a6adee85fce/python/ray/_private/accelerators/amd_gpu.py#L102-L103
# https://github.com/ray-project/ray/blob/161849364a784442cc659fb9780f1a6adee85fce/python/ray/_private/accelerators/npu.py#L94-L95
# https://github.com/ray-project/ray/blob/161849364a784442cc659fb9780f1a6adee85fce/python/ray/_private/accelerators/hpu.py#L116-L117
# https://github.com/ray-project/ray/blob/161849364a784442cc659fb9780f1a6adee85fce/python/ray/_private/accelerators/neuron.py#L108-L109
# https://github.com/ray-project/ray/blob/161849364a784442cc659fb9780f1a6adee85fce/python/ray/_private/accelerators/tpu.py#L171-L172
# https://github.com/ray-project/ray/blob/161849364a784442cc659fb9780f1a6adee85fce/python/ray/_private/accelerators/intel_gpu.py#L97-L98
NOSET_VISIBLE_DEVICES_ENV_VARS_LIST = [
    "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES",
    "RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES",
    "RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES",
    "RAY_EXPERIMENTAL_NOSET_HABANA_VISIBLE_MODULES",
    "RAY_EXPERIMENTAL_NOSET_NEURON_RT_VISIBLE_CORES",
    "RAY_EXPERIMENTAL_NOSET_TPU_VISIBLE_CHIPS",
    "RAY_EXPERIMENTAL_NOSET_ONEAPI_DEVICE_SELECTOR",
]

RAY_DEFAULT_ENV_VARS = {
    # Ray's uvloop integration has caused intermittent async actor issues.
    "RAY_USE_UVLOOP": "0",
    # Let sglang's JIT loader (load_jit) find slime-hosted custom kernel .cuh
    # sources (e.g. glm5_router_gemm) and try-cache compile them, so they need
    # not be patched into the sglang source tree.
    "SGLANG_JIT_KERNEL_EXTRA_PATH": os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "backends",
        "sglang_utils",
        "jit_kernels",
    ),
}


def add_default_ray_env_vars(env_vars: dict[str, str] | None = None) -> dict[str, str]:
    return RAY_DEFAULT_ENV_VARS | (env_vars or {})


def memory_saver_loader_env() -> dict[str, str]:
    """Conda/pip stand-in for the official image's /usr/local/cuda/lib64.

    Official slime:latest puts system CUDA on LD_LIBRARY_PATH via image ENV.
    This stack keeps libcudart under site-packages/nvidia/*/lib. SGLang's
    configure_subprocess() only sets LD_PRELOAD; the preload .so still needs
    these dirs in the scheduler process or it will not load.
    """
    nvidia = (
        Path(sys.prefix)
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
        / "nvidia"
    )
    extras = [str(p) for p in (*nvidia.glob("*/lib"), *nvidia.glob("*/lib64"))]
    extras.append(str(Path(sys.prefix) / "lib"))
    merged: list[str] = []
    for part in (*os.environ.get("LD_LIBRARY_PATH", "").split(":"), *extras):
        if part and part not in merged:
            merged.append(part)
    return {"LD_LIBRARY_PATH": ":".join(merged)}


def _preload_dynlib() -> Path:
    import torch_memory_saver

    parent = Path(torch_memory_saver.__file__).resolve().parent.parent
    pkg = Path(torch_memory_saver.__file__).resolve().parent
    for candidate in (
        *(
            parent / name
            for name in (
                "torch_memory_saver_hook_mode_preload_cu13.abi3.so",
                "torch_memory_saver_hook_mode_preload_cu12.abi3.so",
                "torch_memory_saver_hook_mode_preload.abi3.so",
            )
        ),
        *parent.glob("torch_memory_saver_hook_mode_preload*.so"),
        *pkg.glob("torch_memory_saver_hook_mode_preload*.so"),
    ):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "Cannot find torch_memory_saver preload hook. "
        "Install zhuzilin/torch_memory_saver@8d30c59 (the slime:latest pin)."
    )


def memory_saver_preload_env(*, enable_from_start: bool = False) -> dict[str, str]:
    """LD_PRELOAD + loader path for Megatron train actors.

    Official slime sets this pair only when offload_train. Do not use it for
    SGLang engines: configure_subprocess() prepends LD_PRELOAD, and a second
    copy makes get_path_binary() see two entries and reject preload mode.
    """
    env = memory_saver_loader_env()
    env["LD_PRELOAD"] = str(_preload_dynlib())
    if enable_from_start:
        env["TMS_INIT_ENABLE"] = "1"
        env["TMS_INIT_ENABLE_CPU_BACKUP"] = "1"
    return env


# Nothing here pins torch_memory_saver.hook_mode, deliberately. The zhuzilin
# pin already defaults to "preload"; the only thing that ever moved it off
# that default was Megatron's megatron/core/inference/contexts/dynamic_context.py
# assigning "torch" at import time, and that is now patched out at build time --
# see docker/envs/slime-v0.3.1/patches/Megatron-LM/0001-tms-hook-mode.patch.
#
# It has to be fixed there and not here. hook_mode is a process-global set
# during an import, so whatever the parent does is both undone by that import
# and not inherited across the scheduler's spawn anyway.


def ray_noset_visible_devices(env_vars=os.environ):
    return any(env_vars.get(env_var) for env_var in NOSET_VISIBLE_DEVICES_ENV_VARS_LIST)


def get_physical_gpu_id():
    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    return str(props.uuid)


@ray.remote
class Lock(RayActor):
    def __init__(self):
        self._locked = False  # False: unlocked, True: locked

    def acquire(self):
        """
        Try to acquire the lock. Returns True if acquired, False otherwise.
        Caller should retry until it returns True.
        """
        if not self._locked:
            self._locked = True
            return True
        return False

    def release(self):
        """Release the lock, allowing others to acquire."""
        assert self._locked, "Lock is not acquired, cannot release."
        self._locked = False
