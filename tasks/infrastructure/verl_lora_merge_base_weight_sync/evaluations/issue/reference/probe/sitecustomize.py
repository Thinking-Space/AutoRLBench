"""Diagnostic probes for the verl #7014 experiment. Only active when ARB_PROBE_DIR is set; it is put on PYTHONPATH
for dedicated probe runs, so the two arms' code is unchanged.

Hooks are installed lazily, when their module is first imported (in any process: driver, Ray workers, vLLM workers):
  actor side    ActorRolloutRefWorker.update_weights: at the start of every sync, for a few LoRA-target layers,
                save W_base and dW = scaling * B @ A read from the live PEFT parameters (what should reach vLLM).
  consumer side vLLMColocateWorkerExtension._update_weights: save the tensors vLLM actually receives for them.
"""
import functools
import importlib.abc
import importlib.util
import os
import sys

PROBE_DIR = os.environ.get("ARB_PROBE_DIR")
TARGETS = (
    "model.layers.0.self_attn.q_proj",
    "model.layers.14.mlp.down_proj",
    "model.layers.27.self_attn.v_proj",
)


def _log(msg):
    with open(os.path.join(PROBE_DIR, "probe.log"), "a") as f:
        f.write(f"[pid {os.getpid()}] {msg}\n")


def _record_actor(worker):
    import torch
    import torch.distributed as dist
    from torch.distributed.tensor import DTensor

    worker._arb_probe_k = getattr(worker, "_arb_probe_k", -1) + 1
    k = worker._arb_probe_k
    module = worker.actor.engine.module
    params = dict(module.named_parameters())
    modules = dict(module.named_modules())
    rank0 = not dist.is_initialized() or dist.get_rank() == 0
    for target in TARGETS:
        base = next(n for n in params if n.endswith(f"{target}.base_layer.weight"))
        prefix = base[: -len(".base_layer.weight")]

        def full(name):
            p = params[name].detach()
            return (p.full_tensor() if isinstance(p, DTensor) else p).float()

        w_base = full(base)
        lora_a = full(f"{prefix}.lora_A.default.weight")
        lora_b = full(f"{prefix}.lora_B.default.weight")
        scaling = modules[prefix].scaling["default"]
        delta = scaling * (lora_b @ lora_a)
        if rank0:
            torch.save(
                {"k": k, "name": target, "scaling": scaling, "w_base": w_base.cpu(), "delta": delta.cpu()},
                os.path.join(PROBE_DIR, f"actor_k{k:03d}_{target}.pt"),
            )
    if rank0:
        _log(f"actor sync k={k}: recorded {len(TARGETS)} layers")


def _patch_actor(mod):
    cls = mod.ActorRolloutRefWorker
    orig = cls.update_weights

    @functools.wraps(orig)
    async def update_weights(self, *args, **kwargs):
        try:
            _record_actor(self)
        except Exception as e:  # never break the run being observed
            _log(f"actor probe failed: {e!r}")
        return await orig(self, *args, **kwargs)

    cls.update_weights = update_weights
    _log("patched ActorRolloutRefWorker.update_weights")


def _patch_consumer(mod):
    import torch

    cls = mod.vLLMColocateWorkerExtension
    orig_ipc, orig_update = cls.update_weights_from_ipc, cls._update_weights
    names = {f"{t}.weight" for t in TARGETS}

    @functools.wraps(orig_ipc)
    def update_weights_from_ipc(self, *args, **kwargs):
        self._arb_probe_k = getattr(self, "_arb_probe_k", -1) + 1
        return orig_ipc(self, *args, **kwargs)

    @functools.wraps(orig_update)
    def _update_weights(self, weights, *args, **kwargs):
        try:
            k = getattr(self, "_arb_probe_k", 0)
            for name, tensor in weights:
                if name in names:
                    torch.save(
                        {"k": k, "name": name[: -len(".weight")], "recv": tensor.detach().to("cpu", copy=True)},
                        os.path.join(PROBE_DIR, f"recv_k{k:03d}_pid{os.getpid()}_{name[: -len('.weight')]}.pt"),
                    )
        except Exception as e:
            _log(f"consumer probe failed: {e!r}")
        return orig_update(self, weights, *args, **kwargs)

    cls.update_weights_from_ipc = update_weights_from_ipc
    cls._update_weights = _update_weights
    _log("patched vLLMColocateWorkerExtension")


PATCHES = {
    "verl.workers.engine_workers": _patch_actor,
    "verl.workers.rollout.vllm_rollout.utils": _patch_consumer,
}


class _PatchOnImport(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name not in PATCHES:
            return None
        sys.meta_path.remove(self)
        try:
            spec = importlib.util.find_spec(name)
        finally:
            sys.meta_path.insert(0, self)
        if spec is None or spec.loader is None:
            return spec
        exec_module = spec.loader.exec_module

        def exec_and_patch(module):
            exec_module(module)
            PATCHES[name](module)

        spec.loader.exec_module = exec_and_patch
        return spec


if PROBE_DIR:
    os.makedirs(PROBE_DIR, exist_ok=True)
    sys.meta_path.insert(0, _PatchOnImport())
