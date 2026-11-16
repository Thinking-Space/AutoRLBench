"""Optimizer-step evidence hook (``--custom-megatron-before-train-step-hook-path``).

Wraps ``optimizer.step`` once so every update made by the bundled Megatron
trainer is recorded with its identity, the returned grad norm and the L2 change
of a fixed sample of parameter tensors.  Cloning every parameter of a 35B MoE
on each rank would cost tens of GB, so only a deterministic subset is copied:
every 16th trainable tensor of at most 64M elements, copied to host memory.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import os
import time
from pathlib import Path


def _write(record: dict, rollout_id: int, step_id: int) -> None:
    root = os.environ.get("RUN_EVIDENCE_DIR")
    if root:
        import torch.distributed as dist

        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        path = Path(root) / "optimizer" / f"step.{rollout_id}.{step_id}.rank{rank}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print("OPTIMIZER_STEP_EVIDENCE " + json.dumps(record, sort_keys=True), flush=True)


def observe_train_step(args, rollout_id, step_id, model, optimizer, opt_param_scheduler):
    import torch
    from miles.backends.megatron_utils import model as trainer_module

    optimizer._evidence_identity = (rollout_id, step_id)
    if hasattr(optimizer, "_evidence_original_step"):
        return
    trainer_code = inspect.unwrap(trainer_module.train_one_step).__code__
    trainer_file = Path(trainer_code.co_filename).resolve()
    trainer_sha = hashlib.sha256(trainer_file.read_bytes()).hexdigest()
    params = [p for chunk in model for p in chunk.parameters() if p.requires_grad]
    # snapshot a deterministic subset of tensors on the host: every 16th trainable tensor
    # that is at most 64M elements, so the hook adds no device memory to the optimizer step
    picked = [i for i in range(0, len(params), 16) if params[i].numel() <= 1 << 26]
    original = optimizer.step
    optimizer._evidence_original_step = original

    def called_from_trainer() -> bool:
        frame = inspect.currentframe().f_back
        while frame is not None:
            if frame.f_code is trainer_code:
                return True
            frame = frame.f_back
        return False

    def observed_step(*a, **kw):
        before = [params[i].detach().to("cpu", copy=True) for i in picked]
        started = time.time()
        result = original(*a, **kw)
        torch.cuda.synchronize()
        success, grad_norm, num_zeros = result
        delta = sum((params[i].detach().cpu().float() - old.float()).square().sum().item()
                    for i, old in zip(picked, before)) ** 0.5
        r, s = optimizer._evidence_identity
        _write({
            "rollout_id": r, "step_id": s, "update_successful": bool(success),
            "grad_norm": float(grad_norm) if grad_norm is not None else None,
            "num_zeros_in_grad": None if num_zeros is None else float(num_zeros),
            "sampled_parameter_delta_l2": delta, "sampled_tensors": len(picked),
            "trainable_tensors": len(params), "local_trainable_parameters": sum(p.numel() for p in params),
            "optimizer_class": f"{type(optimizer).__module__}.{type(optimizer).__qualname__}",
            "called_from": f"{trainer_module.__name__}.{trainer_code.co_name}", "called_from_trainer": called_from_trainer(),
            "trainer_source_sha256": trainer_sha, "step_wall_s": round(time.time() - started, 3), "t": time.time(),
        }, r, s)
        return result

    optimizer.step = observed_step
