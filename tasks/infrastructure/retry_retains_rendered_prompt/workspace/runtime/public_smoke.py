"""Public data, deterministic retry scheduling, and observational optimizer hook.

No optimizer, training loop, advantage, loss, or Sample reset is implemented
here. Those execute in the vendored Miles Ray/Megatron framework.
"""
import copy
import hashlib
import inspect
import json
import os
import time
from collections import deque
from pathlib import Path

from miles.rollout.data_source import DataSource
from miles.rollout.filter_hub.base_types import DynamicFilterOutput
from miles.utils.types import Sample

MESSAGES = [{"role": "user", "content": "Write a short sequence of red blue green tokens ."}]


def evidence_dir():
    return Path(os.environ["RETRY_SMOKE_LOG_DIR"])


def atomic_json(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


class PublicDataSource(DataSource):
    def __init__(self, args):
        self.args = args
        self._buffer = deque([self.new_group(0)])
        self.last_accepted = None
        self.recycled = 0

    def new_group(self, group_index):
        # label is a frozen public task and can reconstruct native messages.
        label = json.dumps({"messages": MESSAGES}, sort_keys=True)
        return [Sample(group_index=group_index, index=group_index * self.args.n_samples_per_prompt + i,
                       prompt=copy.deepcopy(MESSAGES), label=label, metadata={"public_task": "colors"})
                for i in range(self.args.n_samples_per_prompt)]

    def get_samples(self, num_samples):
        return [self._buffer.popleft() for _ in range(min(num_samples, len(self._buffer)))]

    def add_samples(self, samples):
        self.recycled += len(samples)
        self._buffer.extend(samples)

    def save(self, rollout_id):
        pass

    def load(self, rollout_id=None):
        pass

    def get_buffer_length(self):
        return len(self._buffer)


def trigger_filter(args, group):
    # Explicit scheduling perturbation, not a claim about model solve rate.
    # Reuse the historical allowlisted reason to enter the ORIGINAL clone path.
    sample = group[0]
    if sample.group_index == 1 and sample.metadata.get("dynamic_sampling_retry", 0) == 0:
        return DynamicFilterOutput(keep=False, reason="solve_none")
    return DynamicFilterOutput(keep=True)


def _wait_for_update(rollout_id):
    path = evidence_dir() / f"optimizer_step.{rollout_id}.0.json"
    deadline = time.monotonic() + 600
    while not path.exists():
        if time.monotonic() > deadline:
            raise TimeoutError(f"Native Miles optimizer update {rollout_id} was not observed")
        time.sleep(0.1)
    record = json.loads(path.read_text())
    assert record["update_successful"] and record["parameter_delta_l2"] > 0
    assert record["grad_norm"] > 0


def rollout(args, rollout_id, data_source, evaluation=False):
    from . import fully_async
    if rollout_id > 0:
        _wait_for_update(rollout_id - 1)
        worker = fully_async.get_global_worker(args, data_source)
        if rollout_id == 1:
            with worker._data_source_lock:
                data_source._buffer.append(data_source.new_group(1))
        elif rollout_id == 2:
            # Independent direct recycle probe after a successfully repaired retry.
            worker.recycle_source_group(data_source.last_accepted, reason="public_lifecycle_probe")
    result = fully_async.generate_rollout_fully_async(args, rollout_id, data_source, evaluation=evaluation)
    rewards = [s.reward for g in result.samples for s in g]
    assert max(rewards) > min(rewards), "Inconclusive smoke: generated rewards have zero variance"
    data_source.last_accepted = result.samples[0]
    atomic_json(evidence_dir() / f"rollout.{rollout_id}.json", {
        "rollout_id": rollout_id, "samples": len(rewards), "reward_min": min(rewards),
        "reward_max": max(rewards), "group_index": result.samples[0][0].group_index,
        "dynamic_sampling_retry": result.samples[0][0].metadata.get("dynamic_sampling_retry", 0),
        "source_recycles": data_source.recycled,
    })
    return result


# Let Miles discover all original rollout-specific CLI arguments.
from .fully_async import generate_rollout_fully_async
rollout.add_arguments = generate_rollout_fully_async.add_arguments


def observe_train_step(args, rollout_id, step_id, model, optimizer, opt_param_scheduler):
    """Wrap the real optimizer step for evidence; leave its implementation intact."""
    import torch
    from miles.backends.megatron_utils import model as native_trainer
    assert optimizer is not None, "The smoke requires the native Megatron optimizer"
    optimizer._public_step_identity = (rollout_id, step_id)
    if hasattr(optimizer, "_public_original_step"):
        return
    original_step = optimizer.step
    workspace = Path(__file__).resolve().parents[1]
    trainer_path = (workspace / "vendor/miles/backends/megatron_utils/model.py").resolve()
    native_train_code = inspect.unwrap(native_trainer.train_one_step).__code__
    assert Path(native_train_code.co_filename).resolve() == trainer_path, "Unexpected Miles trainer source"
    optimizer_class_path = Path(inspect.getsourcefile(type(optimizer))).resolve()
    optimizer_method_path = Path(inspect.getsourcefile(inspect.unwrap(original_step))).resolve()
    allowed_roots = [(workspace / "vendor/megatron").resolve(), (workspace / "vendor/miles").resolve()]
    assert all(any(path.is_relative_to(root) for root in allowed_roots)
               for path in [optimizer_class_path, optimizer_method_path]), "Unexpected optimizer source"
    def trainer_identity():
        frame = inspect.currentframe().f_back
        while frame is not None:
            if frame.f_code is native_train_code:
                return {
                    "symbol": frame.f_globals["__name__"] + "." + frame.f_code.co_name,
                    "file": str(trainer_path.relative_to(workspace)),
                    "sha256": hashlib.sha256(trainer_path.read_bytes()).hexdigest(),
                }
            frame = frame.f_back
        raise AssertionError("The observation hook was not called by the bundled Miles training function")
    trainer_identity()
    optimizer._public_original_step = original_step

    def observed_step(*a, **kw):
        parameters = [p for chunk in model for p in chunk.parameters() if p.requires_grad]
        before = [p.detach().float().clone() for p in parameters]
        result = original_step(*a, **kw)
        torch.cuda.synchronize()
        success, grad_norm, zeros = result
        delta_sq = sum((p.detach().float() - old).square().sum().item()
                       for p, old in zip(parameters, before, strict=True))
        record = {
            "trainer_identity": trainer_identity(),
            "optimizer_class": type(optimizer).__module__ + "." + type(optimizer).__name__,
            "optimizer_method": original_step.__func__.__qualname__,
            "optimizer_class_source": str(optimizer_class_path.relative_to(workspace)),
            "optimizer_method_source": str(optimizer_method_path.relative_to(workspace)),
            "optimizer_method_sha256": hashlib.sha256(optimizer_method_path.read_bytes()).hexdigest(),
            "update_successful": bool(success), "grad_norm": float(grad_norm),
            "parameter_delta_l2": delta_sq ** 0.5,
            "parameter_count": sum(p.numel() for p in parameters),
        }
        assert success and delta_sq > 0 and float(grad_norm) > 0, record
        # The hook is called before each step; capture its current rollout below.
        current_rollout, current_step = optimizer._public_step_identity
        atomic_json(evidence_dir() / f"optimizer_step.{current_rollout}.{current_step}.json", record)
        print("NATIVE_MILES_OPTIMIZER_EVIDENCE " + json.dumps(record), flush=True)
        return result

    optimizer.step = observed_step
    optimizer._public_step_identity = (rollout_id, step_id)
