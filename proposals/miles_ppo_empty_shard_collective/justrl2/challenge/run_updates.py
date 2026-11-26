#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run bounded native CPU PPO replay updates using Gloo."""
from __future__ import annotations
import argparse
from datetime import timedelta
import json
import math
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import tempfile
import time
import torch
from types import SimpleNamespace
import runtime_support as runtime


def default_config():
    return dict(cp=2, dp=2, order=[0, 1, 2, 3], updates=5, copies=2, seed=3,
                assignment="shuffle", reporting=True, compute_timeout=4.0,
                report_timeout=1.0, outer_timeout=45.0)


def validate_config(config):
    defaults = default_config()
    if set(config) != set(defaults):
        raise ValueError("Invalid configuration")
    cp, dp, order = config["cp"], config["dp"], config["order"]
    if type(cp) is not int or cp not in (1, 2, 4) or type(dp) is not int or dp != 2:
        raise ValueError("Unsupported parallel layout")
    if cp * dp > 12 or sorted(order) != list(range(cp * dp)):
        raise ValueError("Invalid rank order")
    rows = [order[d * cp:(d + 1) * cp] for d in range(dp)]
    columns = [order[c::cp] for c in range(cp)]
    if any(group != sorted(group) for group in rows + columns):
        raise ValueError("Invalid rank order")
    if type(config["updates"]) is not int or not 1 <= config["updates"] <= 12:
        raise ValueError("Invalid update count")
    if config["copies"] != 2 or config["assignment"] not in ("shuffle", "interleave"):
        raise ValueError("Invalid configuration")
    if type(config["reporting"]) is not bool or type(config["seed"]) is not int:
        raise ValueError("Invalid configuration")
    for key in ("compute_timeout", "report_timeout", "outer_timeout"):
        value = config[key]
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= 45:
            raise ValueError("Invalid timeout")
    return config


def _order(config, update):
    ids = list(range(4))
    if config["assignment"] == "shuffle":
        random.Random(config["seed"] + update).shuffle(ids)
    else:
        ids = [0, 2, 1, 3]
    return ids


def _batch_pool(source):
    samples = json.loads((source / "inputs/replay.json").read_text())["samples"]
    if len(samples) != 2:
        raise ValueError("Invalid input")
    pool = []
    for row in samples:
        length = row["response_length"]
        total = len(row["tokens"])
        if type(length) is not int or not 0 < length < total or len(row["loss_mask"]) != length:
            raise ValueError("Invalid input")
        pool.append(dict(total=total, length=length, masks=row["loss_mask"],
                         reward=float(row["reward"]), values=[0.52] * length))
    return [pool[0], pool[0], pool[1], pool[1]]


def _metric_part(rows):
    return sum(float((values * mask).sum() / full_mask.sum().clamp_min(1))
               for values, mask, full_mask in rows)


def _configure_native():
    native = runtime.NATIVE_FUNCTIONS
    native['miles/backends/training_utils/log_utils.py'] = ('aggregate_train_losses',)
    key = 'miles/backends/training_utils/cp_utils.py'
    if 'get_local_response_loss_masks' not in native[key]:
        native[key] += ('get_local_response_loss_masks',)


def _worker(source, output, scratch, config, rank):
    import torch.distributed as dist
    import torch.nn.functional as F
    code, completed = 2, 0
    first_error = None
    try:
        torch.set_num_threads(1)
        _configure_native()
        world, order = config["cp"] * config["dp"], config["order"]
        dp_rank, cp_rank = divmod(order.index(rank), config["cp"])
        dist.init_process_group("gloo", init_method=(scratch / "store").as_uri(),
                                world_size=world, rank=rank, timeout=timedelta(seconds=15))
        control = dist.new_group(list(range(world)), timeout=timedelta(seconds=20))
        cp_group = dp_group = None
        for d in range(config["dp"]):
            members = order[d * config["cp"]:(d + 1) * config["cp"]]
            group = dist.new_group(members, timeout=timedelta(seconds=config["compute_timeout"]))
            if rank in members:
                cp_group = group
        for c in range(config["cp"]):
            members = order[c::config["cp"]]
            group = dist.new_group(members, timeout=timedelta(seconds=config["compute_timeout"]))
            if rank in members:
                dp_group = group
        reporting = dist.new_group(list(range(world)), timeout=timedelta(seconds=config["report_timeout"]))
        state = SimpleNamespace(cp=SimpleNamespace(rank=cp_rank, size=config["cp"], group=cp_group),
            intra_dp=SimpleNamespace(rank=dp_rank, size=config["dp"], group=dp_group),
            intra_dp_cp=SimpleNamespace(group=reporting, size=world, rank=rank))
        namespace = dict(torch=torch, dist=dist, F=F, get_parallel_state=lambda: state,
            Namespace=SimpleNamespace, RolloutBatch=dict,
            logger=__import__("logging").getLogger("replay"),
            tracking_utils=SimpleNamespace(log=lambda *a, **k: None),
            compute_rollout_step=lambda args, update: update)
        runtime._load_native(source, namespace)
        pool = _batch_pool(source)
        settings = SimpleNamespace(use_rollout_logprobs=False, kl_coef=0, advantage_estimator="ppo",
            gamma=1.0, lambd=1.0, gae_lambda_k=0.5, normalize_advantages=True, qkv_format="thd")
        code = 0
        for update in range(config["updates"]):
            selected = _order(config, update)[dp_rank * 2:dp_rank * 2 + 2]
            samples = [pool[i] for i in selected]
            failed = 0
            operation = "compute_advantages_and_returns"
            try:
                values = [namespace["slice_log_prob_with_cp"](
                    torch.tensor(x["values"]), x["total"], x["length"]) for x in samples]
                batch = dict(values=values, log_probs=[torch.zeros_like(x) for x in values],
                    rewards=[x["reward"] for x in samples], total_lengths=[x["total"] for x in samples],
                    response_lengths=[x["length"] for x in samples],
                    loss_masks=[torch.tensor(x["masks"], dtype=torch.float32) for x in samples])
                namespace["compute_advantages_and_returns"](settings, batch)
                if config["reporting"]:
                    operation = "prepare_metrics"
                    masks = namespace["get_local_response_loss_masks"](
                        batch["total_lengths"], batch["response_lengths"], batch["loss_masks"], settings.qkv_format)
                    adv_rows = zip([x.square() for x in batch["advantages"]], masks,
                                   batch["loss_masks"], strict=True)
                    ret_rows = zip([x.square() for x in batch["returns"]], masks,
                                   batch["loss_masks"], strict=True)
                    vector = torch.tensor([len(samples), _metric_part(adv_rows), _metric_part(ret_rows)])
                    operation = "aggregate_train_losses"
                    namespace["aggregate_train_losses"]([dict(
                        keys=["adv_sample_mean_square", "return_sample_mean_square"], values=vector)])
            except Exception as exc:
                first_error = dict(update=update, operation=operation, type=type(exc).__name__,
                                   message=str(exc), observed_at_ns=time.monotonic_ns())
                failed = 1
            flag = torch.tensor([failed], dtype=torch.int32)
            dist.all_reduce(flag, op=dist.ReduceOp.MAX, group=control)
            if flag.item():
                code = 1
                break
            completed += 1
    except Exception:
        code = 2
    finally:
        if "dist" in locals() and dist.is_initialized():
            try:
                dist.destroy_process_group()
            except Exception:
                code = 2
        (scratch / f"worker-{rank}.json").write_text(
            json.dumps(dict(code=code, completed=completed, first_error=first_error)))
    return code


def run_updates(source, output, config=None):
    config = validate_config(dict(default_config() if config is None else config))
    source, output = Path(source).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    result = dict(status="infrastructure_error", exit_code=2,
                  completed_updates=0, outer_timeout=False, first_error=None)
    processes, streams = [], []
    try:
        _configure_native()
        _, before = runtime._source_functions(source)
        with tempfile.TemporaryDirectory(prefix=".updates-", dir=output) as temp:
            scratch = Path(temp)
            (scratch / "config.json").write_text(json.dumps(config))
            deadline = time.monotonic() + config["outer_timeout"]
            try:
                for rank in range(config["cp"] * config["dp"]):
                    stream = (output / f"rank{rank}.stdout.log").open("w")
                    streams.append(stream)
                    processes.append(subprocess.Popen(
                        [sys.executable, "-B", str(Path(__file__).resolve()),
                         "--source", str(source), "--output", str(output),
                         "--config", str(scratch / "config.json"),
                         "--_worker", str(rank), "--_scratch", str(scratch)],
                        stdout=stream, stderr=subprocess.STDOUT,
                        env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1")))
                while any(p.poll() is None for p in processes):
                    if time.monotonic() > deadline:
                        result["outer_timeout"] = True
                        break
                    time.sleep(0.02)
            finally:
                for process in processes:
                    if process.poll() is None:
                        process.terminate()
                for process in processes:
                    try:
                        process.wait(timeout=1)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                for stream in streams:
                    stream.close()
            statuses = [json.loads((scratch / f'worker-{rank}.json').read_text())
                        for rank in range(config['cp'] * config['dp'])]
            _, after = runtime._source_functions(source)
            if (not result["outer_timeout"] and len(statuses) == config["cp"] * config["dp"]
                    and all(x["code"] in (0, 1) for x in statuses)
                    and len(processes) == len(statuses)
                    and all(p.returncode == s['code'] for p, s in zip(processes, statuses))
                    and before == after):
                failed = any(x["code"] == 1 for x in statuses)
                result.update(status="failed" if failed else "completed",
                    exit_code=int(failed), completed_updates=min(x["completed"] for x in statuses))
                errors = [x["first_error"] for x in statuses if x.get("first_error") is not None]
                if errors:
                    first = dict(min(errors, key=lambda e: e["observed_at_ns"]))
                    first.pop("observed_at_ns")
                    first['message'] = re.sub(r'^\[[^\]]*gloo[^\]]*\]', '[gloo transport]', first['message'])
                    result['first_error'] = first
    except Exception:
        pass
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--_worker", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--_scratch", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    config = json.loads(args.config.read_text()) if args.config else default_config()
    if args._worker is not None:
        raise SystemExit(_worker(args.source, args.output, args._scratch, config, args._worker))
    result = run_updates(args.source, args.output, config)
    print(json.dumps(result))
    raise SystemExit(result["exit_code"])
