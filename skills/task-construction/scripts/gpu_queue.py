#!/usr/bin/env python3
"""Persistent, one-shot GPU availability queue. No training is launched or GPU reserved."""

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import time
import uuid


def iso(epoch):
    return datetime.fromtimestamp(epoch, timezone.utc).isoformat()


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(path, value):
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def command(config, query):
    argv = [config["nvidia_smi"], query, "--format=csv,noheader,nounits"]
    if config["ssh_target"]:
        argv = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "--", config["ssh_target"], shlex.join(argv)]
    completed = subprocess.run(argv, capture_output=True, text=True, encoding="utf-8", timeout=30)
    if completed.returncode:
        raise RuntimeError(completed.stderr.strip() or completed.stdout.strip() or "GPU query failed")
    return completed.stdout


def parse_inventory(gpu_text, process_text):
    processes = set()
    for row in csv.reader(process_text.splitlines()):
        if not row or not any(part.strip() for part in row):
            continue
        gpu_uuid, _pid = [part.strip() for part in row]
        if not gpu_uuid.startswith("GPU-"):
            raise ValueError("Cannot establish compute-process ownership from nvidia-smi output")
        processes.add(gpu_uuid)
    gpus = []
    for row in csv.reader(gpu_text.splitlines()):
        if not row or not any(part.strip() for part in row):
            continue
        index, gpu_uuid, name, total, used, utilization = [part.strip() for part in row]
        total, used, utilization = int(total), int(used), int(utilization)
        gpus.append({
            "index": index, "uuid": gpu_uuid, "name": name, "total_mib": total,
            "used_mib": used, "utilization_percent": utilization,
            "compute_process_present": gpu_uuid in processes,
        })
    return gpus


def probe(config):
    gpu_text = command(config, "--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu")
    process_text = command(config, "--query-compute-apps=gpu_uuid,pid")
    return parse_inventory(gpu_text, process_text)


def select_gpus(gpus, config):
    allowed = config["gpu_ids"]
    matches = [gpu for gpu in gpus if (
        (not allowed or gpu["index"] in allowed or gpu["uuid"] in allowed)
        and (not config["gpu_name"] or config["gpu_name"].casefold() in gpu["name"].casefold())
        and gpu["used_mib"] <= config["max_used_mib"]
        and gpu["total_mib"] - gpu["used_mib"] >= config["min_free_mib"]
        and gpu["utilization_percent"] <= config["max_utilization"]
        and not gpu["compute_process_present"]
    )]
    return sorted(matches, key=lambda gpu: (gpu["used_mib"], gpu["index"]))[:config["gpu_count"]]


def initialize(args, now):
    if args.gpu_count < 1 or args.interval_minutes < 1:
        raise ValueError("GPU count and check interval must be positive")
    if not 0 <= args.max_utilization <= 100 or args.max_used_mib < 0 or args.min_free_mib < 0:
        raise ValueError("Invalid GPU availability thresholds")
    config = {name: getattr(args, name) for name in (
        "gpu_count", "ssh_target", "nvidia_smi", "gpu_name", "max_used_mib", "min_free_mib",
        "max_utilization", "interval_minutes",
    )}
    config["gpu_ids"] = [value.strip() for value in args.gpu_ids.split(",") if value.strip()]
    return {
        "queue_id": uuid.uuid4().hex, "status": "waiting", "config": config,
        "created_at": iso(now),
        "next_check_epoch": now, "next_check_at": iso(now),
        "selected_gpus": [],
    }


def advance(state, now, claim=False):
    try:
        gpus = probe(state["config"])
        selected = select_gpus(gpus, state["config"])
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as error:
        state.update(status="error", selected_gpus=[], last_error=str(error))
    else:
        state.update(last_inventory=gpus, last_error=None)
        if len(selected) == state["config"]["gpu_count"]:
            state.update(status="ready", selected_gpus=selected)
            if claim:
                state.update(status="claimed", claim_id=uuid.uuid4().hex, claimed_at=iso(now))
        else:
            state.update(status="waiting", selected_gpus=[])
    due = now + state["config"]["interval_minutes"] * 60
    state.update(last_checked_at=iso(now), next_check_epoch=due, next_check_at=iso(due))
    return state


def check_state(path, claim=False):
    state = load(path)
    now = time.time()
    if state["status"] in ("claimed", "cancelled"):
        return state, False
    if claim and state["status"] != "ready":
        raise ValueError("Only a ready queue can be claimed")
    state = advance(state, now, claim)
    save(path, state)
    return state, True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init", help="Create a persistent queue without probing or allocating devices")
    for sub in (init, commands.add_parser("check"), commands.add_parser("claim"), commands.add_parser("cancel")):
        sub.add_argument("--state", type=Path, required=True)
    init.add_argument("--gpu-count", type=int, required=True)
    init.add_argument("--ssh-target", default=None)
    init.add_argument("--nvidia-smi", default="nvidia-smi")
    init.add_argument("--gpu-name", default="")
    init.add_argument("--gpu-ids", default="")
    init.add_argument("--max-used-mib", type=int, default=5120)
    init.add_argument("--min-free-mib", type=int, default=0)
    init.add_argument("--max-utilization", type=int, default=10)
    init.add_argument("--interval-minutes", type=int, default=15)
    args = parser.parse_args(argv)
    try:
        path = args.state.expanduser().resolve()
        probed = False
        if args.command in ("check", "claim"):
            state, probed = check_state(path, claim=args.command == "claim")
        else:
            if args.command == "init":
                path.parent.mkdir(parents=True, exist_ok=True)
            now = time.time()
            if args.command == "init":
                if path.exists():
                    raise FileExistsError("Queue state already exists; resume it instead of overwriting")
                state = initialize(args, now)
            else:
                state = load(path)
                state.update(status="cancelled", cancelled_at=iso(now), selected_gpus=[])
            save(path, state)
        print(json.dumps({"state_file": str(path), "probed": probed, **state}, ensure_ascii=False, indent=2))
        return 2 if state["status"] == "error" else 0
    except (OSError, ValueError) as error:
        print(json.dumps({"status": "error", "error": str(error)}, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    sys.exit(main())
