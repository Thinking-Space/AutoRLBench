# Copyright 2026 QAT contributors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
"""File-based checkpoint requests consumed by the trainer at a step boundary.

This module uses only the standard library so the control command runs without
initializing Torch, CUDA, or Ray. Acknowledgment follows the full trainer save,
including dataloader state and the latest-checkpoint pointer.
"""

import argparse
import json
import os
import socket
import time
import uuid
from pathlib import Path


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with tmp.open("x") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


class SaveNow:
    def __init__(self, checkpoint_dir):
        self.root = Path(checkpoint_dir).resolve()
        self.control = self.root / ".save_now"

    def announce(self):
        self.control.mkdir(parents=True, exist_ok=True)
        atomic_json(self.control / "handler.json", {
            "protocol": 1, "host": socket.gethostname(), "pid": os.getpid(),
            "job_id": os.environ.get("SLURM_JOB_ID"), "started_at": time.time(),
            "checkpoint_dir": str(self.root), "boundary": "after_update_before_validation",
        })

    def pending(self):
        if not self.control.exists():
            return []
        requests = []
        for path in sorted(self.control.glob("request-*.json")):
            if not (self.control / path.name.replace("request-", "done-", 1)).exists():
                requests.append(path)
        return requests

    def complete(self, requests, step):
        # Each request has a unique name. A new request arriving during the save
        # remains pending for the next boundary; no shared flag is cleared.
        for path in requests:
            atomic_json(self.control / path.name.replace("request-", "done-", 1), {
                "request_id": path.stem.removeprefix("request-"), "status": "saved",
                "step": int(step), "completed_at": time.time(),
                "checkpoint": str(self.root / f"global_step_{step}"),
            })

    def save_if_needed(self, step, scheduled, save):
        requests = self.pending()
        if not scheduled and not requests:
            return False
        # Exceptions propagate: a partial checkpoint must never receive an ACK.
        save()
        self.complete(requests, step)
        if requests:
            print(f"[save-now] completed step={step} requests={len(requests)}", flush=True)
        return True

    def request(self):
        if not (self.control / "handler.json").is_file():
            raise RuntimeError(
                "This run has not registered a save_now handler; "
                "an already loaded old trainer needs separate handling."
            )
        request_id = uuid.uuid4().hex
        atomic_json(self.control / f"request-{request_id}.json", {
            "request_id": request_id, "requested_at": time.time(),
        })
        return request_id


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint_dir", help="trainer.default_local_dir (usually RUN/checkpoints)")
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()
    control = SaveNow(args.checkpoint_dir)
    if args.status:
        handler = control.control / "handler.json"
        print(json.dumps({
            "handler": json.loads(handler.read_text()) if handler.exists() else None,
            "pending": [p.stem.removeprefix("request-") for p in control.pending()],
            "completed": [json.loads(p.read_text()) for p in sorted(control.control.glob("done-*.json"))],
        }, indent=2))
    else:
        request_id = control.request()
        print(json.dumps({"status": "queued", "request_id": request_id,
                          "ack": str(control.control / f"done-{request_id}.json")}, indent=2))


if __name__ == "__main__":
    main()
