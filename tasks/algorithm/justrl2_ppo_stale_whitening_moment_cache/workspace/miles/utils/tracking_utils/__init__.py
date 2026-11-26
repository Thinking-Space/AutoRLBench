import json
import os
from pathlib import Path


def _append_metrics_jsonl(metrics, step, step_key):
    path_str = os.environ.get("MILES_METRICS_FILE")
    if not path_str:
        return
    clean = {}
    for k, v in metrics.items():
        if isinstance(v, (int, float, bool, str)) or v is None:
            clean[k] = v
        elif hasattr(v, "item"):
            try:
                clean[k] = v.item()
            except Exception:
                pass
    p = Path(path_str)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"step": step, "step_key": step_key, "data": clean}, ensure_ascii=False) + "\n")

import logging

from .base import TrackingManager

logger = logging.getLogger(__name__)
_manager = TrackingManager()


def init_tracking(args, primary: bool = True, **kwargs):
    _manager.init(args, primary=primary, **kwargs)


def log(args, metrics, step_key: str):
    step = metrics.get(step_key)
    _append_metrics_jsonl(metrics, step, step_key)
    _manager.log(metrics, step=step, step_key=step_key)


def log_histogram(args, tag: str, values, step: int):
    _manager.log_histogram(tag, values, step=step)


def finish_tracking():
    _manager.finish()
