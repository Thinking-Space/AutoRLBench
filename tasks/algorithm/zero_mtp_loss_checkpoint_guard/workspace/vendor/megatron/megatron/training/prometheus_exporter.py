# Copyright (c) 2026, ModelBest Inc. All rights reserved.

"""Prometheus-compatible metrics exporter for Megatron-LM training."""

import glob as _glob
import json
import os
import threading
import time as _time
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Dict, List, Optional, Tuple

_METRICS_DIR = os.path.join(os.environ.get("TMPDIR", "/tmp"), "megatron_prom")


class PrometheusMetricsStore:
    """Thread-safe store for Prometheus gauge metrics."""

    def __init__(self):
        self._gauges: Dict[Tuple, float] = {}
        self._meta: Dict[str, Tuple[str, str]] = {}
        self._lock = threading.Lock()

    def set_gauge(
        self,
        name: str,
        value: float,
        help_text: str = "",
        labels: Optional[Dict[str, str]] = None,
    ):
        with self._lock:
            label_key = tuple(sorted(labels.items())) if labels else ()
            self._gauges[(name, label_key)] = value
            if name not in self._meta:
                self._meta[name] = (help_text, "gauge")

    def snapshot(self) -> dict:
        with self._lock:
            gauges = [
                {
                    "name": name,
                    "labels": dict(label_tuple) if label_tuple else {},
                    "value": value,
                }
                for (name, label_tuple), value in self._gauges.items()
            ]
            meta = {
                name: {"help": help_text, "type": metric_type}
                for name, (help_text, metric_type) in self._meta.items()
            }
        return {"gauges": gauges, "meta": meta}


class _MetricsHandler(BaseHTTPRequestHandler):
    _metrics_dir: str = _METRICS_DIR

    def do_GET(self):
        if self.path == "/metrics":
            body = self._collect_all_metrics().encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()

    def _collect_all_metrics(self) -> str:
        all_gauges: Dict[Tuple, float] = {}
        all_meta: Dict[str, Tuple[str, str]] = {}

        for filepath in sorted(_glob.glob(os.path.join(self._metrics_dir, "rank_*.json"))):
            try:
                with open(filepath, "r") as f:
                    data = json.load(f)
                for entry in data.get("gauges", []):
                    labels = tuple(sorted(entry["labels"].items())) if entry.get("labels") else ()
                    all_gauges[(entry["name"], labels)] = entry["value"]
                for name, meta in data.get("meta", {}).items():
                    if name not in all_meta:
                        all_meta[name] = (meta["help"], meta["type"])
            except (FileNotFoundError, json.JSONDecodeError, KeyError):
                continue

        lines: List[str] = []
        emitted: set = set()
        for (name, label_tuple), value in sorted(all_gauges.items()):
            if name not in emitted:
                help_text, metric_type = all_meta.get(name, ("", "gauge"))
                if help_text:
                    lines.append(f"# HELP {name} {help_text}")
                lines.append(f"# TYPE {name} {metric_type}")
                emitted.add(name)
            if label_tuple:
                label_str = ",".join(f'{key}="{val}"' for key, val in label_tuple)
                lines.append(f"{name}{{{label_str}}} {value}")
            else:
                lines.append(f"{name} {value}")
        lines.append("")
        return "\n".join(lines)

    def log_message(self, format, *args):
        pass


_store: Optional[PrometheusMetricsStore] = None
_rank: int = 0
_local_rank: int = 0
_prev_timer_elapsed: Dict[str, float] = {}
_prev_wall_time: Optional[float] = None


def get_metrics_store() -> Optional[PrometheusMetricsStore]:
    return _store


def _write_metrics_file():
    if _store is None:
        return
    filepath = os.path.join(_METRICS_DIR, f"rank_{_local_rank}.json")
    tmp = filepath + ".tmp"
    try:
        with open(tmp, "w") as f:
            json.dump(_store.snapshot(), f)
        os.replace(tmp, filepath)
    except OSError:
        pass


def start_server(port: int, rank: int, local_rank: int) -> PrometheusMetricsStore:
    """Initialise metrics store for every rank; start HTTP server on local_rank 0 only."""

    global _store, _rank, _local_rank
    _rank = rank
    _local_rank = local_rank
    _store = PrometheusMetricsStore()

    os.makedirs(_METRICS_DIR, exist_ok=True)

    if local_rank == 0:
        for filepath in _glob.glob(os.path.join(_METRICS_DIR, "rank_*.json")):
            try:
                os.remove(filepath)
            except OSError:
                pass

        handler = type("_Handler", (_MetricsHandler,), {"_metrics_dir": _METRICS_DIR})
        server = HTTPServer(("0.0.0.0", port), handler)
        threading.Thread(target=server.serve_forever, daemon=True, name="prometheus-metrics").start()

    return _store


def update_iteration_timings(*, iteration: int, timers, timers_to_log: list):
    """Update per-step wall-clock and timer delta metrics."""

    store = _store
    if store is None:
        return

    global _prev_wall_time
    rank_labels = {"rank": str(_rank), "gpu": str(_local_rank)}

    store.set_gauge(
        "megatron_training_iteration",
        iteration,
        "Current training iteration",
        labels=rank_labels,
    )

    now = _time.time()
    if _prev_wall_time is not None:
        store.set_gauge(
            "megatron_iteration_time_ms",
            (now - _prev_wall_time) * 1000.0,
            "Wall-clock time of last iteration in milliseconds",
            labels=rank_labels,
        )
    _prev_wall_time = now

    for timer_name in timers_to_log:
        if timer_name not in timers._timers:
            continue
        current = timers._timers[timer_name]._elapsed
        previous = _prev_timer_elapsed.get(timer_name, 0.0)
        delta = current - previous if current >= previous else current
        _prev_timer_elapsed[timer_name] = current
        store.set_gauge(
            "megatron_timer_ms",
            delta * 1000.0,
            "Per-step timer value in milliseconds",
            labels={**rank_labels, "name": timer_name},
        )

    _write_metrics_file()
