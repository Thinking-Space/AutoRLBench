#!/usr/bin/env python3
import argparse
import ast
import json
import re
from pathlib import Path

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
STEP_RE = re.compile(r"step \d+: (\{.*\})")
LOGPROBS_MEM_RE = re.compile(
    r"after actor_log_probs forward.*?peak_alloc=([0-9.]+)GB.*?nvidia_smi_used=([0-9.]+)GB"
)
TRAIN_PEAK_RE = re.compile(
    r"after actor_train peak.*?peak_alloc=([0-9.]+)GB.*?nvidia_smi_used=([0-9.]+)GB"
)
TIMER_RE = re.compile(r"Timer log_probs end \(elapsed: ([0-9.]+)s\)")


def strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text)


def parse_run(label: str, path: Path) -> dict:
    text = strip_ansi(path.read_text(errors="replace"))
    result = {
        "label": label,
        "log_path": str(path),
        "status": "unknown",
        "log_probs_peak_alloc_gb": None,
        "log_probs_nvidia_smi_used_gb": None,
        "actor_train_peak_alloc_gb": None,
        "actor_train_nvidia_smi_used_gb": None,
        "log_probs_elapsed_s": None,
        "logdiff_mean": None,
        "logdiff_p999": None,
        "logdiff_max": None,
    }

    if "Ray job succeeded" in text or "CHUNK ABLATION done!" in text:
        result["status"] = "success"
    if "TorchMemorySaver::malloc return OOM" in text or "CUDA out of memory" in text:
        result["status"] = "oom"
    if "Ray job failed" in text and result["status"] == "unknown":
        result["status"] = "failed"

    log_probs_mem = [(float(a), float(b)) for a, b in LOGPROBS_MEM_RE.findall(text)]
    if log_probs_mem:
        result["log_probs_peak_alloc_gb"] = max(x[0] for x in log_probs_mem)
        result["log_probs_nvidia_smi_used_gb"] = max(x[1] for x in log_probs_mem)

    train_peak = [(float(a), float(b)) for a, b in TRAIN_PEAK_RE.findall(text)]
    if train_peak:
        result["actor_train_peak_alloc_gb"] = max(x[0] for x in train_peak)
        result["actor_train_nvidia_smi_used_gb"] = max(x[1] for x in train_peak)

    elapsed = [float(x) for x in TIMER_RE.findall(text)]
    if elapsed:
        result["log_probs_elapsed_s"] = max(elapsed)

    step_payloads = []
    for match in STEP_RE.findall(text):
        try:
            payload = ast.literal_eval(match)
        except Exception:
            continue
        if isinstance(payload, dict):
            step_payloads.append(payload)
    if step_payloads:
        payload = step_payloads[-1]
        result["logdiff_mean"] = payload.get("train/train_rollout_logprob_abs_diff")
        result["logdiff_p999"] = payload.get("train/logdiff_p999")
        result["logdiff_max"] = payload.get("train/logdiff_max")
        if result["status"] == "unknown":
            result["status"] = "success"

    return result


def fmt(x):
    if x is None:
        return "-"
    if isinstance(x, float):
        return f"{x:.3f}"
    return str(x)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize chunk ablation logs into a Markdown table."
    )
    parser.add_argument(
        "runs",
        nargs="+",
        help="Run spec in the form label=/abs/or/relative/path/to/log",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print JSON instead of Markdown.",
    )
    args = parser.parse_args()

    summaries = []
    for run in args.runs:
        if "=" not in run:
            raise SystemExit(f"Invalid run spec: {run!r}; expected label=path")
        label, raw_path = run.split("=", 1)
        summaries.append(parse_run(label, Path(raw_path)))

    if args.json:
        print(json.dumps(summaries, ensure_ascii=False, indent=2))
        return

    print("| run | status | log_probs peak_alloc (GB) | log_probs nvidia-smi (GB) | actor_train peak_alloc (GB) | actor_train nvidia-smi (GB) | log_probs time (s) | logdiff mean | logdiff p999 | logdiff max |")
    print("|-----|--------|---------------------------|----------------------------|-----------------------------|------------------------------|--------------------|--------------|--------------|-------------|")
    for s in summaries:
        print(
            f"| {s['label']} | {s['status']} | {fmt(s['log_probs_peak_alloc_gb'])} | "
            f"{fmt(s['log_probs_nvidia_smi_used_gb'])} | {fmt(s['actor_train_peak_alloc_gb'])} | "
            f"{fmt(s['actor_train_nvidia_smi_used_gb'])} | {fmt(s['log_probs_elapsed_s'])} | "
            f"{fmt(s['logdiff_mean'])} | {fmt(s['logdiff_p999'])} | {fmt(s['logdiff_max'])} |"
        )


if __name__ == "__main__":
    main()
