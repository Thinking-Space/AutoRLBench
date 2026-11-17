#!/usr/bin/env python3
"""Verify the shipped historical evidence: archive identity, metric re-derivation, redaction scan."""
import gzip
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HIST = ROOT / "workspace/logs/historical"
FORBIDDEN = re.compile(
    rb"/user/|/workspace/user|/home/[a-z]|api[_ -]?key\W{1,4}[A-Za-z0-9_\-]{16,}|sk-[A-Za-z0-9]{24,}|"
    rb"svc\.cluster\.local|startup\.sh|_runtime_env|\bjob-[a-z0-9-]+|pytorchjob|"
    rb"(?<![\d.])(?!127\.0\.0\.1)(?:\d{1,3}\.){3}\d{1,3}(?![\d.])", re.I)

manifest = json.loads((HIST / "manifest.json").read_text())
records = {}
for name, info in manifest["archives"].items():
    path = HIST / name
    assert hashlib.sha256(path.read_bytes()).hexdigest() == info["sha256"], f"sha256 mismatch: {name}"
    with gzip.open(path, "rb") as handle:
        lines = handle.read().splitlines()
    assert len(lines) == info["records"], f"record count mismatch: {name}"
    for line in lines:
        json.loads(line)
    records[name] = len(lines)

subprocess.run([sys.executable, "-B", str(HIST / "derive_metrics.py"), str(HIST), "--check"], check=True,
               stdout=subprocess.DEVNULL)

hits = []
SCANNED = [ROOT / "workspace", ROOT / "evaluations/issue/verifier/resources"]
for path in sorted(p for base in SCANNED for p in base.rglob("*")):
    if not path.is_file() or "experiment_logs" in path.parts:
        continue
    if path.suffix == ".pyc" or "__pycache__" in path.parts:
        hits.append(f"{path.relative_to(ROOT)}: compiled bytecode embeds build-host paths")
        continue
    if "vendor" in path.parts:
        continue
    data = path.read_bytes()
    if path.suffix == ".gz":
        data = gzip.decompress(data)
    for match in FORBIDDEN.finditer(data):
        hits.append(f"{path.relative_to(ROOT)}: {match.group(0)[:40]!r}")
assert not hits, "redaction scan hits:\n" + "\n".join(hits[:20])
print(json.dumps({"ok": True, "archives": records, "metrics_rederived": True, "redaction_hits": 0}, sort_keys=True))
