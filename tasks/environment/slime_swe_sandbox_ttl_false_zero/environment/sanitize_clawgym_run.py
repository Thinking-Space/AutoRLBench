#!/usr/bin/env python3
"""Sanitize the ClawGym defective arm evidence for task packaging."""

import json
import re
import shutil
import sys
from pathlib import Path

ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
REPLACEMENTS = [
    (re.compile(r"/user/chenyinghao/harbor-runs/slime-swe-ttl/runs/defective_ttl120"), "/workspace/observed_run"),
    (re.compile(r"/user/chenyinghao/harbor-runs/slime-swe-ttl/runs/control_ttl400"), "/workspace/control_run"),
    (re.compile(r"/user/chenyinghao/harbor-runs/slime-swe-ttl"), "/workspace"),
    (re.compile(r"/tmp/ray_defective_ttl120"), "/tmp/ray_main"),
    (re.compile(r"200\.20\.\d+\.\d+"), "10.0.0.2"),
    (re.compile(r"devspace-job-autosft-ptb-\d+-main-0"), "trainer-0"),
    (re.compile(r"devspace-chenyinghao-autosft-ptb-\d+"), "trainer-0"),
    (re.compile(r"raysubmit_[A-Za-z0-9_]+"), "raysubmit_redacted"),
    (re.compile(r"chenyinghao"), "team"),
    (re.compile(r"defective_ttl120"), "observed_run"),
    (re.compile(r"harbor-runs"), "workspace"),
    (re.compile(r"modelbest\.tencentcloudcr\.com/code-interpreter/"), "registry.internal/code-interpreter/"),
    (re.compile(r"ap-beijing\.tencentags\.com"), "ags-gateway.internal"),
]

def clean(text: str) -> str:
    text = ANSI_RE.sub("", text)
    for pat, repl in REPLACEMENTS:
        text = pat.sub(repl, text)
    return text

def clean_file(p: Path) -> None:
    try:
        raw = p.read_bytes()
    except OSError:
        return
    if b"\x00" in raw[:4096]:
        return
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return
    new = clean(text)
    if new != text:
        p.write_text(new)

def main() -> int:
    run_root = Path(sys.argv[1])
    out = Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)

    # run.log
    log = run_root / "run.log"
    if log.exists():
        (out / "run.log").write_text(clean(log.read_text(errors="replace")))
        print("[stage] run.log")

    # episodes
    eps_out = out / "episodes"
    eps_out.mkdir(exist_ok=True)
    keep_top = {"episode.json", "sandbox.log", "patch.diff", "episode.events.log", "verify"}
    n = 0
    for ep in sorted((run_root / "episodes").iterdir()):
        if not (ep / "episode.json").exists():
            continue
        dst = eps_out / ep.name
        shutil.copytree(ep, dst, symlinks=True,
                        ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache", "*.pyc"))
        for item in list(dst.iterdir()):
            if item.name not in keep_top and item.name != "sandbox_workdir":
                if item.is_dir():
                    shutil.rmtree(item, ignore_errors=True)
                else:
                    item.unlink()
        for p in dst.rglob("*"):
            if p.is_file() and p.suffix != ".pt":
                clean_file(p)
        n += 1
    print(f"[stage] {n} episodes")

    # rollout dumps (first/middle/last)
    dump_out = out / "rollout_dumps"
    dump_out.mkdir(exist_ok=True)
    import torch
    dumps = sorted((run_root / "rollout_dumps").glob("*.pt"))
    indices = {0, len(dumps)//2, len(dumps)-1} if dumps else set()
    for i in sorted(indices):
        if i < len(dumps):
            d = torch.load(dumps[i], weights_only=False)
            def walk(o):
                if isinstance(o, str): return clean(o)
                if isinstance(o, dict): return {k: walk(v) for k, v in o.items()}
                if isinstance(o, list): return [walk(v) for v in o]
                return o
            torch.save(walk(d), dump_out / dumps[i].name)
            print(f"[stage] dump {dumps[i].name}")

    # leftover scan
    hits = []
    for p in out.rglob("*"):
        if p.is_file() and p.suffix in (".log", ".json", ".err", ".md", ".diff", ".txt", ".jsonl", ".py", ".sh"):
            try:
                t = p.read_text(errors="ignore")
            except OSError:
                continue
            for bad in ("chenyinghao", "200.20.", "devspace-", "defective", "harbor-runs", "tencentags", "tencentcloudcr"):
                if bad in t:
                    hits.append((str(p), bad))
    if hits:
        print("[stage] LEFTOVER HITS:")
        for h in hits[:20]:
            print("   ", h)
        return 1
    print("[stage] leftover scan clean")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
