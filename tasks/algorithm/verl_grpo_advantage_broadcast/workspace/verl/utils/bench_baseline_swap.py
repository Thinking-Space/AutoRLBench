# -*- coding: utf-8 -*-
"""
Bench: during _validate, paste every sealed file under hidden/baseline onto
the workspace, then restore after validation.

Agent never sees /hidden (mounted only for GPU verify). This module only runs
inside the training process.
"""

from __future__ import annotations

import os
import shutil
import sys
from typing import Any, Dict, List, Optional


def baseline_root() -> str:
    env = (os.environ.get("BENCH_BASELINE_REWARD_ROOT") or "").strip()
    if env and os.path.isdir(env):
        return env
    if os.path.isdir("/hidden/baseline"):
        return "/hidden/baseline"
    return ""


def workspace_root() -> str:
    for cand in (
        (os.environ.get("BENCH_WORKSPACE") or "").strip(),
        "/workspace",
        os.getcwd(),
    ):
        if cand and os.path.isdir(os.path.join(cand, "verl")):
            return cand
    return os.getcwd()


def list_baseline_py_rels(root: str) -> List[str]:
    rels: List[str] = []
    if not root or not os.path.isdir(root):
        return rels
    for dirpath, _dirnames, filenames in os.walk(root):
        if "__pycache__" in dirpath.replace("\\", "/").split("/"):
            continue
        for name in filenames:
            if not name.endswith(".py"):
                continue
            full = os.path.join(dirpath, name)
            rel = os.path.relpath(full, root).replace("\\", "/")
            rels.append(rel)
    rels.sort()
    return rels


def _rel_to_modname(rel: str) -> str:
    rel = rel.replace("\\", "/").lstrip("/")
    if rel.endswith(".py"):
        rel = rel[:-3]
    return rel.replace("/", ".")


def _clear_pycache_for(path: str) -> None:
    parent = os.path.dirname(path)
    base = os.path.splitext(os.path.basename(path))[0]
    pycache = os.path.join(parent, "__pycache__")
    if not os.path.isdir(pycache):
        return
    for name in os.listdir(pycache):
        if name.startswith(base + "."):
            try:
                os.remove(os.path.join(pycache, name))
            except OSError:
                pass


def purge_modules_for_rels(rels: List[str]) -> None:
    """Drop swapped modules from sys.modules so next import reads disk."""
    targets = {_rel_to_modname(r) for r in rels}
    # Also drop importlib-sealed helpers in reward_score
    try:
        import verl.utils.reward_score as rs

        cache = getattr(rs, "_BASELINE_FN_CACHE", None)
        if isinstance(cache, dict):
            cache.clear()
    except Exception:
        pass

    to_del = []
    for name in list(sys.modules.keys()):
        if name in targets:
            to_del.append(name)
            continue
        for t in targets:
            if name.startswith(t + "."):
                to_del.append(name)
                break
    for name in to_del:
        sys.modules.pop(name, None)


def install_baseline_files(
    *,
    baseline: Optional[str] = None,
    workspace: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Paste all sealed *.py under baseline onto workspace. Returns state for restore.
    """
    broot = baseline if baseline is not None else baseline_root()
    wroot = workspace if workspace is not None else workspace_root()
    state: Dict[str, Any] = {
        "ok": False,
        "baseline": broot,
        "workspace": wroot,
        "rels": [],
        "backups": {},  # rel -> original bytes or None if missing
        "installed": [],
    }
    if not broot or not os.path.isdir(broot):
        return state
    if not wroot or not os.path.isdir(wroot):
        return state

    rels = list_baseline_py_rels(broot)
    for rel in rels:
        src = os.path.join(broot, rel)
        dst = os.path.join(wroot, rel)
        if os.path.isfile(dst):
            with open(dst, "rb") as f:
                state["backups"][rel] = f.read()
        else:
            state["backups"][rel] = None
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
        _clear_pycache_for(dst)
        state["installed"].append(rel)

    state["rels"] = list(rels)
    state["ok"] = True
    if state["installed"]:
        print(
            f"[bench] validate: pasted {len(state['installed'])} sealed baseline "
            f"file(s) from {broot}",
            flush=True,
        )
    return state


def restore_baseline_files(state: Dict[str, Any]) -> None:
    """Restore Agent workspace files after validate."""
    if not state or not state.get("ok"):
        return
    wroot = state.get("workspace") or ""
    backups: Dict[str, Any] = state.get("backups") or {}
    restored = 0
    for rel, blob in backups.items():
        dst = os.path.join(wroot, rel)
        if blob is None:
            # sealed-only file that did not exist before: remove paste
            try:
                if os.path.isfile(dst):
                    os.remove(dst)
            except OSError:
                pass
            continue
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with open(dst, "wb") as f:
            f.write(blob)
        _clear_pycache_for(dst)
        restored += 1
    if restored or state.get("installed"):
        print(
            f"[bench] validate: restored {restored} Agent file(s) after baseline paste",
            flush=True,
        )


def begin_baseline_validate(reward_loop_manager=None) -> Dict[str, Any]:
    """Paste sealed files + reinit reward workers to pick up baseline code."""
    state = install_baseline_files()
    state["reward_loop_manager"] = reward_loop_manager
    if not state.get("ok") or not state.get("rels"):
        return state
    purge_modules_for_rels(state["rels"])
    mgr = reward_loop_manager
    if mgr is not None and getattr(mgr, "reward_loop_workers", None):
        import ray

        ray.get(
            [
                w.reinit_reward_fn.remote(state["rels"])
                for w in mgr.reward_loop_workers
            ]
        )
    return state


def end_baseline_validate(state: Optional[Dict[str, Any]]) -> None:
    """Restore Agent files + reinit reward workers."""
    if not state:
        return
    restore_baseline_files(state)
    rels = list(state.get("rels") or [])
    if rels:
        purge_modules_for_rels(rels)
    mgr = state.get("reward_loop_manager")
    if mgr is not None and getattr(mgr, "reward_loop_workers", None):
        import ray

        try:
            ray.get([w.reinit_reward_fn.remote(rels) for w in mgr.reward_loop_workers])
        except Exception as e:
            print(f"[bench] WARN: reward worker reinit after restore failed: {e}", flush=True)
