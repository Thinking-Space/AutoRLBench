"""Diagnostic probe for the full_determinism seed experiment. Only active when ARB_PROBE_DIR is set; it is put on
PYTHONPATH for dedicated probe runs, so the arms' code is unchanged.

Wraps vLLMHttpServer.generate (verl/workers/rollout/vllm_rollout/vllm_async_server.py). After each request it records
the request id, the sampling seed actually used (generate sets it on the sampling_params dict it receives), the
temperature, and SHA-1 digests of the prompt and response token ids.
"""
import functools
import hashlib
import importlib.abc
import importlib.util
import json
import os
import sys

PROBE_DIR = os.environ.get("ARB_PROBE_DIR")
TARGET = "verl.workers.rollout.vllm_rollout.vllm_async_server"


def _digest(ids):
    return hashlib.sha1(json.dumps([int(i) for i in ids]).encode()).hexdigest()


def _patch(mod):
    cls = mod.vLLMHttpServer
    orig = cls.generate
    counter = {"n": 0}

    @functools.wraps(orig)
    async def generate(self, prompt_ids, sampling_params, request_id, *args, **kwargs):
        out = await orig(self, prompt_ids, sampling_params, request_id, *args, **kwargs)
        try:
            counter["n"] += 1
            rec = {
                "call": counter["n"],
                "request_id": request_id,
                "priority": int(kwargs.get("priority", 0)),
                "seed": sampling_params.get("seed"),
                "temperature": sampling_params.get("temperature"),
                "prompt_sha": _digest(prompt_ids),
                "out_sha": _digest(out.token_ids),
                "n_out": len(out.token_ids),
            }
            with open(os.path.join(PROBE_DIR, f"gen_{os.getpid()}.jsonl"), "a") as f:
                f.write(json.dumps(rec) + "\n")
        except Exception as e:  # never break the run being observed
            with open(os.path.join(PROBE_DIR, "probe.log"), "a") as f:
                f.write(f"probe failed: {e!r}\n")
        return out

    cls.generate = generate


class _PatchOnImport(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name != TARGET:
            return None
        sys.meta_path.remove(self)
        try:
            spec = importlib.util.find_spec(name)
        finally:
            sys.meta_path.insert(0, self)
        if spec is None or spec.loader is None:
            return spec
        exec_module = spec.loader.exec_module

        def exec_and_patch(module):
            exec_module(module)
            _patch(module)

        spec.loader.exec_module = exec_and_patch
        return spec


if PROBE_DIR:
    os.makedirs(PROBE_DIR, exist_ok=True)
    sys.meta_path.insert(0, _PatchOnImport())
