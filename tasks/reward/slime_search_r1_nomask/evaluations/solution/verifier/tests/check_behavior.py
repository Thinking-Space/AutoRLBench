#!/usr/bin/env python3
"""CPU contract probe: run the submitted generator with fake model/retriever IO.

The actual generator, search formatter and reward wrapper are loaded from the
submission; the probe supplies its own minimal Sample implementation. No
training, model loading or network is needed.
This checks behavior, not spellings of a proposed fix. Run in a separate process
so patched IO modules never affect metric extraction.
"""
from __future__ import annotations

import asyncio
import contextlib
import importlib.util
import io
import json
import secrets
import sys
import types
from pathlib import Path


def probe(workspace: Path) -> dict:
    examples = workspace / "examples/search-r1"
    # Import only the submitted generator and the protected scorer from the
    # workspace.  The probe owns Sample below; otherwise a submission could
    # replace Sample and make the check validate its own test double.
    sys.path.insert(0, str(examples))
    errors = []
    checks = {}
    requests = []
    pending = []
    suffix = secrets.token_hex(4)
    title = "Probe title " + suffix
    body = "First passage fact " + suffix + ".\nSecond passage fact " + suffix + "."
    documents = [{"document": {"contents": title + "\n" + body}}]

    class Tokenizer:
        def __call__(self, text, **kwargs):
            return {"input_ids": [ord(ch) for ch in text]}

        def encode(self, text, **kwargs):
            return self(text)["input_ids"]

        def decode(self, tokens, **kwargs):
            return "".join(map(chr, tokens))

    async def post(url, payload, **kwargs):
        requests.append(payload)
        if not pending:
            raise AssertionError("generator did not stop after the answer")
        text = pending.pop(0)
        return {"text": text, "meta_info": {
            "finish_reason": {"type": "stop"},
            "output_token_logprobs": [[-0.5, ord(ch), None] for ch in text],
        }}

    async def retrieve(*args, **kwargs):
        return documents

    class ProbeSample:
        class Status:
            ABORTED = "aborted"
            COMPLETED = "completed"
            TRUNCATED = "truncated"

        def __init__(self, prompt="", response="", label=None):
            self.prompt = prompt
            self.response = response
            self.label = label or {}
            self.tokens = []
            self.loss_mask = []
            self.response_length = 0
            self.rollout_log_probs = None
            self.rollout_top_p_token_ids = None
            self.rollout_top_p_token_offsets = None
            self.status = None
            self.reward = None
            self._appended = []

        def append_response_tokens(
            self, args, tokens, log_probs=None, trainable=True, meta_info=None
        ):
            self._appended.append({
                "tokens": list(tokens),
                "log_probs": None if log_probs is None else list(log_probs),
                "trainable": bool(trainable),
                "meta_info": meta_info,
            })

    http = types.ModuleType("slime.utils.http_utils")
    http.post = post
    http.is_port_available = lambda *args, **kwargs: True
    rollout = types.ModuleType("slime.rollout.sglang_rollout")
    rollout.GenerateState = lambda args: types.SimpleNamespace(tokenizer=Tokenizer())
    local = types.ModuleType("local_search_server")
    local.local_search = retrieve
    # Install package parents as well as leaf modules.  This prevents imports
    # from falling through to arbitrary submission-side slime modules while
    # retaining the real submitted generator and scorer.
    slime = types.ModuleType("slime")
    slime.__path__ = []
    utils = types.ModuleType("slime.utils")
    utils.__path__ = []
    rollout_pkg = types.ModuleType("slime.rollout")
    rollout_pkg.__path__ = []
    types_mod = types.ModuleType("slime.utils.types")
    types_mod.Sample = ProbeSample
    sys.modules.update({
        "slime": slime,
        "slime.utils": utils,
        "slime.rollout": rollout_pkg,
        "slime.utils.types": types_mod,
        "slime.utils.http_utils": http,
        "slime.rollout.sglang_rollout": rollout,
        "local_search_server": local,
    })
    Sample = ProbeSample

    path = examples / "generate_with_search.py"
    if path.is_symlink() or not path.is_file():
        raise ValueError("missing regular custom generator")
    spec = importlib.util.spec_from_file_location("generate_with_search", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    args = types.SimpleNamespace(partial_rollout=False, sglang_router_ip="localhost",
                                 sglang_router_port=1, rollout_max_context_len=16000,
                                 eval_max_context_len=16000, reward_key=None)
    prompt = "Example: <answer>demo</answer>\n<|im_start|>assistant\n"
    first = "<think>Need evidence.</think><search>first " + suffix + "</search>"
    second = "<think>Check again.</think><search>second " + suffix + "</search>"
    answer = "<think>Done.</think><answer>" + suffix + "</answer>"
    pending.extend([first, second, answer])
    sample = Sample(prompt=prompt, label={"ground_truth": {"target": [suffix]}})
    try:
        result = asyncio.run(module.generate(args, sample, {"max_new_tokens": 8192,
                                                            "no_stop_trim": True}))
        tokens = result.tokens[-result.response_length:]
        mask = list(result.loss_mask)
        checks["alignment"] = (
            len(tokens) == len(mask) == result.response_length
        )
        # Model output is known; everything inserted between its turns is external.
        cursor = 0
        expected = []
        body_visible = len(requests) == 3
        response = result.response
        for i, model_text in enumerate([first, second, answer]):
            start = response.find(model_text, cursor)
            if start < 0:
                raise AssertionError("model output disappeared from the response")
            expected.extend([0] * (start - cursor))
            expected.extend([1] * len(model_text))
            cursor = start + len(model_text)
            if i:
                payload = requests[i]
                context = payload.get("text")
                if context is None:
                    context = Tokenizer().decode(payload.get("input_ids", []))
                body_visible = body_visible and all(line in context for line in body.splitlines())
        expected.extend([0] * (len(response) - cursor))
        checks["passage_body_reaches_model"] = bool(body_visible)
        checks["observations_not_trainable"] = (
            len(expected) == len(mask) and 0 in expected and mask == expected
        )
        checks["model_tokens_trainable"] = (
            len(expected) == len(mask) and all(m == 1 for m, want in zip(mask, expected) if want)
        )
        checks["token_text_alignment"] = Tokenizer().decode(tokens) == response
        appended_tokens = [token for call in result._appended for token in call["tokens"]]
        appended_trainable = [call["trainable"] for call in result._appended for _ in call["tokens"]]
        checks["append_contract"] = (
            appended_tokens == tokens
            and len(appended_trainable) == len(expected)
            and appended_trainable == [bool(want) for want in expected]
        )
        checks["observation_logprobs_zero"] = (
            len(result.rollout_log_probs or []) == len(expected)
            and all(
                value == 0.0
                for value, want in zip(result.rollout_log_probs, expected)
                if not want
            )
        )
        for key, ok in checks.items():
            if not ok:
                errors.append(key)
    except Exception as exc:
        errors.append(f"generator probe failed: {type(exc).__name__}: {exc}")

    # eval/nq_test uses this same reward wrapper without evaluation kwargs.
    # Preserve its binary EM contract: formatting/retrieval alone cannot score.
    rewards = {}
    fixtures = {
        "correct": (f"<think>x</think><answer>{suffix}</answer>", 1.0),
        "correct_after_retrieval": (
            f"<think>x</think><search>q</search><information>{suffix}</information>"
            f"<think>x</think><answer>{suffix}</answer>",
            1.0,
        ),
        "formatted_wrong": ("<think>x</think><answer>wrong</answer>", 0.0),
        "retrieved_but_wrong": (f"<think>x</think><search>q</search><information>{suffix}</information>"
                                "<think>x</think><answer>wrong</answer>", 0.0),
        "unformatted_wrong": ("<answer>wrong</answer>", 0.0),
    }
    for name, (response, expected_reward) in fixtures.items():
        try:
            sample = Sample(prompt=prompt, response=response,
                            label={"ground_truth": {"target": [suffix]}})
            values = [
                asyncio.run(module.reward_func(args, sample)),
                asyncio.run(module.reward_func(args, sample, evaluation=True)),
            ]
            rewards[name] = values[0]
            for value in values:
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (float, int))
                    or abs(value - expected_reward) > 1e-8
                ):
                    errors.append(
                        f"eval reward contract: {name} expected {expected_reward}, got {value!r}"
                    )
        except Exception as exc:
            errors.append(f"reward probe {name} failed: {type(exc).__name__}: {exc}")
    checks["eval_reward_is_binary_em"] = not any("reward" in e for e in errors)
    return {"status": "passed" if not errors else "failed", "checks": checks,
            "reward_cases": rewards, "errors": errors}


def main():
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            result = probe(Path(sys.argv[1]).resolve())
    except Exception as exc:
        result = {"status": "unavailable", "checks": {},
                  "errors": [f"behavior probe unavailable: {type(exc).__name__}: {exc}"]}
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
