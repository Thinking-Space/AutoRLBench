"""Multi-turn Endless Terminals rollout for slime.

Wired in via:
    --custom-generate-function-path et.rollout.generate
    --custom-rm-path                et.rollout.reward_func

One sample == one terminal episode. Each turn the policy emits exactly one
`<command>...</command>` (or `<action>done</action>`); the command runs in the
episode's private shell and the harness-formatted result is appended to the
context as a non-trainable user turn. The episode reward is the task's own
pytest verifier: 1.0 or 0.0, awarded once at the end of the episode.

The system prompt below is copied verbatim from the Endless Terminals reference
transcripts (`solution/o3_summary.json`), as is the observation wording, so the
policy sees the same interface the dataset was built against.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re

from slime.rollout.sglang_rollout import GenerateState
from slime.utils.http_utils import post
from slime.utils.types import Sample

from et.sandbox import Episode, check_exec_capable

SYSTEM_PROMPT = """You are a highly capable Linux terminal agent operating strictly via a single-shell-command interface.
Goal: complete the user's task.

Rules:
- Output exactly one of the following per turn:
  1) <command>THE_SINGLE_SHELL_COMMAND</command>
  OR
  2) <action>done</action>
- Don't use interactive commands and confirmations; use non-interactive flags.
- Prefer simple, robust CLI tools; write files explicitly when needed.
- If you believe the task is solved, emit <action>done</action>.
- Hint: you might want to run commands interactively to see the output and then write the command. Don't just pipe the commands.
- Only your first command in command tags will be executed. So don't respond with multiple commands.
- Hint: Verify your solution once you are done. Eg: you can use cat to see the input and the output.
- Do not just write long bash scripts. Write the commands that you would write in a terminal."""

DATA_ROOT = os.environ.get("ET_DATA_ROOT", "/tmp/et_prompts")
# Must allow execution, so not under /dev/shm -- see et.sandbox.check_exec_capable.
SCRATCH_ROOT = os.environ.get("ET_SCRATCH", "/tmp/et_episodes")
MAX_TURNS = int(os.environ.get("ET_MAX_TURNS", "8"))
MAX_NEW_TOKENS_PER_TURN = int(os.environ.get("ET_MAX_NEW_TOKENS", "512"))
CMD_TIMEOUT = float(os.environ.get("ET_CMD_TIMEOUT", "20"))
MAX_OBS_CHARS = int(os.environ.get("ET_MAX_OBS_CHARS", "1500"))
# Hard ceiling on one episode, covering everything: generation, the shell, the
# verifier and teardown. Every individual step is already bounded, yet the run
# still hung -- slime builds its HTTP client with `httpx.Timeout(None)`, so a
# `/generate` that never comes back waits forever, and asyncio's
# `Process.wait()` has its own way of never resolving (see sandbox.REAP_TIMEOUT).
# Both drivers await every episode before returning, so one episode that never
# finishes stalls the entire rollout rather than failing it. The default is
# deliberately far above any legitimate episode -- 6 turns of generation plus
# 6x20s of commands plus a 180s verifier -- so this only ever fires on a hang.
EPISODE_TIMEOUT = float(os.environ.get("ET_EPISODE_TIMEOUT", "900"))

STOP_STRINGS = ["</command>", "</action>"]
_OBS_SENTINEL = "@@ET_OBSERVATION@@"

_COMMAND_RE = re.compile(r"<command>(.*?)(?:</command>|\Z)", re.DOTALL)
_ACTION_RE = re.compile(r"<action>\s*done\s*</action>", re.IGNORECASE)


class _Chat:
    """Derives the chat-template glue once, without hardcoding ChatML.

    `head` is the token prefix the policy generates from. `tail` is the text
    that must follow a generated assistant turn in order to inject one user
    turn and re-open the assistant turn.
    """

    def __init__(self, tokenizer) -> None:
        kwargs = {"tokenize": False, "add_generation_prompt": True, "enable_thinking": False}
        probe_msgs = [{"role": "system", "content": "S"}, {"role": "user", "content": "U"}]
        head = tokenizer.apply_chat_template(probe_msgs, **kwargs)
        full = tokenizer.apply_chat_template(
            probe_msgs
            + [{"role": "assistant", "content": "A"}, {"role": "user", "content": _OBS_SENTINEL}],
            **kwargs,
        )
        prefix = os.path.commonprefix([head, full])
        assert full[len(prefix) : len(prefix) + 1] == "A", "unexpected chat template shape"
        self._tail = full[len(prefix) + 1 :]
        assert _OBS_SENTINEL in self._tail, "could not locate the observation slot in the chat template"
        self._kwargs = kwargs

    def prompt(self, tokenizer, instruction: str) -> str:
        return tokenizer.apply_chat_template(
            [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": instruction}],
            **self._kwargs,
        )

    def observation(self, text: str) -> str:
        return self._tail.replace(_OBS_SENTINEL, text)


_CHAT: _Chat | None = None
_SCRATCH_CHECKED = False


def _chat(tokenizer) -> _Chat:
    global _CHAT
    if _CHAT is None:
        _CHAT = _Chat(tokenizer)
    return _CHAT


def _check_scratch_once() -> None:
    global _SCRATCH_CHECKED
    if not _SCRATCH_CHECKED:
        check_exec_capable(SCRATCH_ROOT)
        _SCRATCH_CHECKED = True


def parse_action(text: str) -> tuple[str, str | None]:
    """Return ("command", cmd) / ("done", None) / ("invalid", None).

    Mirrors the harness rule "only your first command in command tags will be
    executed": whichever of `<command>` / `<action>` appears first wins.
    """
    cmd_match = _COMMAND_RE.search(text)
    act_match = _ACTION_RE.search(text)
    if act_match and (cmd_match is None or act_match.start() < cmd_match.start()):
        return "done", None
    if cmd_match:
        cmd = cmd_match.group(1).strip()
        return ("command", cmd) if cmd else ("invalid", None)
    return "invalid", None


def _abort(args, sample: Sample, turns: int, **extra) -> Sample:
    """Finish a sample that is giving up, with every field slime will index.

    `_convert_samples_to_train_data` repeatedly decides whether an optional
    per-sample field is present by looking at `samples[0]` alone, and then
    indexes *every* sample in the batch. An episode that gives up while its
    neighbours complete normally is therefore not a local failure -- it takes
    down the whole rollout, from a traceback that names slime rather than this
    file. Three such fields exist on this path:

      * `reward` -- eval metrics take `sum(rewards) / len(rewards)` over the
        split (rollout_metrics.py:232), so one `None` raises;
      * `metadata["round_number"]` -- ray/rollout.py:382, `KeyError`;
      * `rollout_log_probs` -- ray/rollout.py:386, and a sample that never got
        a single response token still has `None` there, which surfaces much
        later as `TypeError: must be real number, not NoneType` inside
        `torch.as_tensor`.

    The last one is why an aborted episode with nothing generated is given one
    end-of-text token, appended as *non-trainable*: that is slime's own
    supported way to add tokens that carry no gradient, so every per-sample
    list ends up present, mutually consistent and zero-weighted rather than
    absent. Centralising the giving-up paths here is the only way to keep all
    of this true as new ones get added.
    """
    sample.status = Sample.Status.ABORTED
    sample.reward = 0.0
    if sample.response_length == 0:
        eos = GenerateState(args).tokenizer.eos_token_id
        sample.append_response_tokens(args, tokens=[eos], trainable=False, text="")
    sample.metadata = {
        **(sample.metadata or {}),
        "et_task": sample.label,
        "round_number": turns,
        **extra,
    }
    return sample


def _turns_so_far(sample: Sample) -> int:
    """How many turns the episode had started when it was interrupted.

    `_generate` stamps the turn counter into metadata as it goes, precisely so
    that a timeout or an exception -- which unwinds past every local variable --
    still has a truthful number to report.
    """
    return int((sample.metadata or {}).get("round_number", 0))


async def generate(args, sample: Sample, sampling_params) -> Sample:
    """slime hook: `custom_generate_function_path`. Never raises, always returns.

    Both drivers that call this -- `generate_rollout` and `eval_rollout` --
    launch every episode as a task and then `await` them through
    `asyncio.as_completed`, so an episode that raises discards the entire
    rollout and an episode that hangs stalls it indefinitely. Both have
    happened: a task printing a line longer than asyncio's stream limit threw
    away several hundred finished episodes, and a task whose instructions are
    to leave `sleep 1000` and `yes` running in the background froze an eval at
    1199 of 1200 episodes.

    Nothing one episode can do is worth a whole rollout, so an episode that
    fails or stops making progress is reported the way a setup failure already
    is: aborted, reward 0, flagged in metadata so it is visible afterwards
    rather than silently absorbed.
    """
    try:
        return await asyncio.wait_for(_generate(args, sample, sampling_params), timeout=EPISODE_TIMEOUT)
    except asyncio.CancelledError:
        raise
    except (asyncio.TimeoutError, TimeoutError):
        logging.getLogger(__name__).warning(
            "episode for task %s hit the %.0fs ceiling and was abandoned", sample.label, EPISODE_TIMEOUT
        )
        return _abort(args, sample, _turns_so_far(sample), et_timeout=EPISODE_TIMEOUT)
    except Exception as exc:  # noqa: BLE001 -- deliberately broad; see above
        logging.getLogger(__name__).exception("episode for task %s failed", sample.label)
        return _abort(args, sample, _turns_so_far(sample), et_error=f"{type(exc).__name__}: {exc}")


async def _generate(args, sample: Sample, sampling_params) -> Sample:
    assert not args.partial_rollout, "partial rollout is not supported by this rollout function"

    # Reset anything a previous attempt left behind.
    sample.rollout_log_probs = None
    sample.rollout_top_p_token_ids = None
    sample.rollout_top_p_token_offsets = None
    sample.response = ""
    sample.response_length = 0
    sample.loss_mask = None
    sample.reward = None

    _check_scratch_once()

    state = GenerateState(args)
    tokenizer = state.tokenizer
    chat = _chat(tokenizer)
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"

    instruction = sample.prompt if isinstance(sample.prompt, str) else str(sample.prompt)
    prompt_text = chat.prompt(tokenizer, instruction)
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    sample.tokens = list(prompt_ids)

    max_response_len = args.rollout_max_response_len
    task_dir = os.path.join(DATA_ROOT, "tasks", sample.label or "")

    episode = Episode(
        task_dir=task_dir,
        scratch_root=SCRATCH_ROOT,
        cmd_timeout=CMD_TIMEOUT,
        max_obs_chars=MAX_OBS_CHARS,
    )
    turns = 0
    finish = "max_turns"
    try:
        if not await episode.start():
            # A task whose setup cannot be reproduced must not pollute training.
            return _abort(args, sample, turns, et_setup_failed=True)

        for turns in range(1, MAX_TURNS + 1):
            # Visible to `generate`'s handlers, which unwind past `turns`.
            sample.metadata = {**(sample.metadata or {}), "round_number": turns}
            budget = max_response_len - sample.response_length
            if budget <= 0:
                sample.status = Sample.Status.TRUNCATED
                finish = "response_budget"
                break

            sp = dict(sampling_params)
            sp["max_new_tokens"] = min(MAX_NEW_TOKENS_PER_TURN, budget)
            sp["stop"] = list(dict.fromkeys(list(sp.get("stop") or []) + STOP_STRINGS))

            output = await post(url, {"input_ids": sample.tokens, "sampling_params": sp, "return_logprob": True})
            meta = output["meta_info"]
            # Every path out of this function goes through `_abort`, which sets
            # the per-sample fields slime indexes across the whole batch. The
            # abort below is the common one: at the end of a rollout slime calls
            # SGLang's `/abort_request` on everything still in flight, so a
            # straggler episode lands here while its neighbours completed
            # normally -- exactly the mix that exposes a missing key.
            if meta["finish_reason"]["type"] == "abort":
                return _abort(args, sample, turns, et_finish="sglang_abort")
            if "output_token_logprobs" not in meta:
                return _abort(args, sample, turns, et_finish="no_logprobs")

            tokens = [t[1] for t in meta["output_token_logprobs"]]
            log_probs = [t[0] for t in meta["output_token_logprobs"]]
            text = tokenizer.decode(tokens)

            # Assistant turn: trainable.
            sample.append_response_tokens(
                args, tokens=tokens, log_probs=log_probs, trainable=True, meta_info=meta, text=text
            )

            if meta["finish_reason"]["type"] == "length" and sample.response_length >= max_response_len:
                finish = "length"
                break

            kind, command = parse_action(text)
            if kind == "done":
                finish = "done"
                break
            if kind == "invalid":
                finish = "invalid_action"
                break

            observation = await episode.exec(command or "")
            obs_text = chat.observation(observation)
            obs_ids = tokenizer(obs_text, add_special_tokens=False)["input_ids"]

            # Environment turn: not trainable, and never passed log probs.
            overflow = sample.response_length + len(obs_ids) - max_response_len
            if overflow > 0:
                obs_ids = obs_ids[: max(0, len(obs_ids) - overflow)]
            if obs_ids:
                sample.append_response_tokens(args, tokens=obs_ids, trainable=False, text=obs_text)
            if overflow > 0:
                sample.status = Sample.Status.TRUNCATED
                finish = "response_budget"
                break

        reward, verifier_log = await episode.score()
        sample.reward = float(reward)

        if sample.status in (Sample.Status.PENDING, None):
            sample.status = Sample.Status.COMPLETED
        sample.metadata = {
            **(sample.metadata or {}),
            "round_number": turns,
            "et_task": sample.label,
            "et_finish": finish,
            "et_reward": float(reward),
            "et_verifier_tail": verifier_log[-200:],
        }
        return sample
    finally:
        await episode.close()


async def reward_func(args, sample: Sample, **kwargs) -> float:
    """Defensive fallback.

    `generate` already sets `sample.reward`, and slime skips the reward model
    whenever that is not None, so this is normally never called.
    """
    return float(sample.reward) if sample.reward is not None else 0.0
