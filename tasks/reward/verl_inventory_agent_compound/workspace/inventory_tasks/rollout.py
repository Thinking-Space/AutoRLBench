"""Token-preserving inventory trajectories for online veRL rollout."""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
import json
from pathlib import Path
from time import monotonic
from typing import Any
from uuid import uuid4

from .domain import InventoryEpisode


TRAIN_SPLIT = "inventory_train"
NORMAL_SPLIT = "inventory_random_split"
CLEAN_SPLIT = "inventory_clean"


def _render(tokenizer, messages, *, generation: bool, template_options: dict[str, Any]) -> list[int]:
    return list(map(int, tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=generation, **template_options
    )))


class ObservationEncoder:
    """Encode new context without changing any token already sampled by the actor."""

    def __init__(self, tokenizer, options: dict[str, Any]):
        self.tokenizer = tokenizer
        self.options = options
        if tokenizer.eos_token_id is None:
            raise ValueError("Inventory rollout requires an assistant end-of-turn token")
        self.eos_token = int(tokenizer.eos_token_id)
        self.newline = list(map(int, tokenizer.encode("\n", add_special_tokens=False)))
        self.anchor = {"role": "system", "content": "Inventory tool protocol."}
        self.anchor_ids = _render(tokenizer, [self.anchor], generation=False, template_options=options)

    def after(self, sampled: list[int], observation: str) -> list[int]:
        user = {"role": "user", "content": "Inventory tool result:\n" + observation}
        rendered = _render(self.tokenizer, [self.anchor, user], generation=True, template_options=self.options)
        if rendered[:len(self.anchor_ids)] != self.anchor_ids:
            raise ValueError("Chat template does not support appending an inventory tool result")
        eot = [] if sampled[-1] == self.eos_token else [self.eos_token]
        return eot + self.newline + rendered[len(self.anchor_ids):]


_TRACE_COUNTS: Counter[tuple[str, str, str]] = Counter()


def _save_trace(directory: str | None, limit: int, trace: dict[str, Any]) -> None:
    if not directory or limit <= 0:
        return
    result = trace["result"]
    bucket = "success" if result["strict_success"] else "failure"
    if any(event.get("response") == "timeout" for event in result["events"]):
        bucket = "response_timeout"
    key = (str(directory), trace["split"], bucket)
    if _TRACE_COUNTS[key] >= limit:
        return
    _TRACE_COUNTS[key] += 1
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    (path / f'{trace["trajectory_id"]}.json').write_text(json.dumps(trace, ensure_ascii=False, indent=2) + "\n")


class InventoryRollout:
    """Runs using only the veRL loop's config, tokenizer and server manager."""

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> dict[str, Any]:
        config = self.config.inventory
        metadata = kwargs.get("extra_info")
        if not isinstance(metadata, Mapping):
            raise ValueError("Missing required inventory dataset metadata")
        split = metadata["split"]
        if split not in {TRAIN_SPLIT, NORMAL_SPLIT, CLEAN_SPLIT} or kwargs.get("data_source") != split:
            raise ValueError(f"Unexpected inventory data source/split: {kwargs.get('data_source')!r}/{split!r}")
        scenario = json.loads(metadata["scenario_json"])
        for key in ("scenario_id", "parent_scenario_id"):
            if scenario[key] != metadata[key]:
                raise ValueError(f"Inventory metadata does not name the encoded scenario: {key}")
        occurrence_id = metadata["occurrence_id"]
        messages = list(kwargs["raw_prompt"])
        if not messages or any(not isinstance(message.get("content"), str) for message in messages):
            raise ValueError("Missing required inventory chat messages")

        limits = self.config.actor_rollout_ref.rollout
        prompt_cap, response_cap = int(limits.prompt_length), int(limits.response_length)
        action_cap = int(config.get("max_action_tokens", 96))
        max_turns = int(config.get("max_turns", 8))
        if min(prompt_cap, response_cap, action_cap, max_turns) < 1:
            raise ValueError("Inventory prompt, response, action and turn limits must be positive")

        template_options = dict(self.config.data.get("apply_chat_template_kwargs") or {})
        prompt_ids = _render(self.tokenizer, messages, generation=True, template_options=template_options)
        if not (0 < len(prompt_ids) <= prompt_cap):
            raise ValueError(f"Full inventory prompt has {len(prompt_ids)} tokens; capacity is {prompt_cap}")

        observations = ObservationEncoder(self.tokenizer, template_options)
        episode = InventoryEpisode(scenario, training=split == TRAIN_SPLIT,
                                   max_turns=max_turns)
        trajectory_id = uuid4().hex
        response_ids: list[int] = []
        response_mask: list[int] = []
        actions = []
        num_turns = len(messages)
        generation_seconds = tool_seconds = 0.0
        stop_reason = "budget"
        for turn in range(max_turns):
            remaining = response_cap - len(response_ids)
            if remaining <= 0:
                break
            turn_tokens = min(action_cap, remaining)
            params = dict(sampling_params, max_new_tokens=turn_tokens)
            started = monotonic()
            output = await self.server_manager.generate(
                request_id=trajectory_id, prompt_ids=prompt_ids + response_ids, sampling_params=params,
            )
            generation_seconds += monotonic() - started
            sampled = list(map(int, output.token_ids))
            if not sampled or len(sampled) > turn_tokens:
                raise ValueError("Inventory vLLM output is empty or exceeds the per-turn token limit")

            response_ids.extend(sampled)
            response_mask.extend([1] * len(sampled))
            command = self.tokenizer.decode(sampled, skip_special_tokens=True).strip()
            num_turns += 1
            if not command:
                actions.append({"turn": turn, "policy_action": command})
                stop_reason = "empty_action"
                break

            started = monotonic()
            observation = episode.step(command)
            tool_seconds += monotonic() - started
            actions.append({"turn": turn, "policy_action": command, "observation": observation})
            if episode.done:
                stop_reason = "episode_done"
                break

            # Only retain tool context if another sampled assistant token fits.
            # This also places veRL's terminal rm_score at a sampled position.
            context = observations.after(sampled, observation)
            if len(response_ids) + len(context) >= response_cap or turn + 1 == max_turns:
                break
            response_ids.extend(context)
            response_mask.extend([0] * len(context))
            num_turns += 1

        state = episode.finalize()
        assert 0 < len(response_ids) == len(response_mask) <= response_cap and response_mask[-1] == 1
        metrics = {
            "acc": float(episode.success),
            "strict_success": float(episode.success),
            "ledger_success": float(episode.ledger_success),
            "receipt_success": float(episode.receipt_success),
            "progress_reward": float(episode.progress_reward),
            "terminal_reward": float(episode.terminal_reward),
        }
        trace = {"trajectory_id": trajectory_id, "split": split, "occurrence_id": occurrence_id,
                 "scenario_id": scenario["scenario_id"], "parent_scenario_id": scenario["parent_scenario_id"],
                 "stop_reason": stop_reason, "actions": actions,
                 "result": {**state, **metrics}}
        _save_trace(config.get("trace_dir"), int(config.get("trace_limit_per_worker", 4)), trace)
        extra_fields: dict[str, Any] = {"reward_extra_info": metrics}
        if config.get("capture_trace", False):
            extra_fields["inventory_trace"] = trace
        return {
            "prompt_ids": prompt_ids,
            "response_ids": response_ids,
            "response_mask": response_mask,
            "num_turns": num_turns,
            "reward_score": float(episode.reward),
            "metrics": {"generate_sequences": generation_seconds, "tool_calls": tool_seconds},
            "extra_fields": extra_fields,
        }
