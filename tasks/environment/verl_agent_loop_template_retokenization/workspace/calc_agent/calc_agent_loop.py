"""GSM8K calculator agent: verl's ToolAgentLoop plus an episode transcript and a rule reward.

Registered as ``calc_agent`` through ``calc_agent/agent_loop.yaml``. Generation is the stock
token-in/token-out loop of ``ToolAgentLoop``; on top of it this class

* keeps an OpenAI-style transcript of the episode in ``agent_data.messages``
  (system/user prompt, assistant turns with their tool calls, tool results). The reasoning
  part of each assistant turn is kept out of the transcript, as an OpenAI-compatible client
  would do, so the transcript carries only what the user sees;
* scores the episode from the final assistant message (``\\boxed{}`` vs. GSM8K ground truth);
* optionally appends a few episodes per step to ``$CALC_AGENT_DUMP_DIR/episodes.<pid>.jsonl``.
"""

import json
import logging
import os
import re
from typing import Any

from verl.experimental.agent_loop.agent_loop import AgentLoopOutput
from verl.experimental.agent_loop.tool_agent_loop import AgentData, AgentState, ToolAgentLoop
from verl.utils.rollout_trace import rollout_trace_op
from verl.utils.tokenizer.chat_template import apply_chat_template
from verl.utils.tokenizer.tokenizer import normalize_token_ids

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

_TOOL_CALL_RE = re.compile(r"<tool_call>.*?</tool_call>", re.DOTALL)
_BOXED_RE = re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")

DUMP_DIR = os.getenv("CALC_AGENT_DUMP_DIR")
DUMP_EVERY = int(os.getenv("CALC_AGENT_DUMP_EVERY", "97"))


def _to_number(text: str):
    text = text.replace(",", "").replace("$", "").replace("\\%", "").replace("%", "").strip()
    text = re.sub(r"\\text\{[^}]*\}", "", text).strip()
    match = re.search(r"-?\d+(?:\.\d+)?(?:/\d+)?", text)
    if match is None:
        return None
    token = match.group(0)
    try:
        if "/" in token:
            num, den = token.split("/")
            return float(num) / float(den)
        return float(token)
    except (ValueError, ZeroDivisionError):
        return None


def gsm8k_score(final_text: str, ground_truth: str) -> float:
    """1.0 if the last \\boxed{} in the final answer equals the ground truth, else 0.0."""
    boxed = _BOXED_RE.findall(final_text or "")
    if not boxed:
        return 0.0
    pred, gt = _to_number(boxed[-1]), _to_number(ground_truth)
    if pred is None or gt is None:
        return 0.0
    return 1.0 if abs(pred - gt) <= 1e-4 * max(1.0, abs(gt)) else 0.0


class CalcAgentLoop(ToolAgentLoop):
    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        raw_prompt = list(kwargs["raw_prompt"])
        request_id = os.urandom(16).hex()
        agent_data = AgentData(
            messages=list(raw_prompt),
            image_data=None,
            video_data=None,
            audio_data=None,
            mm_processor_kwargs=None,
            metrics={},
            request_id=request_id,
            tools_kwargs=kwargs.get("tools_kwargs", {}),
        )
        agent_data._active_tools = self.tools
        agent_data._active_tool_schemas = self.tool_schemas

        state = AgentState.PENDING
        while state != AgentState.TERMINATED:
            if state == AgentState.PENDING:
                state = await self._handle_pending_state(agent_data, sampling_params)
            elif state == AgentState.GENERATING:
                state = await self._handle_generating_state(agent_data, sampling_params)
            elif state == AgentState.PROCESSING_TOOLS:
                state = await self._handle_processing_tools_state(agent_data)
            else:
                logger.error(f"Invalid state: {state}")
                state = AgentState.TERMINATED

        # Training sequence: render the finished transcript with the model's chat template,
        # the same format the model is served with at inference time.
        prompt_ids = await self.apply_chat_template(raw_prompt, tools=self.tool_schemas)
        full_ids = await self.loop.run_in_executor(
            None,
            lambda: normalize_token_ids(
                apply_chat_template(
                    self.tokenizer,
                    agent_data.messages,
                    tools=self.tool_schemas,
                    add_generation_prompt=False,
                    tokenize=True,
                    **self.apply_chat_template_kwargs,
                )
            ),
        )
        response_ids = full_ids[len(prompt_ids) :]
        response_mask = self._assistant_token_mask(response_ids)
        # Carry the rollout log-probs over to the assistant positions, in order.
        sampled_logprobs = iter(
            lp for lp, m in zip(agent_data.response_logprobs, agent_data.response_mask, strict=False) if m
        )
        response_logprobs = [next(sampled_logprobs, 0.0) if m else 0.0 for m in response_mask]

        final_text = next(
            (m.get("content") or "" for m in reversed(agent_data.messages) if m["role"] == "assistant"), ""
        )
        score = gsm8k_score(final_text, kwargs["reward_model"]["ground_truth"])

        output = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids[: self.response_length],
            response_mask=response_mask[: self.response_length],
            response_logprobs=response_logprobs[: self.response_length] if agent_data.response_logprobs else None,
            multi_modal_data={},
            reward_score=score,
            num_turns=agent_data.user_turns + agent_data.assistant_turns + 1,
            metrics=agent_data.metrics,
            extra_fields=agent_data.extra_fields,
        )
        output.extra_fields.update(
            {
                "turn_scores": agent_data.turn_scores,
                "tool_rewards": agent_data.tool_rewards,
                "reward_extra_info": {"acc": score},
            }
        )
        self._maybe_dump(kwargs, agent_data, output, score)
        return output

    async def _handle_generating_state(
        self, agent_data: AgentData, sampling_params: dict[str, Any], ignore_termination: bool = False
    ) -> AgentState:
        state = await super()._handle_generating_state(agent_data, sampling_params, ignore_termination)
        agent_data.messages.append(await self._transcript_message(agent_data.response_ids))
        return state

    async def _transcript_message(self, turn_ids: list[int]) -> dict[str, Any]:
        """OpenAI-style assistant message for one generated turn (reasoning left out)."""
        text = await self.loop.run_in_executor(
            None, lambda: self.tokenizer.decode(turn_ids, skip_special_tokens=True)
        )
        _, calls = await self.tool_parser.extract_tool_calls(turn_ids, [t.tool_schema for t in self.tools.values()])
        if "</think>" in text:
            text = text.split("</think>")[-1]
        message: dict[str, Any] = {"role": "assistant", "content": _TOOL_CALL_RE.sub("", text).strip()}
        if calls:
            message["tool_calls"] = [
                {"type": "function", "function": {"name": c.name, "arguments": c.arguments}}
                for c in calls[: self.max_parallel_calls]
            ]
        return message

    def _assistant_token_mask(self, response_ids: list[int]) -> list[int]:
        """1 for tokens inside an assistant block (header excluded, <|im_end|> included), else 0."""
        im_start = self.tokenizer.convert_tokens_to_ids("<|im_start|>")
        im_end = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        header = self.tokenizer.encode("assistant\n", add_special_tokens=False)
        mask, inside, i = [], True, 0  # the prompt ends with the first assistant header
        while i < len(response_ids):
            tok = response_ids[i]
            if not inside and tok == im_start and response_ids[i + 1 : i + 1 + len(header)] == header:
                mask.extend([0] * (1 + len(header)))
                i += 1 + len(header)
                inside = True
                continue
            mask.append(1 if inside else 0)
            if inside and tok == im_end:
                inside = False
            i += 1
        return mask

    def _maybe_dump(self, kwargs: dict, agent_data: AgentData, output: AgentLoopOutput, score: float) -> None:
        if not DUMP_DIR:
            return
        extra_info = kwargs.get("extra_info") or {}
        index = int(extra_info.get("index", -1))
        if index % DUMP_EVERY != 0:
            return
        n_stream = len(agent_data.response_mask)
        record = {
            "global_steps": agent_data.extra_fields.get("global_steps"),
            "split": extra_info.get("split"),
            "index": index,
            "request_id": agent_data.request_id,
            "score": score,
            "num_turns": output.num_turns,
            "messages": agent_data.messages,
            # token stream as exchanged with the rollout engine
            "stream_response_ids": agent_data.prompt_ids[len(agent_data.prompt_ids) - n_stream :],
            "stream_response_mask": agent_data.response_mask,
            "stream_logprobs": agent_data.response_logprobs,
            # what this loop returns to the trainer
            "prompt_ids": output.prompt_ids,
            "response_ids": output.response_ids,
            "response_mask": output.response_mask,
            "response_logprobs": output.response_logprobs,
        }
        os.makedirs(DUMP_DIR, exist_ok=True)
        with open(os.path.join(DUMP_DIR, f"episodes.{os.getpid()}.jsonl"), "a") as fh:
            fh.write(json.dumps(record) + "\n")
