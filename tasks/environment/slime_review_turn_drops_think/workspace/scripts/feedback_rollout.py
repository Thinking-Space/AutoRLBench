"""Answer-then-review math rollout for slime GRPO.

Turn 1: the policy solves a MATH problem. Turn 2: the user asks it to check its own solution
(nothing about correctness is revealed) and give a final answer, which may revise the first.
The trajectory reward is 1 if the final turn's boxed answer is correct, 0 otherwise.

Plugged into slime with
    --custom-generate-function-path feedback_rollout.generate
    --custom-rm-path feedback_rollout.reward_func
"""

from typing import Any

from slime.rollout.rm_hub.math_utils import extract_answer, grade_answer_verl
from slime.rollout.sglang_rollout import GenerateState
from slime.utils.http_utils import post
from slime.utils.types import Sample

NUM_TURNS = 2

INSTRUCTION = "\n\nPlease reason step by step, and put your final answer within \\boxed{}."
REVIEW = (
    "Please review your solution carefully. If you find a mistake, correct it. "
    "Then give your final answer within \\boxed{}."
)
REVIEW_MISSING = (
    "You did not give a final answer. Please finish solving the problem "
    "and give your final answer within \\boxed{}."
)


def _final_answer_part(text: str, finish: str) -> str:
    """The text after the reasoning block; a turn that ended without closing it counts as a whole,
    unless it was cut off by the length limit (then there is no answer)."""
    if "</think>" in text:
        return text.split("</think>")[-1]
    return text if finish == "stop" else ""


def _render(tokenizer, messages, add_generation_prompt):
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt)


def _assistant_loss_mask(tokenizer, token_ids: list[int]) -> list[int]:
    """1 on the tokens of every assistant message (content + closing <|im_end|>), 0 elsewhere."""
    im_start = tokenizer.convert_tokens_to_ids("<|im_start|>")
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    header = tokenizer("assistant\n", add_special_tokens=False)["input_ids"]
    mask = [0] * len(token_ids)
    i = 0
    while i < len(token_ids):
        if token_ids[i] == im_start and token_ids[i + 1 : i + 1 + len(header)] == header:
            j = i + 1 + len(header)
            while j < len(token_ids) and token_ids[j] != im_end:
                mask[j] = 1
                j += 1
            if j < len(token_ids):
                mask[j] = 1  # the policy emits <|im_end|> itself
            i = j + 1
        else:
            i += 1
    return mask


async def generate(args, sample: Sample, sampling_params: dict[str, Any]) -> Sample:
    assert not args.partial_rollout, "partial rollout is not supported by this rollout"
    state = GenerateState(args)
    tokenizer = state.tokenizer
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
    max_context = args.rollout_max_context_len or args.max_tokens_per_gpu

    messages = [{"role": "user", "content": sample.prompt + INSTRUCTION}]
    attempts = []  # one record per assistant turn
    finish = "stop"
    for k in range(NUM_TURNS):
        prompt_ids = tokenizer(_render(tokenizer, messages, True), add_special_tokens=False)["input_ids"]
        budget = max_context - len(prompt_ids)
        if budget <= 0:
            finish = "length"
            break
        params = dict(sampling_params)
        params["max_new_tokens"] = min(sampling_params["max_new_tokens"], budget)
        output = await post(url, {"input_ids": prompt_ids, "sampling_params": params})
        finish = output["meta_info"]["finish_reason"]["type"]
        if finish == "abort":
            sample.status = Sample.Status.ABORTED
            return sample

        text = output["text"]
        messages.append({"role": "assistant", "content": text})
        answer_part = _final_answer_part(text, finish)
        correct = bool(grade_answer_verl(answer_part, sample.label))
        attempts.append(
            {
                "turn": k + 1,
                "correct": correct,
                "finish": finish,
                "gen_tokens": output["meta_info"]["completion_tokens"],
                "pred": extract_answer(answer_part),
            }
        )
        if k == NUM_TURNS - 1:
            break
        has_answer = extract_answer(answer_part) is not None
        messages.append({"role": "user", "content": REVIEW if has_answer else REVIEW_MISSING})

    # Training sample: the whole conversation, rendered with the model's chat template.
    prompt_ids = tokenizer(_render(tokenizer, messages[:1], True), add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(_render(tokenizer, messages, False), add_special_tokens=False)["input_ids"]
    assert full_ids[: len(prompt_ids)] == prompt_ids, "rendered conversation does not extend the prompt"
    full_ids = full_ids[:max_context]
    mask = _assistant_loss_mask(tokenizer, full_ids)

    sample.tokens = full_ids
    sample.response_length = len(full_ids) - len(prompt_ids)
    sample.loss_mask = mask[len(prompt_ids) :]
    sample.response = tokenizer.decode(full_ids[len(prompt_ids) :])
    sample.rollout_log_probs = None

    final_correct = bool(attempts) and len(attempts) == NUM_TURNS and attempts[-1]["correct"]
    sample.reward = 1.0 if final_correct else 0.0
    sample.metadata = dict(sample.metadata or {})
    sample.metadata.update({"turns": attempts, "num_turns": len(attempts)})
    sample.status = Sample.Status.TRUNCATED if finish == "length" else Sample.Status.COMPLETED
    return sample


async def reward_func(args, sample: Sample, **kwargs):
    return sample.reward
