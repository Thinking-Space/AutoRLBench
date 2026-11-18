"""Two-way agent/tool interaction for Slime's native SGLang rollout worker."""
import asyncio
import re

from slime.rollout.sglang_rollout import GenerateState
from slime.utils.http_utils import post
from slime.utils.types import Sample

LOOKUP = re.compile(r"\bLOOKUP\s*:\s*([A-Z]{2}-[0-9]{3})\b", re.I)
FINAL = re.compile(r"\bFINAL\s*:\s*([a-z]+)", re.I)


async def get_ticket(sample, ticket, evaluation, samples_per_prompt):
    slot = sample.index % samples_per_prompt
    maintenance = not evaluation and sample.group_index % 5 != 0 and slot == 0
    if maintenance:
        await asyncio.sleep(0.08)
    if ticket != sample.metadata["ticket"]:
        return f"No catalog entry for {ticket}."
    return f"Current catalog entry for {ticket}: owner team {sample.label}."


async def generate(args, sample, sampling_params, evaluation=False):
    state = GenerateState(args)
    tokenizer = state.tokenizer
    sample.tokens = tokenizer.encode(sample.prompt, add_special_tokens=False)
    sample.loss_mask = []
    marker = tokenizer.convert_tokens_to_ids("<|im_end|>")
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
    steps = []
    observed = False
    for _ in range(3):
        params = dict(sampling_params, max_new_tokens=96)
        output = await post(url, {"input_ids": sample.tokens, "sampling_params": params,
                                  "return_logprob": True})
        info = output["meta_info"]
        tokens_and_scores = info.get("output_token_logprobs") or []
        if not tokens_and_scores:
            raise RuntimeError(f"Missing actor tokens for finish reason {info['finish_reason']['type']}")
        ids = [item[1] for item in tokens_and_scores]
        text = tokenizer.decode(ids, skip_special_tokens=True)
        sample.append_response_tokens(args, tokens=ids, log_probs=[item[0] for item in tokens_and_scores],
                                      trainable=True, meta_info=info, text=text)
        steps.append({"assistant": text})
        request = LOOKUP.search(text)
        answer = FINAL.search(text)
        if not request or (answer and answer.start() < request.start()):
            break

        ticket = request[1].upper()
        try:
            result = await asyncio.wait_for(
                get_ticket(sample, ticket, evaluation, args.n_samples_per_prompt), 0.03,
            )
        except TimeoutError:
            steps.append({"catalog": "timeout", "ticket": ticket})
            sample.status = Sample.Status.ABORTED
            sample.remove_sample = True
            sample.loss_mask = [0] * sample.response_length
            sample.reward = 0.0
            sample.metadata.update(events=steps, raw_reward=0.0, verified=False)
            return sample

        steps.append({"catalog": result, "ticket": ticket})
        observed |= ticket == sample.metadata["ticket"]
        closing = "" if ids[-1] == marker else "<|im_end|>"
        observation = (f"{closing}\n<|im_start|>user\n<tool_response>\n{result}\n"
                       "</tool_response>\n<|im_end|>\n<|im_start|>assistant\n")
        sample.append_response_tokens(args, tokens=tokenizer.encode(observation, add_special_tokens=False),
                                      trainable=False, text=observation)

    verified = bool(answer and observed and answer[1].lower() == sample.label)
    sample.reward = float(verified) if evaluation else (1.0 if verified else 0.6 if answer else 0.0)
    sample.metadata.update(events=steps, raw_reward=sample.reward, verified=verified)
    return sample
