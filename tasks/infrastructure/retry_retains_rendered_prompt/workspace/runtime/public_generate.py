"""Public replacement for the private agent/RM body; not a training backend.

Source provenance and public replacement boundaries are recorded in the task
environment inventory. No private prompt, private reward service, tools,
remote service credential, or private task is included.
"""
import asyncio
import json
import logging

from miles.rollout.base_types import GenerateFnOutput
from miles.utils.types import Sample

logger = logging.getLogger(__name__)
_cpu_model = None


def public_messages(label):
    return json.loads(label)["messages"]


async def generate(input):
    sample, state = input.sample, input.state
    # Validate the generation input before rendering.
    prompt_value = sample.prompt
    try:
        if not isinstance(prompt_value, list):
            raise ValueError("Qwen planner normalization requires unrendered messages")
        prompt_text = state.tokenizer.apply_chat_template(
            prompt_value, tokenize=False, add_generation_prompt=True
        )
        sample.prompt = prompt_text
        if __import__("os").environ.get("RETRY_GENERATION_BACKEND") == "hf-cpu":
            await _generate_cpu(input)
        else:
            from miles.rollout.sglang_rollout import generate as engine_generate
            await engine_generate(input.args, sample, input.sampling_params)
        if sample.status == Sample.Status.ABORTED:
            return GenerateFnOutput(samples=sample)
        assert sample.response_length > 0, "Model generated no trainable response tokens"
        if sample.loss_mask is None:
            sample.loss_mask = [1] * sample.response_length
        assert len(sample.rollout_log_probs) == len(sample.loss_mask) == sample.response_length
        # Public deterministic reward of actual generated tokens (no private RM).
        response_tokens = sample.tokens[-sample.response_length:]
        sample.reward = sum(response_tokens) / (len(response_tokens) * state.tokenizer.vocab_size)
    except ValueError as exc:
        if str(exc) != "Qwen planner normalization requires unrendered messages":
            raise
        # Report input normalization failures as aborted generation.
        logger.warning("Public generation aborted: %s (group=%s sample=%s)", exc, sample.group_index, sample.index)
        sample.status = Sample.Status.ABORTED
        sample.metadata["abort_reason"] = str(exc)
    return GenerateFnOutput(samples=sample)


async def _generate_cpu(input):
    """Real model inference for one-GPU smoke; Miles still owns GPU training."""
    import torch
    from transformers import AutoModelForCausalLM
    global _cpu_model
    if _cpu_model is None:
        torch.set_num_threads(1)
        _cpu_model = AutoModelForCausalLM.from_pretrained(
            input.args.hf_checkpoint, local_files_only=True, torch_dtype=torch.float32,
            attn_implementation="eager",
        ).eval()
    sample, tokenizer = input.sample, input.state.tokenizer
    prompt_ids = tokenizer.encode(sample.prompt, add_special_tokens=False)
    # Serial CPU generation stays on the producer event loop; no executor or mocks.
    ids = torch.tensor([prompt_ids], dtype=torch.long)
    rng = torch.Generator(device="cpu").manual_seed(101 + sample.index)
    log_probs, generated = [], []
    with torch.inference_mode():
        for _ in range(input.sampling_params["max_new_tokens"]):
            logits = _cpu_model(ids).logits[:, -1, :].float()
            probs = torch.log_softmax(logits / input.args.rollout_temperature, dim=-1)
            token = torch.multinomial(probs.exp(), 1, generator=rng)
            log_probs.append(probs.gather(-1, token).item())
            generated.append(token.item())
            ids = torch.cat((ids, token), dim=1)
    sample.tokens = prompt_ids + generated
    sample.response_length = len(generated)
    sample.response = tokenizer.decode(generated, skip_special_tokens=False)
    sample.rollout_log_probs = log_probs
    sample.status = Sample.Status.COMPLETED
    await asyncio.sleep(0)
