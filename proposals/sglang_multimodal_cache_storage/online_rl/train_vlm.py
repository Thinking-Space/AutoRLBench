"""Online constrained-generation GRPO with a frozen SmolVLM and a trainable LM-head adapter."""

import argparse
import hashlib
import random
import sys
from pathlib import Path

import torch
import transformers
from torch import nn
from transformers import AutoConfig, AutoProcessor, Idefics3ForConditionalGeneration

from beacon_env import BATCH, GROUP, PROMPTS, SEED, beacon, policy_digest, report
from cache_backend import RolloutEmbeddingCache, image_key

MODEL = "HuggingFaceTB/SmolVLM-256M-Instruct"
REVISION = "d039ab0d4b9311b617e8ffec1d06c8464364aeb3"
WEIGHT_SHA256 = "74dea5904032e5ae99a2e0eef5179e6ac0f1dedc3ab0c7c2a5d4d387c843203e"
ANSWERS = (" No", " Yes")
TEMPERATURE = 2.5
HELDOUT_SCENES = 16


def verify_weights(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    if h.hexdigest() != WEIGHT_SHA256:
        raise ValueError(f"model.safetensors must match {MODEL}@{REVISION}")


class FrozenVLM:
    def __init__(self, directory, device):
        verify_weights(directory / "model.safetensors")
        self.device = device
        self.processor = AutoProcessor.from_pretrained(directory, local_files_only=True)
        dtype = {"cpu": torch.float32, "mps": torch.float16, "cuda": torch.bfloat16}[device.type]
        config = AutoConfig.from_pretrained(directory, local_files_only=True)
        # This single-device experiment has no distributed tensor-parallel workers.
        # Transformers 4.52 validates default Llama TP hints against an unavailable
        # style registry on the shared image's PyTorch 2.4.
        config.text_config.base_model_tp_plan = {}
        self.model = Idefics3ForConditionalGeneration.from_pretrained(
            directory, config=config, local_files_only=True, torch_dtype=dtype,
            attn_implementation="sdpa").eval().requires_grad_(False).to(device)
        self.tokens = self.model.model.image_seq_len
        self.channels = self.model.config.text_config.hidden_size
        self.edge = self.model.config.vision_config.image_size
        self.processor.image_processor.do_image_splitting = False
        self.processor.image_processor.size = {"longest_edge": self.edge}
        self.tokenizer = self.processor.tokenizer
        self.tokenizer.padding_side = "right"
        action_ids = [self.tokenizer.encode(text, add_special_tokens=False) for text in ANSWERS]
        if any(len(ids) != 1 for ids in action_ids) or action_ids[0] == action_ids[1]:
            raise ValueError("each legal answer must be a distinct single generated token")
        self.action_ids = torch.tensor([ids[0] for ids in action_ids], device=device)
        # Text is constant for each question. Let the checkpoint processor insert
        # the actual image placeholders and chat delimiters once for each template.
        self.texts = [self.processor.apply_chat_template(
            [{"role": "user", "content": [{"type": "image"}, {"type": "text",
              "text": question + " Answer Yes or No."}]}],
            tokenize=False, add_generation_prompt=True) for question in PROMPTS]
        placeholder = beacon(-1)[0]
        batch = self.processor(
            images=[[placeholder] for _ in PROMPTS], text=self.texts, add_special_tokens=False,
            padding=True, return_tensors="pt")
        self.ids = batch.input_ids.to(device)
        self.mask = batch.attention_mask.to(device)
        if self.tokens != 64 or not (self.ids.eq(self.model.config.image_token_id).sum(1) == self.tokens).all():
            raise ValueError("expected one native 64-token image block per prompt")
        self.calls = self.n_images = 0

    def prepare_pixels(self, images):
        pixels = self.processor.image_processor([[image] for image in images], return_tensors="pt")
        if tuple(pixels.pixel_values.shape[1:]) != (1, 3, self.edge, self.edge):
            raise ValueError("expected a single square RGB input per observation")
        return {"pixel_values": pixels.pixel_values.to(self.device, self.model.dtype),
                "pixel_attention_mask": pixels.pixel_attention_mask.to(self.device)}

    @torch.no_grad()
    def encode(self, images):
        features = self.model.model.get_image_features(**self.prepare_pixels(images))
        # Transformers 4 returned a tensor; 5 exposes the connector output on pooler_output.
        tokens = features if isinstance(features, torch.Tensor) else features.pooler_output
        self.calls += 1
        self.n_images += len(images)
        return tokens.reshape(len(images) * self.tokens, self.channels).contiguous()

    @torch.no_grad()
    def decode_context(self, joined_tokens, prompts):
        ids, mask = self.ids[prompts], self.mask[prompts]
        # HF injects the cached connector output at the <image> token positions
        # and runs the entire pretrained autoregressive text decoder.
        result = self.model.model(input_ids=ids, attention_mask=mask,
                                  image_hidden_states=joined_tokens, use_cache=False, return_dict=True)
        row = torch.arange(len(prompts), device=self.device)
        last = result.last_hidden_state[row, mask.long().sum(-1) - 1]
        base_logits = self.model.lm_head(last).index_select(-1, self.action_ids)
        return last.float(), base_logits.float()

    @torch.no_grad()
    def check_native_forward(self):
        scenes = [135, 246]
        images = [beacon(scene)[0] for scene in scenes]
        prompts = torch.arange(len(scenes), device=self.device)
        direct = self.model(input_ids=self.ids, attention_mask=self.mask, use_cache=False,
                            return_dict=True, **self.prepare_pixels(images))
        last = direct.logits[prompts, self.mask.long().sum(-1) - 1]
        reference = last.index_select(-1, self.action_ids).float()
        _, split = self.decode_context(self.encode(images), prompts)
        error = (reference - split).abs().max().item()
        # A full-sequence LM head and a last-position LM head can select
        # different matmul kernels. Allow one relative dtype epsilon at peak logit.
        limit = max(1e-4, reference.abs().max().item() * torch.finfo(self.model.dtype).eps)
        if not error <= limit:
            raise ValueError(f"split VLM path differs from native forward: max {error}, limit {limit}")
        return {"scene_ids": scenes, "n_logits": reference.numel(), "max_abs_diff": error,
                "tolerance": limit, "native_logits": reference.tolist(), "split_logits": split.tolist()}


class LanguagePolicy(nn.Module):
    """A trainable residual on the two admissible vocabulary logits at the last language position."""

    def __init__(self, hidden_size):
        super().__init__()
        self.adapter = nn.Linear(hidden_size, len(ANSWERS), bias=False)
        nn.init.zeros_(self.adapter.weight)

    def forward(self, hidden, base_logits):
        return ((base_logits + self.adapter(hidden)) / TEMPERATURE).log_softmax(-1)


def encode_scenes(vlm, scene_ids, cache=None):
    images, sides = zip(*(beacon(scene) for scene in scene_ids))
    if cache is None:
        joined = vlm.encode(images)
    else:
        joined = cache.encode(images, [image_key(im) for im in images], [vlm.tokens] * len(images),
                              vlm.encode, vlm.device, requests_per_image=True)
    return joined, torch.tensor(sides, device=vlm.device)


def heldout_inputs(vlm):
    batches = []
    for start in range(0, HELDOUT_SCENES, BATCH):
        scenes = [-(1000 + i) for i in range(start, start + BATCH)]
        tokens, sides = encode_scenes(vlm, scenes)
        for question in range(len(PROMPTS)):
            prompts = torch.full_like(sides, question)
            batches.append((*vlm.decode_context(tokens, prompts), sides.eq(prompts)))
    return batches


@torch.no_grad()
def accuracy(policy, batches):
    correct = sum(policy(h, base).argmax(-1).eq(target).sum().item() for h, base, target in batches)
    return correct / sum(target.numel() for _, _, target in batches)


def run(args):
    torch.set_num_threads(2)
    torch.manual_seed(SEED)
    if args.device == "cuda":
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device(args.device)
    rng = torch.Generator(device="cpu").manual_seed(SEED + 1)
    prompt_rng = random.Random(SEED + 2)
    vlm = FrozenVLM(args.model_dir, device)
    native_check = vlm.check_native_forward()
    policy = LanguagePolicy(vlm.channels).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.0008, foreach=False)
    capacity = 5 * vlm.tokens * vlm.channels * vlm.model.lm_head.weight.element_size()
    cache = RolloutEmbeddingCache(capacity, mode=args.mode, source_dir=args.source_dir)
    heldout = heldout_inputs(vlm)
    before_accuracy = accuracy(policy, heldout)
    print(f"{args.mode} {MODEL} {device} | initial held-out accuracy {before_accuracy:.3f} "
          f"| native-logit max difference {native_check['max_abs_diff']:.5g}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as log:
        report(log, type="meta", mode=args.mode, device=args.device, gpu=(
            torch.cuda.get_device_name() if device.type == "cuda" else args.device),
            seed=SEED, batch_size=BATCH, group_size=GROUP, initial_policy_digest=policy_digest(policy),
            model=MODEL, revision=REVISION, weights_sha256=WEIGHT_SHA256,
            transformers=transformers.__version__, torch=str(torch.__version__), cuda=torch.version.cuda,
            dtype=str(vlm.model.dtype), channels=vlm.channels, tokens_per_image=vlm.tokens,
            capacity_bytes=capacity, trainable_parameters=sum(p.numel() for p in policy.parameters()),
            frozen_parameters=sum(p.numel() for p in vlm.model.parameters()),
            initial_heldout_accuracy=before_accuracy, native_forward_check=native_check,
            rendered_prompts=vlm.texts,
            legal_answers=ANSWERS, legal_token_ids=vlm.action_ids.tolist(), temperature=TEMPERATURE)
        prior = []
        for step in range(1, args.steps + 1):
            reused = prior[-4:] if (step - 1) % 4 else []
            scenes = reused + [SEED*100 + step*BATCH + i for i in range(BATCH - len(reused))]
            prior = scenes
            questions = [prompt_rng.randrange(len(PROMPTS)) for _ in scenes]
            prompts = torch.tensor(questions, device=device)
            before_calls, before_images = vlm.calls, vlm.n_images
            joined, sides = encode_scenes(vlm, scenes, cache)
            h, base = vlm.decode_context(joined, prompts)
            del joined
            measurement = cache.snapshot()
            with torch.no_grad():
                old = policy(h, base)
                draws = torch.rand((BATCH, GROUP), generator=rng).to(device)
                choices = (draws < old[:, 1:2].exp()).long()
                generated_ids = vlm.action_ids[choices]
                target = sides.eq(prompts)
                rewards = choices.eq(target[:, None]).float()
                old_logp = old.gather(1, choices)
                std = rewards.std(1, keepdim=True, unbiased=False)
                advantage = (rewards - rewards.mean(1, keepdim=True)) / std.clamp_min(1e-6)
            before = policy.adapter.weight.detach().clone()
            logp = policy(h, base).gather(1, choices)
            ratio = (logp - old_logp).exp()
            loss = -torch.minimum(ratio * advantage, ratio.clamp(0.8, 1.2) * advantage).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad = policy.adapter.weight.grad.norm().item()
            optimizer.step()
            delta = (policy.adapter.weight.detach() - before).norm().item()
            mean = rewards.mean().item()
            report(log, type="step", step=step, scene_ids=scenes, prompt_ids=questions,
                   actions=generated_ids.flatten().tolist(), rewards=rewards.long().flatten().tolist(),
                   logprobs=old_logp.flatten().tolist(), reward_mean=mean, loss_before_update=loss.item(),
                   grad_norm=grad, weight_delta=delta, valid_groups=int((std > 0).sum().item()),
                   policy_digest=policy_digest(policy), encoder_calls=vlm.calls-before_calls,
                   encoder_images=vlm.n_images-before_images, cache=measurement)
            print(f"step={step} reward={mean:.3f} grad={grad:.4g} "
                  f"retained_bytes={measurement['storage_bytes']} budget={capacity}", flush=True)
        final_accuracy = accuracy(policy, heldout)
        report(log, type="final", steps=args.steps, heldout_accuracy=final_accuracy,
               heldout_decisions=HELDOUT_SCENES * len(PROMPTS), policy_digest=policy_digest(policy))
        print(f"final held-out accuracy {final_accuracy:.3f}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True, help=f"local snapshot of {MODEL}@{REVISION}")
    parser.add_argument("--source-dir", type=Path, help="cache directory for the pinned SGLang sources")
    parser.add_argument("--device", choices=("cuda", "mps", "cpu"), default="cuda")
    parser.add_argument("--mode", choices=("buggy", "fixed"), required=True)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.steps < 1:
        sys.exit("--steps must be positive")
    run(args)
