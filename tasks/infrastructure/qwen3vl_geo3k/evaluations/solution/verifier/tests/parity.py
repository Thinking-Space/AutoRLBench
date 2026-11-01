#!/usr/bin/env python3
"""Fresh vLLM capture and submitted verl/FSDP scoring; no reference-source overlay."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import sys

from metrics import repair_verdict

SPEC_PATH = Path(__file__).resolve().parents[1] / "resources/grader_spec.json"


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def setup(workspace):
    repo = Path(workspace).resolve() / "verl"
    sys.path.insert(0, str(repo))
    import verl
    if not Path(verl.__file__).resolve().is_relative_to(repo):
        raise RuntimeError("Imported verl does not belong to the submitted workspace")
    return repo


def versions():
    return {p: importlib.metadata.version(p) for p in ("torch", "transformers", "vllm", "tensordict")}


def model_path(args):
    return str(Path(args.model or Path(args.workspace) / "models/Qwen3-VL-8B-Instruct").resolve())


def encode(processor, case, case_root, spec):
    from PIL import Image
    images = [Image.open(case_root / p).convert("RGB") for p in case["images"]]
    prompt = processor.apply_chat_template(case["messages"], tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[prompt], images=images or None, return_tensors="pt", **spec["mm_processor_kwargs"])
    return inputs, images


def capture(args):
    # CUDA may be initialized while resolving the model/processor or backend.
    # The engine subprocess must start with a fresh interpreter in that case.
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams

    spec = read(SPEC_PATH)
    model = model_path(args)
    pins = read(SPEC_PATH.parent / "model_pins.json")
    for name, expected in pins["files"].items():
        path = Path(model) / name
        if path.stat().st_size != expected["bytes"] or digest(path) != expected["sha256"]:
            raise RuntimeError(f"Verifier requires the immutable Qwen3-VL-8B checkpoint: {name}")
    case_root = Path(args.assets or args.workspace) / "data/parity"
    cases = read(case_root / "cases.json")["cases"]
    if {case["name"] for case in cases} != set(spec["cases"]):
        raise ValueError("Unexpected case manifest")
    processor = AutoProcessor.from_pretrained(model, local_files_only=True)
    prepared = [encode(processor, case, case_root, spec) for case in cases]
    prompts = []
    for case, (_, images) in zip(cases, prepared, strict=True):
        # vLLM performs multimodal placeholder expansion itself. Supplying
        # already-expanded HF IDs would expand image placeholders a second time.
        prompt = {"prompt": processor.apply_chat_template(
            case["messages"], tokenize=False, add_generation_prompt=True)}
        if images:
            prompt["multi_modal_data"] = {"image": images}
        prompts.append(prompt)
    # Validate preprocessing before allocating the model. ModelConfig/processor
    # construction loads metadata and tokenizers, not the 8B weight tensors.
    import torch
    from vllm.config import ModelConfig
    from vllm.multimodal import MULTIMODAL_REGISTRY
    from vllm.multimodal.processing.context import TimingContext
    from vllm.multimodal.processing.inputs import ProcessorInputs
    processing_config = ModelConfig(
        model=model, tokenizer=model, dtype="bfloat16", max_model_len=4096,
        enforce_eager=True, limit_mm_per_prompt={"image": 2},
        mm_processor_cache_gb=0, mm_processor_kwargs=spec["mm_processor_kwargs"])
    rollout_processor = MULTIMODAL_REGISTRY.create_processor(processing_config)
    preflight = []
    for case, (inputs, images), prompt in zip(cases, prepared, prompts, strict=True):
        processed = rollout_processor.apply(ProcessorInputs(
            prompt["prompt"], rollout_processor.info.parse_mm_data(
                {"image": images} if images else {}),
            hf_processor_mm_kwargs=spec["mm_processor_kwargs"]), TimingContext(enabled=False))
        check = {"name": case["name"],
                 "prompt_ids_match": processed["prompt_token_ids"] == inputs["input_ids"][0].tolist(),
                 "image_grids_match": True, "bf16_pixel_values_match": True}
        if images:
            items = processed["mm_kwargs"]["image"]
            grids = [item["image_grid_thw"].data.tolist() for item in items]
            pixels = torch.cat([item["pixel_values"].data for item in items], dim=0)
            hf_pixels = inputs["pixel_values"].to(torch.bfloat16)
            check.update(image_grids_match=grids == inputs["image_grid_thw"].tolist(),
                         bf16_pixel_values_match=torch.equal(pixels.to(torch.bfloat16), hf_pixels),
                         image_grid_thw=grids,
                         rollout_pixel_dtype=str(pixels.dtype), hf_pixel_dtype=str(inputs["pixel_values"].dtype))
        preflight.append(check)
    preflight_path = Path(args.out).with_suffix(".input_preflight.json")
    write(preflight_path, {"records": preflight})
    if not all(row[key] for row in preflight for key in (
            "prompt_ids_match", "image_grids_match", "bf16_pixel_values_match")):
        raise RuntimeError(f"HF/vLLM preprocessing differs; see {preflight_path}")
    print("HF/vLLM preflight: identical prompt IDs, grids, and BF16 pixels for all six cases.", flush=True)
    engine = LLM(model=model, dtype="bfloat16", tensor_parallel_size=args.tp,
                 max_model_len=4096, max_num_seqs=1, gpu_memory_utilization=0.7,
                 enforce_eager=True, enable_prefix_caching=False, seed=17,
                 generation_config="vllm", logprobs_mode="raw_logprobs",
                 limit_mm_per_prompt={"image": 2}, mm_processor_kwargs=spec["mm_processor_kwargs"])
    tokens = spec["response_tokens"]
    sampling = SamplingParams(temperature=1.0, top_p=1.0, top_k=-1, seed=17,
                              max_tokens=tokens, min_tokens=tokens, ignore_eos=True,
                              logprobs=1, detokenize=False)
    outputs = engine.generate(prompts, sampling)
    alignment = []
    for case, (inputs, _), output in zip(cases, prepared, outputs, strict=True):
        expected_ids = inputs["input_ids"][0].tolist()
        actual_ids = list(output.prompt_token_ids or [])
        alignment.append({"name": case["name"], "matches": actual_ids == expected_ids,
                          "hf_prompt_ids": expected_ids, "vllm_prompt_ids": actual_ids})
    alignment_path = Path(args.out).with_suffix(".prompt_alignment.json")
    write(alignment_path, {"input_format": "raw_chat_text", "records": alignment})
    mismatched = [row["name"] for row in alignment if not row["matches"]]
    if mismatched:
        raise RuntimeError(f"vLLM/HF prompt ID mismatch in {mismatched}; see {alignment_path}")
    rows = []
    for case, (inputs, _), output in zip(cases, prepared, outputs, strict=True):
        prompt_ids = inputs["input_ids"][0].tolist()
        generated = output.outputs[0]
        response_ids = list(generated.token_ids)
        if len(response_ids) != tokens:
            raise RuntimeError("Incomplete rollout")
        logps = [float(scores[token].logprob) for token, scores in zip(response_ids, generated.logprobs, strict=True)]
        rows.append({**case, "prompt_ids": prompt_ids, "response_ids": response_ids, "vllm_logps": logps,
                     "image_sha256": [digest(case_root / p) for p in case["images"]],
                     "image_grid_thw": inputs["image_grid_thw"].tolist() if case["images"] else None})
    # Both score processes must use the same model path and unchanged weight files.
    weights = sorted(Path(model).glob("*.safetensors"))
    if not weights:
        raise RuntimeError("Missing model safetensors")
    write(args.out, {"model": model, "model_config_sha256": digest(Path(model) / "config.json"),
                     "weight_stats": {p.name: [p.stat().st_size, p.stat().st_mtime_ns] for p in weights},
                     "case_manifest_sha256": digest(case_root / "cases.json"),
                     "spec_sha256": digest(SPEC_PATH), "versions": versions(), "records": rows})


def score(args):
    import functools
    from types import SimpleNamespace
    import torch
    import torch.distributed as dist
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
    from tensordict import NonTensorData, NonTensorStack, TensorDict
    from transformers import Qwen3VLForConditionalGeneration
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextDecoderLayer, Qwen3VLVisionBlock

    repo = setup(args.workspace)
    from verl.experimental.agent_loop.agent_loop import AgentLoopWorker
    from verl.models.transformers.monkey_patch import apply_monkey_patch
    from verl.utils import tensordict_utils as tu
    from verl.utils.dataset.dataset_utils import DatasetPadMode
    from verl.utils.tokenizer import hf_tokenizer
    from verl.workers.config import FSDPEngineConfig, HFModelConfig
    from verl.workers.engine.fsdp.transformer_impl import FSDPEngineWithLMHead

    spec = read(SPEC_PATH)
    recorded = read(args.capture)
    model = model_path(args)
    case_root = Path(args.assets or args.workspace) / "data/parity"
    if recorded["model"] != model or recorded["spec_sha256"] != digest(SPEC_PATH):
        raise RuntimeError("Model or verifier parameters changed")
    if recorded["model_config_sha256"] != digest(Path(model) / "config.json"):
        raise RuntimeError("Model config changed")
    stats = {p.name: [p.stat().st_size, p.stat().st_mtime_ns] for p in sorted(Path(model).glob("*.safetensors"))}
    if recorded["weight_stats"] != stats or recorded["case_manifest_sha256"] != digest(case_root / "cases.json"):
        raise RuntimeError("Weights or case manifest changed")
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    dist.init_process_group("nccl")
    device = torch.device("cuda", torch.cuda.current_device())
    # This is the submitted production factory, not a verifier-created substitute.
    config = HFModelConfig(path=model, use_remove_padding=True, use_fused_kernels=False,
                           override_config={"attn_implementation": "flash_attention_2"})
    processor = config.processor
    if processor is None or type(processor).__name__ != "Qwen3VLProcessor":
        raise RuntimeError("Expected a Qwen3VLProcessor")
    tokenizer = hf_tokenizer(model, local_files_only=True)
    worker = object.__new__(AgentLoopWorker)
    worker.processor, worker.tokenizer = processor, tokenizer
    worker.mm_processor_kwargs = spec["mm_processor_kwargs"]

    # Match the training engine's fp32 parameter storage / bf16 computation;
    # keep RoPE buffers in fp32 to avoid a separate BF16-buffer precision defect.
    raw_model = Qwen3VLForConditionalGeneration.from_pretrained(
        model, dtype=torch.float32, attn_implementation="flash_attention_2", local_files_only=True)
    rope_model = raw_model.model
    apply_monkey_patch(raw_model, use_remove_padding=True, ulysses_sp_size=1, use_fused_kernels=False)
    module = FSDP(raw_model, device_id=device, use_orig_params=True,
                  mixed_precision=MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.float32,
                                                 buffer_dtype=torch.float32),
                  auto_wrap_policy=functools.partial(transformer_auto_wrap_policy,
                      transformer_layer_cls={Qwen3VLTextDecoderLayer, Qwen3VLVisionBlock}))
    module.eval()
    # Use production input preparation, causal labels, model forward and output
    # extraction. Only heavyweight optimizer/checkpoint initialization is omitted.
    engine = object.__new__(FSDPEngineWithLMHead)
    engine.module = module
    engine.model_config = config
    engine.engine_config = FSDPEngineConfig(use_torch_compile=False)
    engine.use_ulysses_sp = False
    engine.ulysses_sequence_parallel_size = 1
    engine.pad_to_length = False
    engine._autocast_dtype = torch.bfloat16
    results = []
    for row in recorded["records"]:
        if [digest(case_root / p) for p in row["images"]] != row["image_sha256"]:
            raise RuntimeError("Image bytes changed")
        inputs, images = encode(processor, row, case_root, spec)
        if inputs["input_ids"][0].tolist() != row["prompt_ids"]:
            raise RuntimeError("Trainer and rollout prompt tokenization differ")
        ids = torch.tensor([row["prompt_ids"] + row["response_ids"]])
        mask = torch.ones_like(ids)
        output = SimpleNamespace(multi_modal_data={"images": images} if images else {},
                                 mm_processor_kwargs=spec["mm_processor_kwargs"])
        # Both calls are the real agent-loop methods used in the RL run.
        mm = worker._compute_multi_modal_inputs(output, ids)
        actual_grid = mm["image_grid_thw"].tolist() if images else None
        if actual_grid != row["image_grid_thw"]:
            raise RuntimeError("Rollout/trainer image grid mismatch; cannot attribute to positions")
        positions = worker._compute_position_ids(ids, mask, mm, spec["mm_processor_kwargs"])
        # Canonical image coordinates from the model's own HF implementation.
        rope_kwargs = {"input_ids": ids, "attention_mask": mask, "image_grid_thw": mm.get("image_grid_thw")}
        if "mm_token_type_ids" in inspect.signature(rope_model.get_rope_index).parameters:
            types = torch.zeros_like(ids)
            types[ids == config.hf_config.image_token_id] = 1
            types[ids == config.hf_config.video_token_id] = 2
            rope_kwargs["mm_token_type_ids"] = types
        expected, _ = rope_model.get_rope_index(**rope_kwargs)
        nested_ids = torch.nested.as_nested_tensor([ids[0].to(device)], layout=torch.jagged)
        nested_positions = torch.nested.as_nested_tensor([positions[0].to(device)], layout=torch.jagged)
        mm_gpu = {k: v.to(device) if torch.is_tensor(v) else v for k, v in mm.items()}
        batch = TensorDict({"input_ids": nested_ids, "position_ids": nested_positions,
                            "temperature": torch.ones(1, device=device)}, batch_size=[1])
        batch["multi_modal_inputs"] = NonTensorStack(NonTensorData(mm_gpu))
        tu.assign_non_tensor(batch, use_remove_padding=True, use_fused_kernels=False,
                             pad_mode=DatasetPadMode.NO_PADDING, calculate_entropy=False)
        actual_inputs, _ = engine.prepare_model_inputs(batch)
        model_positions = actual_inputs["position_ids"]
        if model_positions.ndim == 2:
            comparable = model_positions.unsqueeze(0).expand(3, -1, -1)
        elif model_positions.ndim == 3 and model_positions.shape[0] in (3, 4):
            comparable = model_positions[-3:]
        else:
            raise RuntimeError(f"Unexpected model position format {model_positions.shape}")
        position_diff = int((comparable.cpu() - expected.cpu()).abs().max())
        with torch.no_grad():
            _, output = engine.forward_step(batch, loss_function=None, forward_only=True)
        # Engine logits[i] predicts input_ids[i+1]. Include first response token.
        full = output["model_output"]["log_probs"].values()
        start, count = len(row["prompt_ids"]) - 1, len(row["response_ids"])
        logps = full[start:start + count].float().cpu().tolist()
        results.append({"name": row["name"], "response_ids": row["response_ids"],
                        "vllm_logps": row["vllm_logps"], "trainer_logps": logps,
                        "position_shape": list(positions.shape), "image_grid_thw": actual_grid,
                        "engine_position_shape": list(model_positions.shape),
                        "engine_vs_reference_positions_max_diff": position_diff,
                        "processor_type": type(processor).__name__,
                        "processor_has_get_rope_index": hasattr(processor, "get_rope_index")})
    if dist.get_rank() == 0:
        write(args.out, {"capture_sha256": digest(args.capture), "workspace": str(Path(args.workspace).resolve()),
                         "verl_import_root": str(repo), "world_size": dist.get_world_size(),
                         "versions": versions(), "records": results})
    dist.barrier()
    dist.destroy_process_group()


def verify(args):
    result = repair_verdict(read(args.scores), read(SPEC_PATH))
    write(args.out, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name, func in (("capture", capture), ("score", score)):
        p = sub.add_parser(name)
        p.add_argument("--workspace", required=True)
        p.add_argument("--assets", help="Asset workspace; defaults to --workspace")
        p.add_argument("--model")
        p.add_argument("--out", required=True)
        if name == "capture":
            p.add_argument("--tp", type=int, default=1)
        else:
            p.add_argument("--capture", required=True)
        p.set_defaults(func=func)
    p = sub.add_parser("verify")
    p.add_argument("--scores", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(func=verify)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
