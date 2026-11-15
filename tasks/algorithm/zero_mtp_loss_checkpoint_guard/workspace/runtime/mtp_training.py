"""Small real Megatron GPTModel probe. No stubs of framework execution."""
from __future__ import annotations
import hashlib
import importlib.util
from pathlib import Path
import socket
import sys


def file_identity(path):
    path = Path(path).resolve()
    return {"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def select_framework(megatron_root, miles_root):
    roots = [(Path(megatron_root).resolve(), "megatron/core/models/gpt/gpt_model.py"),
             (Path(miles_root).resolve(), "miles/backends/megatron_utils/model.py")]
    for root, sentinel in roots:
        if not (root / sentinel).is_file():
            raise RuntimeError(f"Missing staged complete framework: {root / sentinel}")
    for root, _ in reversed(roots):
        sys.path.insert(0, str(root))
    import megatron.core.models.gpt.gpt_model as gpt
    import megatron.core.transformer.multi_token_prediction as mtp
    import miles
    for module, root in [(gpt, roots[0][0]), (mtp, roots[0][0]), (miles, roots[1][0])]:
        if not Path(module.__file__).resolve().is_relative_to(root):
            raise RuntimeError(f"Unexpected framework origin: {module.__name__}")
    return {"gpt_model": file_identity(gpt.__file__),
            "multi_token_prediction": file_identity(mtp.__file__),
            "miles_model": file_identity(roots[1][0] / roots[1][1])}


def load_hook(path):
    spec = importlib.util.spec_from_file_location("historical_mtp_hook", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.megatron_init(None)
    return file_identity(path)


def train_probe(args, report):
    import torch
    import torch.distributed as dist
    report["framework_sources"] = select_framework(args.megatron_root, args.miles_root)
    if not torch.cuda.is_available():
        raise RuntimeError("Packed TransformerEngine probe requires one CUDA GPU; no CPU substitute is used")
    torch.cuda.set_device(0)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    from megatron.core import parallel_state
    from megatron.core.models.gpt.gpt_model import GPTModel
    from megatron.core.models.gpt.gpt_layer_specs import (
        get_gpt_layer_with_transformer_engine_spec, get_gpt_mtp_block_spec)
    from megatron.core.packed_seq_params import PackedSeqParams
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from megatron.core.transformer.transformer_config import TransformerConfig
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    dist.init_process_group("nccl", init_method=f"tcp://127.0.0.1:{port}", rank=0, world_size=1)
    try:
        parallel_state.initialize_model_parallel(tensor_model_parallel_size=1,
                                                pipeline_model_parallel_size=1,
                                                context_parallel_size=1)
        model_parallel_cuda_manual_seed(args.seed)
        report["hook"] = load_hook(args.hook_file)
        config = TransformerConfig(
            num_layers=1, hidden_size=128, num_attention_heads=4, ffn_hidden_size=256,
            mtp_num_layers=1, mtp_loss_scaling_factor=0.0, normalization="RMSNorm",
            bf16=True, params_dtype=torch.bfloat16, use_cpu_initialization=False,
            hidden_dropout=0.0, attention_dropout=0.0, sequence_parallel=False,
            pipeline_dtype=torch.bfloat16, gradient_accumulation_fusion=False,
            bias_activation_fusion=False, bias_dropout_fusion=False,
            masked_softmax_fusion=False, persist_layer_norm=False,
            apply_rope_fusion=False, cross_entropy_loss_fusion=False)
        spec = get_gpt_layer_with_transformer_engine_spec()
        mtp_spec = get_gpt_mtp_block_spec(config=config, spec=spec, use_transformer_engine=True)
        def build():
            model = GPTModel(config=config, transformer_layer_spec=spec, vocab_size=128,
                             max_sequence_length=32, position_embedding_type="rope",
                             parallel_output=False, share_embeddings_and_output_weights=False,
                             mtp_block_spec=mtp_spec).cuda()
            model.train()
            return model
        model = build()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
        original_config = model.config
        original_keys = tuple(model.state_dict())
        mtp_before = {k: v.detach().clone() for k, v in model.named_parameters() if k.startswith("mtp.")}
        if not mtp_before:
            raise AssertionError("The real model must retain MTP parameters")
        base_before = model.output_layer.weight.detach().clone()
        tokens = torch.randint(0, 128, (1, 32), device="cuda")
        cu = torch.tensor([0, 16, 32], device="cuda", dtype=torch.int32)
        packed = PackedSeqParams(qkv_format="thd", cu_seqlens_q=cu, cu_seqlens_kv=cu,
                                 max_seqlen_q=16, max_seqlen_kv=16)
        mask = torch.ones((1, 32), device="cuda", dtype=torch.float32)
        targets = torch.roll(tokens, shifts=-1, dims=-1)
        advantages = torch.linspace(-1, 1, 32, device="cuda").reshape(1, -1)
        report["mtp_parameter_tensors"] = len(mtp_before)
        for step in range(args.steps):
            optimizer.zero_grad(set_to_none=True)
            report["phase"] = "real_model_forward"
            # Same nullable interface as Miles, with a complete real model.
            logits = model(input_ids=tokens, position_ids=None, attention_mask=None,
                           labels=None, packed_seq_params=packed, loss_mask=mask)
            report["phase"] = "real_backward_optimizer"
            log_probs = logits.float().log_softmax(-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
            loss = -(log_probs * advantages * mask).sum() / mask.sum()
            if not torch.isfinite(loss):
                raise AssertionError("Nonfinite surrogate loss")
            loss.backward()
            grads = [p.grad for n, p in model.named_parameters() if not n.startswith("mtp.") and p.grad is not None]
            if not grads or not all(torch.isfinite(g).all() for g in grads):
                raise AssertionError("Missing/nonfinite base gradients")
            if any(p.grad is not None for n, p in model.named_parameters() if n.startswith("mtp.")):
                raise AssertionError("Disabled MTP auxiliary path accumulated gradients")
            optimizer.step()
            report["optimizer_steps"] = step + 1
            report.setdefault("losses", []).append(float(loss.detach()))
        if model.config is not original_config or model.config.mtp_num_layers != 1:
            raise AssertionError("MTP configuration was not restored")
        if tuple(model.state_dict()) != original_keys:
            raise AssertionError("Model state-dict keys changed")
        if torch.equal(base_before, model.output_layer.weight):
            raise AssertionError("Base output parameters did not update")
        current_parameters = dict(model.named_parameters())
        if any(not torch.equal(v, current_parameters[k]) for k, v in mtp_before.items()):
            raise AssertionError("Disabled MTP parameters changed")
        report["phase"] = "small_checkpoint_roundtrip"
        # Real small torch checkpoint, not the historical distributed format.
        checkpoint = Path(args.output_dir) / "tiny_model_optimizer.pt"
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "step": args.steps, "mtp_num_layers": config.mtp_num_layers}, checkpoint)
        restored = torch.load(checkpoint, map_location="cuda", weights_only=False)
        clone = build()
        clone.load_state_dict(restored["model"], strict=True)
        clone_optimizer = torch.optim.AdamW(clone.parameters(), lr=1e-3)
        clone_optimizer.load_state_dict(restored["optimizer"])
        if restored["step"] != args.steps or restored["mtp_num_layers"] != 1:
            raise AssertionError("Checkpoint metadata changed")
        if any(not torch.equal(v, dict(model.named_parameters())[k]) for k, v in clone.named_parameters()):
            raise AssertionError("Checkpoint parameters did not roundtrip")
        if not clone_optimizer.state:
            raise AssertionError("Optimizer state did not roundtrip")
        report.update(checkpoint=file_identity(checkpoint), config_restored=True,
                      mtp_weights_preserved=True, base_parameters_updated=True, phase="completed")
    finally:
        parallel_state.destroy_model_parallel()
        dist.destroy_process_group()
