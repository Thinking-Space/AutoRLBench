"""QAT integration for the pinned, packed-cache KVarN vLLM fork.

The upstream fork owns the quantizer and attention kernels. This module only
builds the engine and audits the live workers; it never fake-quantizes KV.
"""
from dataclasses import asdict

KVARN_REVISION = "7586257f1c632e63187bfacbbe21ccb51540f7b3"


class KVarNAuditExtension:
    def kvarn_audit(self):
        import torch

        layers = []
        for name, module in self.model_runner.model.named_modules():
            impl = getattr(module, "impl", None)
            cfg = getattr(impl, "kvarn_config", None)
            if cfg is not None:
                cache = module.kv_cache
                layers.append({"name": name, "impl": type(impl).__name__,
                               "config": asdict(cfg),
                               "tile_bytes": cfg.tile_bytes_aligned,
                               "flash_attn_version": impl.fa_version,
                               "cache_dtype": str(cache.dtype),
                               "cache_shape": list(cache.shape)})
        config = self.vllm_config
        return {
            "gpu": torch.cuda.get_device_name(),
            "compute_capability": list(torch.cuda.get_device_capability()),
            "gpu_total_bytes": torch.cuda.get_device_properties(0).total_memory,
            "model_dtype": str(config.model_config.dtype),
            "cache_dtype": config.cache_config.cache_dtype,
            "block_size": config.cache_config.block_size,
            "num_gpu_blocks": config.cache_config.num_gpu_blocks,
            "max_num_seqs": config.scheduler_config.max_num_seqs,
            "torch_allocated_bytes": torch.cuda.memory_allocated(),
            "torch_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "layers": layers,
        }


def make_engine(args):
    from vllm import LLM

    llm = LLM(
        model=args.model, dtype="bfloat16", kv_cache_dtype=args.kv_cache_dtype,
        tensor_parallel_size=1, max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs, max_num_batched_tokens=32768,
        gpu_memory_utilization=0.9, seed=42, enable_prefix_caching=False,
        enforce_eager=args.eager, block_size=128,
        worker_extension_cls="opqad.integrations.kvarn_vllm.KVarNAuditExtension",
    )
    audit = llm.collective_rpc("kvarn_audit")
    if args.kv_cache_dtype.startswith("kvarn"):
        expected = llm.llm_engine.vllm_config.model_config.hf_config.num_hidden_layers
        for worker in audit:
            assert worker["cache_dtype"] == args.kv_cache_dtype, worker
            assert len(worker["layers"]) == expected, worker
            for layer in worker["layers"]:
                cfg = layer["config"]
                assert (cfg["key_bits"], cfg["value_bits"], cfg["group"]) == (4, 2, 128)
                assert cfg["boundary_skip_layers"] == 0
                assert cfg["sink_tokens"] == 128 and cfg["sinkhorn_iters"] == 8
                assert layer["cache_dtype"] == "torch.uint8", layer
    return llm, audit
