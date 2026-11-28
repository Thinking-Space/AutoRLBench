"""Measure storage retained by the pinned SGLang embedding cache on CPU."""

import argparse
import gc
import hashlib
import json
import sys
import weakref
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

import torch


SOURCE_SHA256 = "f1415edeb4554534de2648e2ee1ffacde06236a7b0028f29f94b9b94a954b739"


def load_cache(path):
    source = path.read_bytes()
    digest = hashlib.sha256(source).hexdigest()
    if digest != SOURCE_SHA256:
        raise ValueError(f"Expected upstream SHA-256 {SOURCE_SHA256}, got {digest}")

    # The only external SGLang import in the file is used as a type annotation
    # in the abstract base class. The concrete cache does not call this allocator.
    allocator = ModuleType("sglang.srt.mem_cache.allocator")
    allocator.BaseTokenToKVPoolAllocator = object
    upstream = ModuleType("pinned_sglang_multimodal_cache")
    with patch.dict(sys.modules, {allocator.__name__: allocator, upstream.__name__: upstream}):
        exec(compile(source, str(path), "exec"), upstream.__dict__)
    return upstream


def retained_bytes(cache):
    sizes = {}
    for entry in cache.mm_cache.values():
        storage = entry.embedding.untyped_storage()
        sizes[(storage.device, storage.data_ptr())] = storage.nbytes()
    return sum(sizes.values())


def measure(upstream):
    entries, rows, features, keep = 16, 512, 512, 8
    budget = entries * keep * features * torch.float32.itemsize
    original = upstream.MultiModalStaticCache(budget)
    copy_control = upstream.MultiModalStaticCache(budget)
    batch_refs = []
    for index in range(entries):
        encoded_batch = torch.full((rows, features), float(index), dtype=torch.float32, device="cpu")
        batch_refs.append(weakref.ref(encoded_batch))
        view = encoded_batch[1:1 + keep]
        assert view.is_contiguous()
        assert original.set(index, upstream.EmbeddingResult(embedding=view))
        compact = view
        if view.untyped_storage().nbytes() > view.numel() * view.element_size():
            compact = view.clone(memory_format=torch.contiguous_format)
        assert copy_control.set(index, upstream.EmbeddingResult(embedding=compact))
    del encoded_batch, view, compact

    for index in range(entries):
        expected = original.get_single(index).embedding
        actual = copy_control.get_single(index).embedding
        assert torch.equal(expected, actual)
        assert (expected.shape, expected.dtype, expected.device) == (actual.shape, actual.dtype, actual.device)

    del expected, actual
    assert len(original) == len(copy_control) == entries
    assert original.current_size == copy_control.current_size == budget
    original_bytes = retained_bytes(original)
    control_bytes = retained_bytes(copy_control)
    assert original_bytes == entries * rows * features * torch.float32.itemsize
    assert control_bytes == budget
    assert all(ref() is not None for ref in batch_refs)
    original.clear()
    gc.collect()
    assert all(ref() is None for ref in batch_refs)
    assert not torch.cuda.is_initialized()
    return {
        "device": "cpu",
        "torch_version": str(torch.__version__),
        "upstream_source_sha256": SOURCE_SHA256,
        "entries": entries,
        "logical_capacity_bytes": budget,
        "original_cache_storage_bytes": original_bytes,
        "compact_copy_control_storage_bytes": control_bytes,
        "same_values_shape_dtype_device": True,
        "original_batches_reclaimed_after_original_cache_cleared": True,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, default=Path("multimodal_cache.upstream.py"))
    args = parser.parse_args()
    print(json.dumps(measure(load_cache(args.upstream)), indent=2))
