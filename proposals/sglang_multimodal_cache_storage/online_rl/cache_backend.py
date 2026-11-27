"""Run the public SGLang per-image prefill path inside an online rollout.

The policy provides the actual vision forward pass. A prefill batch can hold a
multi-image request or a separate request for each policy observation. The returned embedding is consumed by the action decoder.
The upstream cache and scheduler source are pinned by SHA-256 on first use.
"""

from __future__ import annotations

import abc
import argparse
import ast
import hashlib
import importlib
import importlib.util
import logging
import os
import tempfile
import urllib.request
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from types import FunctionType, SimpleNamespace
from typing import List, Optional

import torch


REVISION = "02b5e123c490c988d21526d0dd7837fefdd0e1da"
RAW_ROOT = f"https://raw.githubusercontent.com/sgl-project/sglang/{REVISION}/python/"
SOURCES = {
    "multimodal_cache.py": ("sglang/srt/mem_cache/multimodal_cache.py",
                            "f1415edeb4554534de2648e2ee1ffacde06236a7b0028f29f94b9b94a954b739"),
    "mm_schedule.py": ("sglang/srt/managers/mm_schedule.py",
                       "8cee0ab89b8942125473e55e8d1ed845736ab4925e668d33231fae9d68c0605d"),
}
CACHE_DEFS = ["MultimodalCache", "_get_tensor_size", "EmbeddingResult", "MultiModalStaticCache"]
SCHEDULE_DEFS = [
    "_embedding_token_count", "_discard_mismatched_cached_embedding",
    "_can_skip_pre_embed_feature_move", "_move_items_to_device",
    "PerImageRequestInfo", "_batch_encode_per_image_misses",
    "_assemble_per_image_chunk", "_get_chunked_prefill_embedding",
]


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def prepare_sources(source_dir=None):
    root = Path(source_dir or os.environ.get("PR87_SGLANG_SOURCE_DIR") or
                Path.home() / ".cache" / "autorlbench" / "sglang" / REVISION)
    root.mkdir(parents=True, exist_ok=True)
    files = {}
    for name, (upstream_path, digest) in SOURCES.items():
        path = root / name
        if not path.is_file():
            with urllib.request.urlopen(RAW_ROOT + upstream_path, timeout=75) as response:
                data = response.read()
            if _sha(data) != digest:
                raise ValueError(f"unexpected SHA-256 for upstream {name}")
            with tempfile.NamedTemporaryFile(dir=root, delete=False) as tmp:
                tmp.write(data)
            os.replace(tmp.name, path)
        if _sha(path.read_bytes()) != digest:
            raise ValueError(f"{path} does not match pinned source at {REVISION}")
        files[name] = path
    return files


def _runtime_modules():
    spec = importlib.util.find_spec("sglang")
    if spec is None or spec.origin is None:
        return None
    site = Path(spec.origin).parent.parent
    for rel_path, digest in SOURCES.values():
        candidate = site / rel_path
        if not candidate.is_file() or _sha(candidate.read_bytes()) != digest:
            return None
    try:
        cache = importlib.import_module("sglang.srt.mem_cache.multimodal_cache")
        schedule = importlib.import_module("sglang.srt.managers.mm_schedule")
    except (ImportError, RuntimeError, OSError):
        # The training image may contain SGLang source but lack server kernels.
        return None
    return cache, schedule


def _load_definitions(path, names, namespace):
    tree = ast.parse(path.read_bytes(), filename=str(path))
    definitions = {node.name: node for node in tree.body
                   if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
    subset = ast.Module(body=[definitions[name] for name in names], type_ignores=[])
    # Future annotations from this module preserve upstream type annotations
    # without importing the server's KV allocator or distributed process groups.
    exec(compile(subset, str(path), "exec"), namespace)


def _isolated_backend(source_dir):
    files = prepare_sources(source_dir)
    installed = _runtime_modules()
    if installed:
        cache, schedule = installed
        ns = vars(schedule).copy()
        if ns["_is_hip"] or ns["_is_npu"] or ns["_is_xpu"]:
            raise RuntimeError("this experiment exercises SGLang's CUDA per-image dispatch")
        for name in SCHEDULE_DEFS:
            fn = getattr(schedule, name)
            if isinstance(fn, FunctionType):
                ns[name] = FunctionType(fn.__code__, ns, fn.__name__, fn.__defaults__, fn.__closure__)
        return cache.MultiModalStaticCache, ns, "installed-sglang", files["multimodal_cache.py"]
    # Extract verbatim upstream definitions. When server extensions are absent,
    # only import-time dependencies and the CUDA backend flags are supplied.
    ns = dict(__name__=__name__, torch=torch, abc=abc, OrderedDict=OrderedDict,
              dataclass=dataclass, field=field, List=List, Optional=Optional,
              logger=logging.getLogger(__name__), _is_hip=False, _is_npu=False, _is_xpu=False)
    _load_definitions(files["multimodal_cache.py"], CACHE_DEFS, ns)
    _load_definitions(files["mm_schedule.py"], SCHEDULE_DEFS, ns)
    return ns["MultiModalStaticCache"], ns, "pinned-sglang-definitions", files["multimodal_cache.py"]


def image_key(image):
    """Stable content key for the public RGB observations, independent of episode."""
    digest = hashlib.sha256(f"{image.mode}:{image.width}:{image.height}:".encode("ascii"))
    digest.update(image.tobytes())
    return int.from_bytes(digest.digest()[:8], "big")


class RolloutEmbeddingCache:
    def __init__(self, capacity_bytes, mode="buggy", source_dir=None):
        if mode not in {"buggy", "fixed"} or capacity_bytes < 0:
            raise ValueError("mode must be buggy or fixed and capacity must be nonnegative")
        parent, namespace, self.backend, cache_source = _isolated_backend(source_dir)
        self.mode = mode
        self.requests = self.prefill_batches = 0
        self.assembled_sha256 = None
        self.assembled_shape = []
        self.assembled_dtype = None
        self.hits = self.misses = self.evictions = 0
        self.encoder_calls = self.encoded_images = self.multi_item_split_batches = 0
        self.lookup_hash = hashlib.sha256()
        if mode == "fixed":
            if __package__:
                from .reference_cache import reference_class
            else:
                from reference_cache import reference_class
            parent = reference_class(parent, cache_source)

        owner = self

        class ObservedCache(parent):
            def get_single(self, key):
                result = super().get_single(key)
                if result is None:
                    owner.misses += 1
                else:
                    owner.hits += 1
                owner.lookup_hash.update(f"{key}:{int(result is not None)}\n".encode())
                return result

            def set(self, key, result, loc=None):
                n_before = len(self.mm_cache)
                new = key not in self.mm_cache
                accepted = super().set(key, result, loc)
                owner.evictions += n_before + int(new and accepted) - len(self.mm_cache)
                return accepted

        self.cache = ObservedCache(int(capacity_bytes))
        namespace["embedding_cache"] = self.cache
        self.prefill = namespace["_get_chunked_prefill_embedding"]

    @torch.no_grad()
    def encode(self, images, keys, token_counts, encode_misses, device, *, requests_per_image=False):
        """Return [total_visual_tokens, hidden] in the original image order.

        encode_misses receives the scheduler's ordered list of unique missing PIL
        images. It returns the vision model's concatenated tensor on 'device'.
        With requests_per_image=True, one policy batch contains N single-image
        requests and upstream deduplicates and encodes misses across them.
        """
        if not images or len(images) != len(keys) or len(keys) != len(token_counts):
            raise ValueError("images, keys and token_counts must have equal, nonzero lengths")
        items = []
        offsets = []
        cursor = 0
        for image, key, count in zip(images, keys, token_counts):
            count = int(count)
            if count <= 0:
                raise ValueError("each image must contribute at least one visual token")
            start = 0 if requests_per_image else cursor
            span = (start, start + count - 1)
            offsets.append(span)
            items.append(SimpleNamespace(feature=image, hash=int(key), offsets=[span]))
            cursor += count

        def vision_forward(misses):
            self.encoder_calls += 1
            self.encoded_images += len(misses)
            self.multi_item_split_batches += int(len(misses) > 1)
            tensor = encode_misses([item.feature for item in misses])
            if not isinstance(tensor, torch.Tensor) or tensor.ndim != 2:
                raise TypeError("encode_misses must return a concatenated 2-D tensor")
            if tensor.device != input_ids.device:
                raise ValueError("vision output and rollout policy must use the same device")
            return tensor

        # The real scheduler uses offsets and the input device here. It does
        # not read placeholder values for per-image requests.
        if requests_per_image:
            boundaries = list(range(len(items) + 1))
            lengths = [end + 1 for _, end in offsets]
            request_offsets = [[span] for span in offsets]
        else:
            boundaries = [0, len(items)]
            lengths = [cursor]
            request_offsets = [offsets]
        input_ids = torch.zeros(cursor, dtype=torch.long, device=device)
        result, _ = self.prefill(vision_forward, items, boundaries, [0] * len(lengths),
                                 lengths, request_offsets, input_ids)
        # Hash the assembled bytes on CPU, then release the temporary copy.
        # Viewing as uint8 also supports types such as bfloat16 in NumPy.
        raw = result.detach().cpu().contiguous().view(torch.uint8).numpy()
        self.assembled_sha256 = hashlib.sha256(memoryview(raw)).hexdigest()
        self.assembled_shape = list(result.shape)
        self.assembled_dtype = str(result.dtype)
        self.requests += len(lengths)
        self.prefill_batches += 1
        return result

    def snapshot(self):
        """Deduplicate live allocations reachable from cache entries only."""
        storages = {}
        keys = list(self.cache.mm_cache)
        for key, entry in self.cache.mm_cache.items():
            tensor = entry.embedding
            storage = tensor.untyped_storage()
            identity = (str(tensor.device), storage.data_ptr())
            if identity not in storages:
                storages[identity] = {"device": str(tensor.device), "storage_bytes": storage.nbytes(),
                                      "logical_bytes": 0, "key_count": 0, "keys": []}
            group = storages[identity]
            group["keys"].append(key)
            group["logical_bytes"] += tensor.numel() * tensor.element_size()
            group["key_count"] += 1
        # Dict iteration follows the first live LRU entry for each storage; the
        # temporary identity pointer is never written to the trace.
        storage_groups = list(storages.values())
        devices = sorted({d for d, _ in storages})
        # Framework allocator counters also include model weights, activations
        # and tensors outside this cache. Keep these separate from live storage.
        allocator = {}
        for device in devices:
            if device.startswith("cuda:"):
                allocator[device] = {
                    "process_cuda_memory_allocated_bytes": torch.cuda.memory_allocated(device),
                    "process_cuda_memory_reserved_bytes": torch.cuda.memory_reserved(device),
                }
            elif device.startswith("mps:") and hasattr(torch.mps, "current_allocated_memory"):
                allocator[device] = {"process_mps_current_allocated_bytes": torch.mps.current_allocated_memory()}
        return {
            "revision": REVISION, "backend": self.backend, "mode": self.mode,
            "capacity_bytes": self.cache.max_size, "logical_bytes": self.cache.current_size,
            "storage_bytes": sum(group["storage_bytes"] for group in storage_groups),
            "storage_count": len(storage_groups), "storage_groups": storage_groups,
            "assembled_embedding_sha256": self.assembled_sha256,
            "assembled_embedding_shape": self.assembled_shape,
            "assembled_embedding_dtype": self.assembled_dtype,
            "process_allocator_by_device": allocator,
            "device": ",".join(devices), "devices": devices, "keys": keys,
            "requests": self.requests, "prefill_batches": self.prefill_batches,
            "hits": self.hits, "misses": self.misses,
            "evictions": self.evictions, "encoder_calls": self.encoder_calls,
            "scheduler_entry": "_get_chunked_prefill_embedding",
            "split_batches": self.encoder_calls, "multi_item_split_batches": self.multi_item_split_batches,
            "encoded_images": self.encoded_images,
            "lookup_sha256": self.lookup_hash.hexdigest(),
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch and verify the pinned upstream SGLang source")
    parser.add_argument("--prepare", required=True, metavar="DIRECTORY")
    for source in prepare_sources(parser.parse_args().prepare).values():
        print(source)
