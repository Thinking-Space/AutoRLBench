# SPDX-License-Identifier: Apache-2.0
"""Check tracker-selected torch_dist payloads and bind conversion to public HF files."""
import hashlib
import json
from pathlib import Path


def file_sha(path):
    sha = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            sha.update(block)
    return sha.hexdigest()


def convert_bin_to_safetensors(source, target):
    """Re-encode trusted pinned HF tensors, retaining every key and exact dtype/value."""
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    source, target = Path(source), Path(target)
    if target.exists():
        raise ValueError("Conversion target already exists")
    state = torch.load(source, map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(state, dict) or not state or not all(
        isinstance(key, str) and isinstance(value, torch.Tensor) for key, value in state.items()
    ):
        raise ValueError("Expected a flat tensor state dict")
    # Clone so separately named tensors sharing storage retain all keys. No
    # quantization, dtype conversion, pruning or model configuration edit.
    tensors = {key: value.detach().contiguous().clone() for key, value in state.items()}
    temp = target.with_suffix(target.suffix + ".partial")
    save_file(tensors, str(temp), metadata={"format": "pt"})
    del tensors
    with safe_open(str(temp), framework="pt", device="cpu") as saved:
        if set(saved.keys()) != set(state):
            raise ValueError("Converted tensor keys differ")
        for key, original in state.items():
            actual = saved.get_tensor(key)
            if actual.dtype != original.dtype or actual.shape != original.shape or not torch.equal(actual, original):
                raise ValueError("Converted tensor differs: " + key)
    temp.replace(target)
    return {"source_sha256": file_sha(source), "target_sha256": file_sha(target),
            "target_size": target.stat().st_size, "tensor_count": len(state),
            "all_tensors_exactly_equal": True}


def inspect_checkpoint(root, *, expected_iteration=None, require_optimizer=False):
    from torch.distributed.checkpoint import FileSystemReader

    root = Path(root).resolve()
    tracker = root / "latest_checkpointed_iteration.txt"
    if not tracker.is_file():
        raise ValueError("Missing checkpoint root tracker")
    iteration = tracker.read_text().strip()
    if expected_iteration is not None and iteration != str(expected_iteration):
        raise ValueError("Unexpected checkpoint iteration")
    if iteration == "release":
        selected = root / "release"
    elif iteration.isdigit():
        selected = root / f"iter_{int(iteration):07d}"
    else:
        raise ValueError("Invalid checkpoint tracker")
    metadata_path = selected / ".metadata"
    if not metadata_path.is_file():
        raise ValueError("Missing selected torch_dist metadata")
    # Trusted locally produced model metadata only. This is the standard PyTorch
    # DCP reader, not an ad hoc pickle load from an unknown external artifact.
    metadata = FileSystemReader(selected).read_metadata()
    keys = list(metadata.state_dict_metadata)
    # Megatron ShardedTensor.key is often flat rather than under "model.".
    model_prefixes = ("model.", "model0.", "embedding.", "decoder.", "output_layer.")
    if not any(key.startswith(model_prefixes) for key in keys):
        raise ValueError("Checkpoint metadata has no model state")
    if require_optimizer and not any("optim" in key.lower() for key in keys):
        raise ValueError("Checkpoint metadata has no optimizer state")
    referenced = {}
    for info in (metadata.storage_data or {}).values():
        relative = Path(info.relative_path)
        path = (selected / relative).resolve()
        if relative.is_absolute() or not path.is_relative_to(selected.resolve()):
            raise ValueError("Checkpoint storage reference escapes selected directory")
        if not path.is_file() or path.stat().st_size < info.offset + info.length:
            raise ValueError("Missing or truncated referenced checkpoint shard")
        referenced[str(relative)] = path
    if not referenced:
        raise ValueError("Checkpoint metadata references no payload")
    files = {".metadata": {"size": metadata_path.stat().st_size, "sha256": file_sha(metadata_path)}}
    for name, path in sorted(referenced.items()):
        files[name] = {"size": path.stat().st_size, "sha256": file_sha(path)}
    for name in ["common.pt", "metadata.json"]:
        path = selected / name
        if path.is_file():
            files[name] = {"size": path.stat().st_size, "sha256": file_sha(path)}
    return {"iteration": iteration, "selected": selected.name, "state_keys": keys,
            "referenced_files": files, "restore_tested": False}


