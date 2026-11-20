"""Build saver and loader bookkeeping for a native Actor checkpoint."""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path


VERSION = 1


def canonical_name(name):
    # FSDP/checkpoint wrappers do not change a parameter's semantic identity.
    wrappers = {"_fsdp_wrapped_module", "_checkpoint_wrapped_module", "_orig_mod"}
    return ".".join(part for part in str(name).split(".") if part not in wrappers)


def _name_map(model):
    result, names = {}, set()
    for name, parameter in model.named_parameters():
        name = canonical_name(name)
        if name in names:
            raise ValueError(f"Non-unique canonical parameter identity: {name}")
        names.add(name)
        result[id(parameter)] = name
    return result


def _signature(parameter):
    local = parameter.to_local() if hasattr(parameter, "to_local") else parameter
    return tuple(parameter.shape), tuple(local.shape), str(parameter.dtype), str(parameter.layout)


def _json_signature(parameter):
    shape, local_shape, dtype, layout = _signature(parameter)
    return {"shape": list(shape), "local_shape": list(local_shape), "dtype": dtype, "layout": layout}


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")
    temporary.replace(path)


def identity_path(checkpoint_dir, rank=0, world_size=1):
    return Path(checkpoint_dir) / f"optimizer_identity_world_size_{world_size}_rank_{rank}.json"


def optimizer_identity(model, optimizer, state_dict):
    """Describe the actual serialized positional IDs using stable model names."""
    names = _name_map(model)
    if len(state_dict["param_groups"]) != len(optimizer.param_groups):
        raise ValueError("Checkpoint/live optimizer group count differs")
    groups, seen = [], set()
    for saved, live in zip(state_dict["param_groups"], optimizer.param_groups, strict=True):
        if len(saved["params"]) != len(live["params"]):
            raise ValueError("Checkpoint/live optimizer group size differs")
        entries = []
        for saved_id, parameter in zip(saved["params"], live["params"], strict=True):
            if id(parameter) not in names:
                raise ValueError("Optimizer parameter has no stable model identity")
            name = names[id(parameter)]
            if name in seen:
                raise ValueError(f"Parameter appears in multiple optimizer positions: {name}")
            seen.add(name)
            entries.append({"state_id": int(saved_id), "name": name, **_json_signature(parameter)})
        groups.append(entries)
    payload = {"version": VERSION, "groups": groups,
               "state_entry_count": len(state_dict["state"]),
               "parameter_count": len(seen)}
    payload["identity_sha256"] = hashlib.sha256(
        json.dumps(groups, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return payload


def save_optimizer_identity(model, optimizer, state_dict, checkpoint_dir, rank=0, world_size=1):
    value = optimizer_identity(model, optimizer, state_dict)
    value.update(rank=int(rank), world_size=int(world_size))
    _write(identity_path(checkpoint_dir, rank, world_size), value)
    return value


def rebuild_parameter_groups(model, optimizer):
    """Build parameter-group enumeration for the current runtime."""
    names = _name_map(model)
    for group in optimizer.param_groups:
        original = list(group["params"])
        buckets = defaultdict(list)
        for index, parameter in enumerate(original):
            if id(parameter) not in names:
                raise ValueError("Optimizer parameter has no stable model identity")
            buckets[_signature(parameter)].append(index)
        rebuilt = list(original)
        for positions in buckets.values():
            ordered = sorted((original[index] for index in positions), key=lambda p: names[id(p)])
            for index, parameter in zip(positions, ordered, strict=True):
                rebuilt[index] = parameter
        group["params"] = rebuilt
        if "param_names" in group:
            group["param_names"] = [names[id(parameter)] for parameter in rebuilt]


def _validated_manifest(value, state_dict):
    if value.get("version") != VERSION or len(value["groups"]) != len(state_dict["param_groups"]):
        raise ValueError("Unsupported/incompatible optimizer identity sidecar")
    digest = hashlib.sha256(json.dumps(value["groups"], sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if digest != value.get("identity_sha256"):
        raise ValueError("Optimizer identity sidecar digest mismatch")
    all_names, all_ids = set(), set()
    for identity, group in zip(value["groups"], state_dict["param_groups"], strict=True):
        if [item["state_id"] for item in identity] != group["params"]:
            raise ValueError("Optimizer identity sidecar does not match serialized IDs")
        for item in identity:
            if item["name"] in all_names or item["state_id"] in all_ids:
                raise ValueError("Duplicate optimizer identity")
            all_names.add(item["name"])
            all_ids.add(item["state_id"])
    if not state_dict["state"]:
        raise ValueError("Cannot restore an empty optimizer state")






def pause_after_checkpoint(global_step, checkpoint_root):
    """Stop phase one after its real native checkpoint and step log exist."""
    target = os.environ.get("MECHANISM_STOP_AFTER_CHECKPOINT")
    if not target:
        return False
    target = int(target)
    if target < 1:
        raise ValueError("Checkpoint pause step must be positive")
    if global_step < target:
        return False
    if global_step != target:
        raise ValueError("Requested optimizer-resume checkpoint pause was missed")
    root = Path(checkpoint_root)
    step_dir = root / f"global_step_{target}"
    marker = root / "latest_checkpointed_iteration.txt"
    required = [step_dir / "data.pt", step_dir / "actor/model_world_size_1_rank_0.pt",
                step_dir / "actor/optim_world_size_1_rank_0.pt",
                step_dir / "actor/extra_state_world_size_1_rank_0.pt",
                identity_path(step_dir / "actor")]
    if (not marker.is_file() or marker.read_text().strip() != str(target)
            or any(not path.is_file() for path in required)):
        raise ValueError("Native model/optimizer/RNG/dataloader checkpoint is incomplete")
    return True

def restore_optimizer(model, optimizer, state_dict, identity):
    _validated_manifest(identity, state_dict)
    names = _name_map(model)
    rebuild_parameter_groups(model, optimizer)
    incoming = {**state_dict, "state": dict(state_dict["state"]),
                "param_groups": [{**group, "params": list(group["params"])}
                                 for group in state_dict["param_groups"]]}
    for saved_entries, live_group in zip(identity["groups"], optimizer.param_groups, strict=True):
        by_name = {entry["name"]: entry for entry in saved_entries}
        live_names = [names[id(parameter)] for parameter in live_group["params"]]
        if set(live_names) != set(by_name) or len(live_names) != len(saved_entries):
            raise ValueError("Optimizer group membership changed; only enumeration may differ")
        for parameter, name in zip(live_group["params"], live_names, strict=True):
            expected = by_name[name]
            if _json_signature(parameter) != {key: expected[key] for key in
                    ("shape", "local_shape", "dtype", "layout")}:
                raise ValueError(f"Optimizer parameter signature changed: {name}")
    optimizer.load_state_dict(incoming)


def load_optimizer_with_identity(model, optimizer, state_dict, checkpoint_dir, rank=0, world_size=1):
    identity = json.loads(identity_path(checkpoint_dir, rank, world_size).read_text())
    if identity.get("rank") != int(rank) or identity.get("world_size") != int(world_size):
        raise ValueError("Optimizer resume topology differs from the recorded checkpoint")
    restore_optimizer(model, optimizer, state_dict, identity)
