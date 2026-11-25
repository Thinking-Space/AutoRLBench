"""Read unchanged KIVI softmax binaries through a private Triton cache."""
from __future__ import annotations

import base64
import json
from pathlib import Path

import triton
from triton.runtime.cache import FileCacheManager

METADATA = "_kivi_packed_softmax_kernel.json"
STATS = {"groups_reused": 0, "bytes_copied": 0, "shared_misses": 0}


class KiviReadThroughCacheManager(FileCacheManager):
    """Preserve Triton's key and binary; materialize matching groups locally."""

    donor_roots = ()
    expected_target = None

    def get_group(self, filename):
        local = super().get_group(filename)
        if local is not None or filename != METADATA:
            return local
        for root in self.donor_roots:
            donor = root / self.key
            group_file = donor / f"__grp__{filename}"
            if not group_file.is_file():
                continue
            group = json.loads(group_file.read_text())["child_paths"]
            if filename not in group or not all(Path(p).is_file() for p in group.values()):
                continue
            metadata = json.loads(Path(group[filename]).read_text())
            encoded_hash = base64.b32encode(bytes.fromhex(metadata["hash"])).decode().rstrip("=")
            if encoded_hash != self.key:
                raise ValueError(f"Triton donor key mismatch: {donor}")
            if metadata["target"] != self.expected_target or metadata["triton_version"] != triton.__version__:
                raise ValueError(f"Triton donor target/version mismatch: {donor}")
            if metadata["name"] != "_kivi_packed_softmax_kernel":
                raise ValueError(f"Unexpected donor kernel: {donor}")
            local_group = {}
            copied = 0
            for name, source in group.items():
                if Path(name).name != name or Path(source).parent != donor:
                    raise ValueError(f"Unexpected donor child path: {source}")
                data = Path(source).read_bytes()
                local_group[name] = self.put(data, name)
                copied += len(data)
            self.put_group(filename, local_group)
            STATS["groups_reused"] += 1
            STATS["bytes_copied"] += copied
            if STATS["groups_reused"] == 1:
                print("KIVI_CACHE_FIRST_HIT " + json.dumps({"donor": str(donor), "files": len(local_group), "bytes": copied}), flush=True)
            return super().get_group(filename)
        STATS["shared_misses"] += 1
        return None


def install_kivi_readthrough_cache(config):
    """Install after the evaluator selects the configured resumed array task."""
    import atexit
    import torch
    from triton import knobs

    if torch.__version__ != config["torch"] or triton.__version__ != config["triton"]:
        raise RuntimeError("KIVI cache reuse requires the validated torch/Triton environment")
    if list(torch.cuda.get_device_capability()) != config["device_capability"]:
        raise RuntimeError("KIVI cache reuse requires the validated device")
    KiviReadThroughCacheManager.donor_roots = tuple(Path(p) for p in config["donor_roots"])
    KiviReadThroughCacheManager.expected_target = config["target"]
    knobs.cache.manager_class = KiviReadThroughCacheManager
    atexit.register(lambda: print("KIVI_CACHE_STATS " + json.dumps(STATS), flush=True))
    print("KIVI_CACHE_REUSE " + json.dumps({"local": knobs.cache.dir, "donors": config["donor_roots"], "target": config["target"], "triton": triton.__version__}), flush=True)
