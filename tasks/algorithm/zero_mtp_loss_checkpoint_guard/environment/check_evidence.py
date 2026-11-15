"""Check archived coverage, exact published metrics, and package identities."""
from __future__ import annotations

import ast
import gzip
import hashlib
import json
from pathlib import Path
import re


def check(task: Path) -> dict:
    source = json.loads((task / "environment/source_manifest.json").read_text())
    for relative, expected in source["published_file_hashes"].items():
        actual = hashlib.sha256((task / relative).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Published file identity changed: {relative}")
    inventory = task / "environment/runtime_sources.json"
    if inventory.exists():
        for entry in json.loads(inventory.read_text())["files"]:
            path = task / "workspace" / entry["path"]
            if hashlib.sha256(path.read_bytes()).hexdigest() != entry["shipped_sha256"]:
                raise ValueError(f"Framework inventory mismatch: {entry['path']}")
    exclusions = json.loads((task / "environment/privacy_exclusions.json").read_text())
    for entry in exclusions["files"]:
        path = task / entry["path"]
        if entry["action"] == "deleted":
            if path.exists():
                raise ValueError("Omitted implementation unexpectedly present")
        elif hashlib.sha256(path.read_bytes()).hexdigest() != entry["current_sha256"]:
            raise ValueError(f"Privacy inventory mismatch: {entry['path']}")
    for manifest_path in [task / "workspace/logs/validation/manifest.json",
                          task / "evaluations/issue/reference/validation/manifest.json"]:
        if not manifest_path.exists():
            continue
        for entry in json.loads(manifest_path.read_text())["artifacts"]:
            path = manifest_path.parent / entry["artifact"]
            if hashlib.sha256(path.read_bytes()).hexdigest() != entry["published_sha256"]:
                raise ValueError("New validation archive identity changed")
            with gzip.open(path, "rb") as stream:
                count = len(stream.read().splitlines())
            if count != entry["published_lines"] or count != entry["original_lines"]:
                raise ValueError("New validation log coverage changed")
    archives = []
    for manifest_path in [task / "workspace/logs/historical/manifest.json",
                          task / "evaluations/issue/historical_control/manifest.json"]:
        if not manifest_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text())
        observed = {}
        for item in manifest["artifacts"]:
            path = manifest_path.parent / item["artifact"]
            if hashlib.sha256(path.read_bytes()).hexdigest() != item["published_sha256"]:
                raise ValueError(f"Archive identity changed: {path.name}")
            count = 0
            with gzip.open(path, "rt") as stream:
                for count, line in enumerate(stream, 1):
                    row = json.loads(line)
                    if row["record"] != count:
                        raise ValueError("Archive record order/coverage changed")
                    match = re.fullmatch(r"(step|rollout|perf) (\d+): (\{.*\})", row["message"])
                    if match:
                        observed[match[1], int(match[2])] = ast.literal_eval(match[3])
            if count != item["published_records"] or count != item["original_records"]:
                raise ValueError("Published archive lost source record positions")
            archives.append({"artifact": item["artifact"], "records": count})
        for kind, indices in manifest["metric_coverage"].items():
            actual = sorted(i for k, i in observed if k == kind)
            if actual != indices:
                raise ValueError(f"Observed {kind} coverage differs from manifest")
            for index in indices:
                row = json.loads((manifest_path.parent / "metrics" / kind / f"{index:04d}.json").read_text())
                if row["metrics"] != observed[kind, index]:
                    raise ValueError(f"Metric values differ from archive: {kind}/{index}")
    return {"ok": True, "hashed_files": len(source["published_file_hashes"]), "archives": archives}


if __name__ == "__main__":
    print(json.dumps(check(Path(__file__).resolve().parents[1]), indent=2))
