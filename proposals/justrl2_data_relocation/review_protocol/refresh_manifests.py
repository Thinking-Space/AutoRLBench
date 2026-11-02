"""Rebuild concise maintainer file inventories for the paired tasks."""
import hashlib
import json
from pathlib import Path


def refresh(task):
    destination = task / 'provenance/source_manifest.json'
    files = {}
    for path in sorted(task.rglob('*')):
        if path.is_file() and path != destination and '__pycache__' not in path.parts and path.suffix != '.pyc':
            files[path.relative_to(task).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    receipt = {'schema_version': 3, 'scope': 'Current task files, excluding this inventory', 'files': files}
    destination.write_text(json.dumps(receipt, indent=2) + '\n')
    return len(files)


if __name__ == '__main__':
    repo = Path(__file__).resolve().parents[3]
    for name in ('justrl2_data_relocation', 'justrl2_data_review_control'):
        print(name, refresh(repo / 'tasks/data' / name))
