"""Maintainer reference repair; kept outside the answer workspace."""
import json
from pathlib import Path


def resolve(root, record):
    root = Path(root)
    catalog = json.loads((root / 'catalog.json').read_text())
    locations = catalog['rows']
    candidates = [row for row in locations if row['id'] == record['id']
                  and row['sha256'] == record['sha256']]
    if len(candidates) != 1:
        raise FileNotFoundError('No unique current location for requested sample')
    path = candidates[0]['path']
    if not (root / path).is_file():
        raise FileNotFoundError('Current backing file is unavailable')
    return path
