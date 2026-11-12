"""Verify pinned runtime input files after cache materialization."""
import hashlib
import json
import os
from pathlib import Path


root = Path(os.environ['AUTORLBENCH_WORKSPACE'])
manifest = json.loads(Path(__file__).with_name('asset_pins.json').read_text())
for asset in manifest['assets']:
    for item in asset['files']:
        path = root / asset['destination'] / item['path']
        assert path.is_file() and not path.is_symlink(), path
        assert path.stat().st_size == item['bytes'], path
        with path.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        assert digest == item['sha256'], path
print('ASSET_INPUTS_VERIFIED')
