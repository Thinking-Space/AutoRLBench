"""Preserve small immutable input assets for later independent validation."""
import json
import os
from pathlib import Path
import shutil

from input_contract import digest, regular, write_json


workspace = Path(os.environ['AUTORLBENCH_WORKSPACE']).resolve()
prepared = Path(os.environ['AUTORLBENCH_PREPARED']).resolve()
destination = prepared / 'input_assets'
destination.mkdir(parents=True, exist_ok=True)
pins = json.loads(Path(__file__).with_name('asset_pins.json').read_text())
copied = {}
for asset in pins['assets']:
    for entry in asset['files']:
        if asset['type'] == 'model':
            if entry['path'] == 'model.safetensors':
                continue
            relative = 'tokenizer/' + entry['path']
        elif entry['path'].startswith('gsm8k/'):
            relative = entry['path']
        else:
            continue
        source = regular(workspace / asset['destination'] / entry['path'], workspace)
        if digest(source) != entry['sha256']:
            raise ValueError('Input hash mismatch')
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        # Independent bytes:an agent editing its workspace must not change a
        # read-only prepared file through a shared hard-link inode.
        shutil.copyfile(source, target)
        copied[relative] = entry['sha256']
write_json(destination / 'manifest.json', copied)
