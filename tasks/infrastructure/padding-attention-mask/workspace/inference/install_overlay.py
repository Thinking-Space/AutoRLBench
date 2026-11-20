#!/usr/bin/env python3
"""Create a private vLLM overlay after verifying the captured base hashes."""
import argparse
import base64
import hashlib
from importlib import metadata
import json
from pathlib import Path
import shutil
import sys
import tempfile

HERE=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=HERE/'runtime')
    p.add_argument('--check',action='store_true',help='Verify compatibility without writing')
    a=p.parse_args()
    manifest=json.loads((HERE/'manifest.json').read_text())
    try: dist=metadata.distribution('vllm')
    except metadata.PackageNotFoundError: p.error('Install the matching vLLM build first')
    if dist.version!=manifest['version']: p.error('Requires vLLM '+manifest['version']+', found '+dist.version)
    source=Path(dist.locate_file('vllm')).resolve()
    problems=[]
    for item in manifest['files']:
        packaged=HERE/'vllm_patch'/item['path']
        if hashlib.sha256(packaged.read_bytes()).hexdigest()!=item['sha256']:
            problems.append(item['path']+': packaged patch hash mismatch');continue
        installed=Path(dist.locate_file(item['path']))
        if not installed.exists():
            if item['base_record_hash'] is not None: problems.append(item['path']+': missing baseline file')
            continue
        blob=installed.read_bytes(); h=hashlib.sha256(blob).digest()
        baseline='sha256='+base64.urlsafe_b64encode(h).decode().rstrip('=')
        if h.hex()!=item['sha256'] and baseline!=item['base_record_hash']:
            problems.append(item['path']+': baseline differs from captured build')
    if problems: p.error('Refusing an incompatible overlay: '+ '; '.join(problems))
    print('Verified',len(manifest['files']),'patch modules against vLLM',dist.version)
    if a.check: return 0
    output=a.output.expanduser().resolve()
    if output.exists(): p.error('Output exists; choose a new --output or remove the old overlay explicitly')
    if output==source or source in output.parents: p.error('Output must be outside the installed package')
    output.parent.mkdir(parents=True,exist_ok=True)
    tmp=Path(tempfile.mkdtemp(prefix='.vllm-overlay-',dir=output.parent))
    try:
        def copyfile(src,dst):
            # Native libraries stay linked to the installed environment. Python is isolated.
            if '.so' in Path(src).name: Path(dst).symlink_to(Path(src).resolve())
            else: shutil.copy2(src,dst)
        shutil.copytree(source,tmp/'vllm',copy_function=copyfile,ignore=shutil.ignore_patterns('__pycache__','*.pyc','*.bak*'))
        for item in manifest['files']:
            target=tmp/item['path'];target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(HERE/'vllm_patch'/item['path'],target)
        (tmp/'overlay_manifest.json').write_text(json.dumps(manifest,indent=2))
        tmp.rename(output)
    except BaseException:
        shutil.rmtree(tmp,ignore_errors=True);raise
    print('Overlay:',output)
    print('The bundled launchers automatically prefer training/inference/runtime.')
    print('For a custom location, set OPQAD_VLLM_OVERLAY to the output directory.')
    return 0
if __name__=='__main__': raise SystemExit(main())
