"""Fetch pinned assets with streaming size/SHA checks, or copy matching caches."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import urllib.parse
import urllib.request

workspace=Path(os.environ.get('AUTORLBENCH_WORKSPACE','/workspace')).resolve()
here=Path(__file__).parent
resources=workspace/'resources';resources.mkdir(exist_ok=True)
def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
for asset in json.loads((here/'assets.json').read_text()):
    destination=resources/('model' if asset['kind']=='model' else 'raw')
    cache=Path(os.environ.get('AUTORLBENCH_MODELS' if asset['kind']=='model' else 'AUTORLBENCH_DATASETS','/autorlbench/cache'))
    for row in asset['files']:
        if row['path']=='.gitattributes':continue
        path=destination/row['path'];path.parent.mkdir(parents=True,exist_ok=True)
        if path.is_file() and path.stat().st_size==row['bytes'] and sha(path)==row['sha256']:continue
        if path.exists():raise ValueError('existing asset mismatch '+str(path))
        temp=path.with_suffix(path.suffix+'.partial')
        candidates=[cache/row['sha256'],cache/asset['repository']/asset['revision']/row['path']]
        cached=next((p for p in candidates if p.is_file() and p.stat().st_size==row['bytes'] and sha(p)==row['sha256']),None)
        if cached:
            with cached.open('rb') as src,temp.open('wb') as dst:shutil.copyfileobj(src,dst)
        else:
            url='https://huggingface.co/'+asset['repository']+'/resolve/'+asset['revision']+'/'+urllib.parse.quote(row['path'],safe='/')
            with urllib.request.urlopen(url,timeout=120) as src,temp.open('wb') as dst:shutil.copyfileobj(src,dst,1024*1024)
        if temp.stat().st_size!=row['bytes'] or sha(temp)!=row['sha256']:raise ValueError('download checksum '+row['path'])
        temp.replace(path)
subprocess.run([sys.executable,str(here/'prepare_data.py'),'--raw-root',str(resources/'raw'),'--output',str(resources)],check=True)
