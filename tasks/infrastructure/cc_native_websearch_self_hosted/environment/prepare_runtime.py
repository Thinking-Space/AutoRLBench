#!/usr/bin/env python3
"""Verify inputs and fetch the public CC Linux artifact without credentials."""
import hashlib,json,os,urllib.request
from pathlib import Path

SIZE=266972344
DIGEST='674deba0bba9aee479f4503c438b49303b6fdd304b9a471b14977aaec5215453'
URL='https://downloads.claude.ai/claude-code-releases/2.1.300/linux-x64/claude'

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(4*1024*1024),b''):h.update(chunk)
    return h.hexdigest()

if __name__ == '__main__':
    w=Path(os.environ['AUTORLBENCH_WORKSPACE'])
    e=Path(os.environ['AUTORLBENCH_ENVIRONMENT'])
    # Check runtime assets from the recorded model; README may change upstream
    # without affecting execution and is intentionally excluded from this check.
    for item in json.loads((w/'assets/model-files-sha256.json').read_text()):
        if item['path'] in ('README.md','LICENSE'):continue
        p=w/'staged-model/Qwen3.5-9B'/item['path']
        assert p.stat().st_size==item['bytes'],item['path']
        assert digest(p)==item['sha256'],item['path']
    data=json.loads((e/'data_manifest.json').read_text())
    assert digest(w/data['file'])==data['sha256']
    target=w/'code/bin/claude';target.parent.mkdir(parents=True,exist_ok=True)
    if not (target.is_file() and target.stat().st_size==SIZE and digest(target)==DIGEST):
        temporary=target.with_suffix('.partial')
        h=hashlib.sha256();size=0
        with urllib.request.urlopen(URL,timeout=90) as source,temporary.open('wb') as out:
            for chunk in iter(lambda:source.read(4*1024*1024),b''):
                out.write(chunk);h.update(chunk);size+=len(chunk)
        assert size==SIZE and h.hexdigest()==DIGEST,'CC artifact size/hash mismatch'
        temporary.replace(target)
    target.chmod(0o755)
    print('Recorded runtime model files, QA data and public CC artifact verified.')
