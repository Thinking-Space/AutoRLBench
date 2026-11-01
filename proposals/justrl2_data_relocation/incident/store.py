"""Trusted data reader; editable loader chooses location, never payload or reward."""
import json
import os
from pathlib import Path
import subprocess
import sys

from loader_contract import read_resolved


class SceneStore:
    def __init__(self, root):
        self.root = Path(root)
        self.rows = json.loads((self.root / 'manifest.json').read_text())['rows']
        self.loader = Path(os.environ['RELOCATION_LOADER'])

    def resolve(self, root, record):
        call=subprocess.run([sys.executable,'-I',str(Path(__file__).with_name('resolve_worker.py')),str(self.loader)],
                            input=json.dumps({'root':root,'record':record}),text=True,capture_output=True,
                            timeout=30)
        if call.returncode:
            raise RuntimeError('Resolver process failed: '+call.stderr[-2000:])
        result=json.loads(call.stdout)
        if not result['ok']:
            if result['kind']=='missing':raise FileNotFoundError(2,result['message'],record['path'])
            raise RuntimeError('Resolver error: '+result.get('type','')+': '+result['message'])
        return result['path']

    def read(self, record, *, sample_index=None, step=None):
        # read_resolved opens and verifies bytes independently of loader code.
        for attempt in range(1,4):
            try:
                payload, resolved = read_resolved(self.root, record, self.resolve)
            except FileNotFoundError:
                self.trace(record,sample_index,step,attempt,'missing')
                if attempt==3:raise
                continue
            self.trace(record,sample_index,step,attempt,'ok',resolved)
            return payload

    def trace(self, record, index, step, attempt, outcome, resolved=None):
        output=os.environ.get('RELOCATION_OUTPUT')
        if output is None:return
        path=Path(output)/'read_attempts.jsonl';path.parent.mkdir(parents=True,exist_ok=True)
        event={'id':record['id'],'index':index,'step':step,'attempt':attempt,'outcome':outcome,
               'path':resolved,'sha256':record['sha256'] if outcome=='ok' else None}
        with path.open('a') as f:f.write(json.dumps(event)+'\n')
