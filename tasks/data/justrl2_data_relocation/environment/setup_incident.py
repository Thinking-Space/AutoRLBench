"""Prepare-time only: install the trusted engine and already-failed scene."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def main():
    environment=Path(os.environ['AUTORLBENCH_ENVIRONMENT'])
    workspace=Path(os.environ['AUTORLBENCH_WORKSPACE'])
    prepared=Path(os.environ['AUTORLBENCH_PREPARED'])
    engine=prepared/'incident_engine'
    if engine.exists():
        shutil.rmtree(engine)
    shutil.copytree(environment/'incident_engine',engine,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    sys.path.insert(0,str(engine))
    from materialize import install
    snapshot=environment/'snapshot.json'
    pin=json.loads(snapshot.read_text())
    source=os.environ.get('RELOCATION_SCENE_SOURCE')
    datasets=os.environ.get('AUTORLBENCH_DATASETS')
    cached=Path(datasets)/'incidents'/pin['scene_sha256'] if datasets else None
    if not source and cached is not None and cached.is_dir():
        source=cached
    if not source:
        if not pin.get('archive_url'):
            expected=str(cached) if cached is not None else '$AUTORLBENCH_DATASETS/incidents/'+pin['scene_sha256']
            raise RuntimeError('Provide RELOCATION_SCENE_SOURCE or copy the pinned scene to '+expected+
                               ' and pass --datasets-dir to the harness. Setup never trains a new checkpoint.')
        source=prepared/'scene.tar'
        subprocess.run(['autorlbench_fetch_url','--url',pin['archive_url'],'--sha256',pin['archive_sha256'],
                        '--size',str(pin['archive_size']),'--dest',str(source)],check=True)
    shutil.copyfile(snapshot,engine/'snapshot.json')
    scene=install(source,workspace,prepared,snapshot)
    (prepared/'env').write_text('RELOCATION_SCENE='+str(scene)+'\n')
    print('Incident installed; checkpoint3 and stale data mapping are available before agent entry.',flush=True)


if __name__=='__main__':main()
