"""Stage bundled data and the trusted query engine; no network or models."""
import os
from pathlib import Path
import shutil
import sys


def main():
    environment = Path(os.environ['AUTORLBENCH_ENVIRONMENT']).resolve()
    workspace = Path(os.environ['AUTORLBENCH_WORKSPACE']).resolve()
    prepared = Path(os.environ['AUTORLBENCH_PREPARED']).resolve()
    sys.path.insert(0, str(environment / 'incident_engine'))
    from dataset import verify_dataset
    snapshot = environment / 'snapshot.json'
    verify_dataset(environment / 'scene', snapshot)
    prepared.mkdir(parents=True, exist_ok=True)
    workspace.mkdir(parents=True, exist_ok=True)
    scene = prepared / 'incident'
    if not scene.exists():
        shutil.copytree(environment / 'scene', scene)
    verify_dataset(scene, snapshot)
    engine = prepared / 'incident_engine'
    if engine.is_symlink():
        raise ValueError('Engine must not be a symlink')
    shutil.copytree(environment / 'incident_engine', engine, dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copyfile(snapshot, prepared / 'snapshot.json')
    link = workspace / 'incident'
    if link.is_symlink():
        if link.resolve() != scene.resolve():
            raise ValueError('Unexpected existing scene link')
    elif link.exists():
        raise FileExistsError('Workspace scene already exists')
    else:
        link.symlink_to(scene, target_is_directory=True)
    print('Bundled data ready: 64 training and 16 evaluation records; no checkpoint required.')


if __name__ == '__main__':
    main()
