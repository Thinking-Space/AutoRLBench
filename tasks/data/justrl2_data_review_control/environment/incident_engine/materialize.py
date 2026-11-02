"""Install a pinned incident artifact before the agent starts, without training."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

from scene import copy_scene, verify_scene


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):digest.update(block)
    return digest.hexdigest()


def install(source, workspace, prepared, lock):
    source,workspace,prepared=map(Path,(source,workspace,prepared))
    pin=json.loads(Path(lock).read_text())
    destination=prepared/'incident'
    if destination.exists():
        verify_scene(destination)
        if sha(destination/'SCENE.json')!=pin['scene_sha256']:raise ValueError('Installed scene differs from artifact pin')
    elif source.is_dir():
        if sha(source/'SCENE.json')!=pin['scene_sha256']:raise ValueError('Source scene differs from artifact pin')
        copy_scene(source,destination)
    else:
        if sha(source)!=pin['archive_sha256'] or source.stat().st_size!=pin['archive_size']:
            raise ValueError('Scene archive pin mismatch')
        destination.mkdir(parents=True)
        with tarfile.open(source) as archive:archive.extractall(destination,filter='data')
        verify_scene(destination)
        if sha(destination/'SCENE.json')!=pin['scene_sha256']:raise ValueError('Extracted scene differs from pin')
    link=workspace/'incident'
    if link.is_symlink():
        if link.resolve()!=destination.resolve():raise ValueError('Unexpected existing incident link')
    elif link.exists():raise FileExistsError('Workspace incident already exists')
    else:link.symlink_to(destination.resolve(),target_is_directory=True)
    return destination


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['source','workspace','prepared','lock']:p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args();print(install(args.source,args.workspace,args.prepared,args.lock))


if __name__=='__main__':main()
