# SPDX-License-Identifier: Apache-2.0
"""Self-contained pinned source archive utilities for this case."""
import gzip
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile

HERE = Path(__file__).resolve().parent
EVIDENCE = HERE
REVISION = '75814318f243af7e39af337c1173fc3927c9635b'

def sha(data):
    return hashlib.sha256(data).hexdigest()

def inventory(root):
    return {str(p.relative_to(root)): sha(p.read_bytes()) for p in sorted(root.rglob('*')) if p.is_file()}

def replace_once(text, before, after):
    if text.count(before) != 1:
        raise ValueError('Source anchor missing or ambiguous')
    return text.replace(before, after, 1)

def read_source(source):
    if (source / '.git').exists():
        actual = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
        if actual != REVISION:
            raise ValueError('Source HEAD differs from pinned revision')
        data = subprocess.check_output(['git', '-C', str(source), 'archive', REVISION])
        with tarfile.open(fileobj=io.BytesIO(data)) as archive:
            files = {}
            for member in archive.getmembers():
                if member.isdir():
                    continue
                if not member.isfile() or member.name.startswith('/') or '..' in Path(member.name).parts:
                    raise ValueError('Source contains a link or invalid archive path')
                files[member.name] = archive.extractfile(member).read()
    else:
        if any((p.is_symlink() for p in source.rglob('*'))):
            raise ValueError('Source tree must not contain symlinks')
        files = {str(p.relative_to(source)): p.read_bytes() for p in source.rglob('*') if p.is_file()}
    pins = json.loads((EVIDENCE / 'source_files.json').read_text())['files']
    if {name: sha(data) for (name, data) in files.items()} != pins:
        raise ValueError('Source file inventory differs from pinned public revision')
    return files

def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + '\n')

def archive_workspace(workspace, destination):
    with destination.open('wb') as raw, gzip.GzipFile(fileobj=raw, mode='wb', filename='', mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode='w') as archive:
            for p in sorted(workspace.rglob('*')):
                if not p.is_file():
                    continue
                data = p.read_bytes()
                info = tarfile.TarInfo(str(p.relative_to(workspace)))
                (info.size, info.mode, info.mtime) = (len(data), 420, 0)
                archive.addfile(info, io.BytesIO(data))
