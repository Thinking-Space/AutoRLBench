#!/usr/bin/env python3
"""Contributor utility: verify remote dataset bytes and pin an uploaded commit.

This is not called during benchmark prepare: prepare never follows mutable main.
The repository root contains route/, images_with_box/, images_with_zwz/ and
tool_results/. Tool-result remote/local paths are declared in file_mappings.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import datetime
import hashlib
import json
from pathlib import Path
import re
import time
import urllib.error
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parent


def read_url(url):
    request = urllib.request.Request(url, headers={'User-Agent': 'maestro-autorlbench-packager'})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                return response.read(), response.headers
        except (urllib.error.URLError, TimeoutError) as error:
            if isinstance(error, urllib.error.HTTPError) and error.code not in (429, 500, 502, 503, 504):
                raise
            if attempt == 2:
                raise
            # Retry only transient transport failures, never size/hash failures.
            time.sleep(attempt + 1)


def pin(revision):
    path = ROOT / 'data_manifest.json'
    manifest = json.loads(path.read_text())
    dataset = manifest['download']
    base = 'https://huggingface.co/api/datasets/' + dataset['repo']
    raw, _ = read_url(base + '/revision/' + urllib.parse.quote(revision, safe=''))
    commit = json.loads(raw)['sha']
    if not re.fullmatch('[0-9a-f]{40}', commit):
        raise ValueError('Hugging Face did not return a full commit SHA')
    print(f'[pin] Inspecting {dataset["repo"]}@{commit}', flush=True)
    url = base + '/tree/' + commit + '?recursive=true&expand=false&limit=1000'
    remote = {}
    while url:
        raw, headers = read_url(url)
        for entry in json.loads(raw):
            if entry['type'] == 'file':
                remote[entry['path']] = entry
        match = re.search(r'<([^>]+)>;\s*rel="next"', headers.get('Link', ''))
        url = match.group(1) if match else None
        if url and not url.startswith(base + '/tree/'):
            raise ValueError('unexpected dataset pagination URL')
    print(f'[pin] Listed {len(remote)} remote files; verifying recorded assets.', flush=True)
    expected = []
    mappings = dataset.get('file_mappings', {})
    for entry in manifest['files']:
        name = mappings.get(entry['path'])
        if name is None:
            name = str(Path(entry['path']).relative_to(dataset.get('destination', 'data')))
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('unsafe remote dataset path: ' + name)
        expected.append((name, entry))
    missing = [name for name, _ in expected if name not in remote]
    if missing:
        raise ValueError(f'{dataset["repo"]}@{commit} is missing {len(missing)} files '
                         f'(first five: {missing[:5]}). Manifest was not changed.')

    def verify(item):
        name, entry = item
        info = remote[name]
        if info['size'] != entry['bytes']:
            raise ValueError('remote dataset size mismatch: ' + name)
        sha = info.get('lfs', {}).get('oid')
        if sha is None:
            url = ('https://huggingface.co/datasets/' + dataset['repo'] + '/resolve/'
                   + commit + '/' + urllib.parse.quote(name, safe='/'))
            raw, _ = read_url(url)
            sha = hashlib.sha256(raw).hexdigest()
        if sha != entry['sha256']:
            raise ValueError('remote dataset SHA-256 mismatch: ' + name)

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(verify, expected))
    dataset.update(revision=commit, status='ready', verified_remote_files=len(expected))
    dataset['checked_date'] = datetime.datetime.now(datetime.timezone.utc).date().isoformat()
    dataset.pop('observed_remote_files', None)
    dataset['revision_provenance'] = 'All remote data and tool-result file sizes and SHA-256 match the recorded input manifest.'
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n')
    print(f'Pinned {dataset["repo"]}@{commit}: {len(expected)} files verified.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--revision', required=True, help='uploaded commit SHA, or main to resolve once')
    args = parser.parse_args()
    pin(args.revision)
