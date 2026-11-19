#!/usr/bin/env python3
"""Invoke the unchanged shared fetcher with a pinned model revision."""
import os
from pathlib import Path
import fetch_hf  # disable Xet before importing huggingface_hub
import huggingface_hub

REVISION='c202236235762e1c871ad0ccb60c8ee5ba337b9a'
original= huggingface_hub.snapshot_download

def pinned_download(**kwargs):
    return original(revision=REVISION, **kwargs)

if __name__ == '__main__':
    huggingface_hub.snapshot_download=pinned_download
    raise SystemExit(fetch_hf.main([
        '--repo','Qwen/Qwen3.5-9B',
        '--cache-as','Qwen/Qwen3.5-9B-'+REVISION,
        '--dest',str(Path(os.environ['AUTORLBENCH_WORKSPACE'])/'staged-model/Qwen3.5-9B')]))
