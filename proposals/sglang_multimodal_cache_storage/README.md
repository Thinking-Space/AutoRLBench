## Proposed infrastructure task

A multimodal RL rollout worker uses SGLang to batch vision-encoder misses. Its embedding LRU has a byte budget. On the branch where the encoder returns one concatenated tensor, the scheduler splits that tensor by image and inserts the resulting views into the LRU. In the recorded source revision, insertion charges each view for its own elements and keeps the view object. An entry can therefore keep the entire encoder batch's storage alive after the request ends. This is separate from the text prefix KV cache.

We traced this path while investigating a real robotics RL workload. A controlled CPU replay of the cache implementation measured the following with 16 independent FP32 batches of shape `(512, 512)`, keeping eight rows from each batch under distinct keys.

| Measurement | Original cache | Selective compact-copy repair |
| --- | ---: | ---: |
| Entries and configured budget | 16, 256 KiB | 16, 256 KiB |
| Bytes retained by distinct tensor storages | 16 MiB | 256 KiB |
| Retrieved embeddings | Reference | Same values, shape, dtype and device |

The 64× factor applies to the CPU replay. Whole-GPU memory savings and an RL job OOM have not been established. For scoring, sum the bytes of *distinct* backing storages referenced by the live cache, grouped by device. A slice may already be contiguous while retaining the original storage, so calling `.contiguous()` alone can preserve the defect. The tested repair makes a compact copy when the backing storage is larger than the logical tensor and preserves the cache entry's metadata.

For a full task, we propose a short real multimodal RL run in a listed shared image. The workload should exercise the concatenated-output branch, retain per-image entries from multiple completed batches, and repeat some images to produce cache hits. Both arms would record cache keys, logical bytes, deduplicated storage bytes and hit/output checks. This would let a verifier distinguish a genuine repair from shrinking or disabling the cache. The original cluster run used a different image and did not instrument retained bytes per entry. This proposal supplies the root-cause evidence and the CPU replay.

The historical cache source matches the pinned Apache-2.0 upstream file byte-for-byte (SHA-256 `f1415edeb4554534de2648e2ee1ffacde06236a7b0028f29f94b9b94a954b739`). [Cache insertion and accounting](https://github.com/sgl-project/sglang/blob/02b5e123c490c988d21526d0dd7837fefdd0e1da/python/sglang/srt/mem_cache/multimodal_cache.py#L102-L121) · [Encoder batch split and cache insertion](https://github.com/sgl-project/sglang/blob/02b5e123c490c988d21526d0dd7837fefdd0e1da/python/sglang/srt/managers/mm_schedule.py#L391-L403)

### Reproduce the CPU measurement

Start in the directory containing `reproduce_cpu.py`. The probe runs on CPU using Python 3.10+ and PyTorch. If PyTorch is already installed, skip the virtual-environment setup.

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install torch
```

```bash
curl --fail --location --silent --show-error \
  https://raw.githubusercontent.com/sgl-project/sglang/02b5e123c490c988d21526d0dd7837fefdd0e1da/python/sglang/srt/mem_cache/multimodal_cache.py \
  --output multimodal_cache.upstream.py
python reproduce_cpu.py --upstream multimodal_cache.upstream.py
```

The probe verifies the downloaded file's SHA-256 before loading it. The allocator imported solely for an abstract type annotation is supplied as a placeholder, so the full SGLang stack is optional. Both arms exercise the upstream cache. The control compacts oversized inputs immediately before calling the same cache insertion method. Compare the byte counts with `measured.json` (recorded on PyTorch 2.14.0).
