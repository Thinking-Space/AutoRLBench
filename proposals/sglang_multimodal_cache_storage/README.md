# Multimodal rollout cache retains entire vision-encoder batches

A buggy online GRPO run and a matched repair are included for maintainer review. On one H200 in an AutoRLBench shared image, the upstream SGLang cache held **864 KiB under a 360 KiB budget** at step 2. The repaired cache held **360 KiB**. Both runs completed 16 policy updates with the public SmolVLM-256M-Instruct checkpoint and 1,024 sampled answers per arm. [Experiment, code and rerun commands](online_rl/README.md) · [H200 evidence](online_rl/results/h200/paired.json)

## Proposed infrastructure task

Diagnose why a byte-budgeted image-embedding LRU can hold more storage than it reports, then restore its bound while preserving useful cache hits, retrieved values, and the training trajectory. In the H200 run, a single surviving image retained the whole backing allocation of its encoder batch. The cache continued to report the correct sum of logical entry sizes, and the actor could still train. The detailed trace separates this cache from the allocator and text KV cache. [Storage groups and training updates, step by step](online_rl/results/h200/buggy.jsonl) · [Reference change](online_rl/reference_cache.py)

## Earlier CPU mechanism probe

The original probe uses 16 independently allocated encoder batches and isolates how references from contiguous split views retain storage. Under a 256 KiB budget, it measured 16 MiB retained on CPU, compared with 256 KiB after the fix. That 64× factor belongs to this constructed probe. The H200 RL run measured a 2.4× peak on its own workload. [CPU probe output](measured.json)

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
