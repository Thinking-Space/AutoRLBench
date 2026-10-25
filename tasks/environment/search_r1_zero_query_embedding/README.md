# Zero query vector in retrieval

Training Qwen2.5-1.5B-Instruct with GRPO on HotpotQA through Search-R1. The query vector stays all zeros, so every query retrieves the same documents.

| | |
| --- | --- |
| **Author** | AutoRLBench |
| **Domain** | environment |
| **Tags** | `verl` `grpo` `hotpotqa` `query_vector` |
| **Model** | Qwen2.5-1.5B-Instruct · e5-base-v2 retriever |
| **Dataset** | HotpotQA distractor, 1024 train / 128 held-out, 12,468-passage corpus |
| **Framework** | Search-R1 with vendored veRL `598e61bd`, FSDP + vLLM 0.6.3 |
| **Environment** | `search-r1-verl` (not yet built) · 1 × A100-40G |
| **Evaluations** | issue · solution |
| **Recorded runs** | 1 defective + 1 control run, same host, seed, data and budget (32 steps) |

## Difficulty

`Search-R1/search_r1/search/retrieval_server.py::Encoder.encode` (lines 117–120) writes the GPU embedding into a tensor made with `torch.as_tensor(cpu_buffer, device=cuda)`. Across devices, that call copies instead of aliasing, so the write lands in a temporary and the function returns the untouched zero buffer. Under inner-product search every document scores `0.0`, and FAISS returns the same arbitrary top-3 for every query.

Shapes and dtypes are valid, and nothing warns. The policy still learns: `val/em` rises from 3.1% to 15.6%, search and format rates stay near 1.0. That makes "the run did not learn" the wrong diagnosis. The real signal is that the identical recipe with the embedding copied to host reaches 32.8%, and the gap is already open at step 0 (3.1% vs 13.3%). A correct copy of the same encoder exists in `search_r1/search/retrieval.py`, which gives the cleanest diagnostic path.

## Reference solution

- **issue**: identify the unwritten host buffer in `Encoder.encode` as the reason the query never crosses devices. "Encoder on GPU, index on CPU" alone, "no learning", or blaming recipe settings (`max_turns`, `topk`, FSDP offload, the 32-step budget) does not pass.
- **solution**: replace the four lines with `query_emb.detach().cpu().numpy().astype(np.float32, order="C")`.

## Verification

- **issue**: an LLM judge scores verdict / root cause / evidence, weighted 0.2 / 0.6 / 0.2. Passing requires ≥ 0.8.
- **solution**: the agent repairs the workspace and submits `report.md` without training; a missing report scores 0. The harness then reruns `run.py --run-name graded` per `run_spec.json` for 32 steps with eval every 10, and grades only that run. The best held-out `val/em` must be ≥ 0.24. That threshold sits 0.084 above the defective run's best (0.156), 0.088 below the control (0.328), and above the control's untrained step 0 (0.133). It is calibrated on one run per side, so treat scores near 0.24 as untested territory.
