#!/usr/bin/env python3
"""Convert SFT mix data (0415template, 6264 records) into RL-ready prompts.

WHY THIS EXISTS
---------------
exp01-06 trained on 264 KernelBench prompts (200 L1L2 + 64 easy_injection),
all sharing the same RL prompt template (single-stage "Convert this PyTorch
module to CUDA"). The new 0415template SFT data adds 6264 unique target
PyTorch modules covering more compound operators (e.g. ConvTranspose3d +
GELU + Softplus, fused multi-op patterns) that exp01-06 never saw.

The SFT data is in `{question, response}` schema where the question is a
"multi-stage CUDA optimization plan" prompt -- not directly RL-runnable
because:

  1. RL training expects `{question, final_answer: {torch_function}, metadata}`
  2. RL reward function `compute_score(solution, ground_truth)` reads
     `ground_truth["torch_function"]` to verify against the sandbox.
  3. exp01-06's prompt template is a single-stage "convert to CUDA" prompt;
     keeping the multi-stage one would make exp07 incomparable to exp06.

This script:
  (a) auto-derives the canonical RL prompt template from existing RL data,
  (b) extracts the target `class Model(nn.Module)` block from each SFT
      question,
  (c) wraps it in the RL template to produce a fresh RL prompt, and
  (d) builds the matching `final_answer.torch_function` + `metadata`
      fields so the result is drop-in for the existing RL training loop.

Output: data/train/kernel_bench_sft_mix4k_extracted.jsonl, intended to be
mixed with the existing 200+64 RL prompts at training time via the new
EXTRA_TRAIN_FILE env var on run-qwen3-8b-16gpu-kernel-bench.sh.

WHAT TO COMPARE (exp07 vs exp06)
--------------------------------
Same backbone (cosine LR, ent=0.005, temp=0.55, mixed_group filter,
over-sampling=32). Only TRAIN_FILES differs: exp07 adds this file as
EXTRA_TRAIN_FILE on top of L1L2 + easy_injection. If reward / effective
ratio improves, the new data adds signal; if it tanks, the new data is
either too hard or distributionally off.
"""
import json
import os
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = REPO_ROOT.parent / "data"

SFT_PATH = DATA_ROOT / "sft" / "0415template" / "raw" / "sft_data_mix4k_union.sample_6264.jsonl"
RL_REF_PATH = DATA_ROOT / "train" / "kernel_bench_dataL1L2.jsonl"
OUT_PATH = DATA_ROOT / "train" / "kernel_bench_sft_mix4k_extracted.jsonl"

MARKER = "Now, write the CUDA equivalent of the following PyTorch module:"

# ---------------------------------------------------------------- helpers


def derive_rl_template(rl_ref_path: Path) -> tuple[str, str]:
    """Return (prefix, suffix) such that prefix + "\n```python\n<class>\n```" + suffix
    yields a question identical in structure to existing RL data.

    Verified on L1L2 (200) and easy_injection (64): both have a single
    unique prefix + suffix, so we can reuse them verbatim.
    """
    with rl_ref_path.open() as f:
        line = f.readline()
        d = json.loads(line)
        q = d["question"]

    idx = q.find(MARKER)
    if idx == -1:
        raise RuntimeError(f"marker {MARKER!r} not found in RL reference question")
    pe = idx + len(MARKER)
    post = q[pe:]
    m = re.search(r"\s*```python\s*\n(.*?)```", post, re.DOTALL)
    if not m:
        raise RuntimeError("no python block after marker in RL reference question")
    return q[:pe], q[pe + m.end():]


def extract_target_class(sft_question: str) -> str | None:
    """Find the *target* PyTorch class that the SFT prompt asks the user to
    optimize. Heuristic: look for a ```python ... ``` block that defines
    `class Model(nn.Module)` and the KernelBench-conventional helpers
    `def get_inputs()` and `def get_init_inputs()`. The 0415template SFT
    data hits this in 100% of samples (verified separately).

    Returns the class block as plain text without the ```python fences,
    or None if the SFT question doesn't conform.
    """
    pat = re.compile(r"```python\s*\n(.*?)```", re.DOTALL)
    for block in pat.findall(sft_question):
        if (
            "class Model(nn.Module)" in block
            and "def get_inputs" in block
            and "def get_init_inputs" in block
        ):
            return block.strip()
    return None


def parse_stage_from_sft(sft_question: str) -> dict:
    """Best-effort parse of the multi-stage metadata from the SFT prompt.
    Returned dict goes into metadata.extra_info purely for offline analysis.
    Empty dict on failure -- training does not depend on these fields.
    """
    info: dict = {}
    m = re.search(r"You are implementing stage (\d+) of a multi-stage", sft_question)
    if m:
        info["sft_stage"] = int(m.group(1))
    m = re.search(r"算子:\s*(.+?)\n", sft_question)
    if m:
        info["sft_op"] = m.group(1).strip()[:120]
    return info


# ---------------------------------------------------------------- main


def main():
    if not RL_REF_PATH.exists():
        sys.exit(f"missing RL reference: {RL_REF_PATH}")
    if not SFT_PATH.exists():
        sys.exit(f"missing SFT mix file: {SFT_PATH}")

    prefix, suffix = derive_rl_template(RL_REF_PATH)
    print(f"[template] prefix={len(prefix)}B suffix={len(suffix)}B "
          f"(derived from {RL_REF_PATH.name})")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    n_in = 0
    n_ok = 0
    n_skip_no_class = 0
    n_skip_other = 0
    seen_classes: set[str] = set()

    with SFT_PATH.open() as fin, OUT_PATH.open("w") as fout:
        for line in fin:
            n_in += 1
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                n_skip_other += 1
                continue
            sft_q = d.get("question", "")
            cls = extract_target_class(sft_q)
            if cls is None:
                n_skip_no_class += 1
                continue
            # de-dup on the class text itself: SFT mix may carry multiple
            # responses to the same module (different stages). For RL we
            # only need the prompt once.
            if cls in seen_classes:
                continue
            seen_classes.add(cls)

            new_q = prefix + "\n```python\n" + cls + "\n```" + suffix
            extra = {"index": n_ok, "name": "Model"}
            extra.update(parse_stage_from_sft(sft_q))

            out = {
                "question": new_q,
                "final_answer": {"torch_function": cls},
                "metadata": {
                    "data_source": "sft_mix4k_extracted",
                    "name": "Model",
                    "ability": "code",
                    "reward_model": {"style": "rule"},
                    "extra_info": extra,
                },
            }
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            n_ok += 1

    print(f"[done] read={n_in}  written={n_ok}  "
          f"skipped_no_class={n_skip_no_class}  skipped_other={n_skip_other}")
    print(f"[done] unique classes (= RL prompts) = {len(seen_classes)}")
    print(f"[done] output: {OUT_PATH}  ({OUT_PATH.stat().st_size/1e6:.1f} MB)")

    # ---- self-test: re-read random samples and verify schema is RL-loadable
    print("\n[self-test] re-loading 5 random samples to verify schema ...")
    import random
    with OUT_PATH.open() as f:
        lines = f.readlines()
    rng = random.Random(0)
    for i in rng.sample(range(len(lines)), min(5, len(lines))):
        s = json.loads(lines[i])
        assert isinstance(s["question"], str) and len(s["question"]) > 1000, "bad q"
        assert isinstance(s["final_answer"], dict) and "torch_function" in s["final_answer"], "bad fa"
        tf = s["final_answer"]["torch_function"]
        assert "class Model(nn.Module)" in tf, "bad torch_function"
        assert "def get_inputs" in tf and "def get_init_inputs" in tf, "missing kbench helpers"
        # also: the question must literally contain torch_function (we just put it there)
        assert tf in s["question"], f"torch_function not embedded in question (i={i})"
        print(f"  sample {i}: q_len={len(s['question'])}  tf_len={len(tf)}  meta={list(s['metadata'].keys())}")

    print("\n[self-test] OK")


if __name__ == "__main__":
    main()
