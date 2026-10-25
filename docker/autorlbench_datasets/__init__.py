"""Turn upstream assets into the shape a task needs: verl-format parquet, or an assembled retrieval index.

Why this step exists: a task's launch script wants `/workspace/<name>_verl/{train,test}.parquet`,
and verl's `RLHFDataset` requires each row to have

    prompt                    list of chat messages (not a string)
    data_source               exact string; selects which scorer to route to
    reward_model.ground_truth the ground truth

(see workspace/verl/utils/dataset/rl_dataset.py:313 and
  workspace/verl/utils/reward_score/__init__.py:127,142)

Upstream `openai/gsm8k` is raw question/answer columns; feeding it in directly crashes on
`example["prompt"]`. So "switch to a standard dataset" = the standard dataset + the deterministic recipe here.

The two slime recipes (`search_r1_nq_hotpotqa` / `wiki18_e5`) are the same kind of job in two
other shapes: the first merges two QA sets into parquet using the Search-R1 template, the second
`cat`s an upstream FAISS index split in two back into one readable file. What they share is
"downloaded bytes != the bytes the task needs".

## Why this package lives under docker/, and why it is named autorlbench_datasets

It is **part of the image**, the same kind of thing as `autorlbench_fetch.py`: a task's
`environment/prepare.sh` calls it inside the container, and the container has no copy of this repo.
docker/ is the image build context, so it lives here
(`COPY autorlbench_datasets /usr/local/libexec/autorlbench_datasets`).

The prefix is not decoration: the recipes do `from datasets import load_dataset` (the HF library),
and a package named `datasets` would shadow that library, with the error surfacing at `load_dataset`,
far from the cause.

Why not one copy per task `environment/`: tasks 011 / 022 / 023 all need **the same
DAPO-Math-17k_verl**, and slime_search_r1 / slime_search_r1_nomask need **the same
nq_hotpotqa_train and the same wiki-18**. Every copy is a recipe that can drift on its own,
and these tasks' ground truth depends on them getting identical bytes. It is the same reason
tasks in one environment share one image.

## Two ways to run

Inside the container (prepare.sh takes this path, with caching; see docker/autorlbench_dataset.py):

    autorlbench_dataset --recipe gsm8k --dest "$AUTORLBENCH_WORKSPACE/gsm8k_verl"

By hand on the host (also prints sha256 + size):

    python3 -m autorlbench_datasets.gsm8k --out-dir /some/where/gsm8k_verl

Requirements: `datasets` + `pandas` + `pyarrow` + `huggingface_hub`, plus network access to HF.
The verl / slime environment images have all of them; the host's stock `/usr/bin/python3`
(3.10.12) has none. `wiki18_e5` also needs ~86GB of scratch disk; see that recipe's docstring.
dapo_math_17k / gsm8k have so far **only had syntax checks and offline unit tests of the split
rules, and have not been run end to end**; search_r1_nq_hotpotqa's output was checked offline for
row counts, columns, and the provenance of the first 3610 rows.
Watch the first run and check the printed row counts and schema.
"""
