# Data Preprocessing Helpers

This directory contains the dataset preparation helpers used by the VIMPO
recipe. They are copied and lightly adapted from the data preprocessing pipeline
in
[`Shenzhi-Wang/Beyond-the-80-20-Rule-RLVR`](https://github.com/Shenzhi-Wang/Beyond-the-80-20-Rule-RLVR),
under `recipe/rlvr_with_high_entropy_tokens_only`.

We acknowledge that our math RLVR preprocessing setup follows and learns from
that repository. The local adaptation changes the repository-root calculation to
match this directory layout. The helpers download the Guru RL math
train/evaluation parquet files, keep the columns expected by verl-style RL
datasets, and duplicate the AIME evaluation split from 8x to 32x.

The upstream project is distributed under the Apache-2.0 license, matching this
repository's license.

Typical usage from the repository root:

```bash
bash recipe/vimpo/data_preprocess/prepare_train_test_datasets.sh
```

The script writes to `data/` by default. Override paths with `DATA_DIR`,
`TRAIN_FILE`, `AIME_TEST_FILE`, and `MATH_500_TEST_FILE` when needed.
