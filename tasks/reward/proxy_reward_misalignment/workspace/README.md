# VIMPO: Value-Implicit Policy Optimization

[![arXiv](https://img.shields.io/badge/arXiv-2606.20008-B31B1B?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2606.20008)
[![Post](https://img.shields.io/badge/Post-000000?style=for-the-badge&logo=x&logoColor=white)](https://x.com/xuandongzhao/status/2068217433001959894)

This repository contains the VIMPO implementation used for the paper **"VIMPO: Value-Implicit Policy Optimization for LLMs."** It is a focused fork of `verl` for mathematical RLVR experiments. The repo keeps the core distributed `verl` trainer stack, the DAPO reward-manager baseline, and a `recipe/vimpo` entry point implementing the VIMPO terminal value objective with the PPO actor branch described in the paper.

## Research Direction

VIMPO is an early step in a direction that differs from mainstream PPO- and
GRPO-style RLVR. Instead of treating the external outcome reward as the only
training signal for policy improvement, VIMPO introduces an explicit bridge
between the model's internal token-level judgement and the external reward,
while keeping those two roles separate. This makes it possible to study whether
LLMs can develop better self-judgement over intermediate reasoning steps and
become less dependent on dense or carefully shaped rewards.

Because this direction is still new, the implementation intentionally keeps a
number of experimental switches beyond the exact paper recipe. These options
reflect hypotheses we considered while developing VIMPO, but did not have
sufficient validation budget to include as formal claims in the paper. We keep
them in the codebase so that researchers who find this path interesting can
continue exploring the behavior and improve the method.

## Method Summary

VIMPO derives a policy-implied value recurrence from the KL-regularized optimality condition for autoregressive generation. For each sampled response token, the implementation builds the token TD term

```text
rho_t - kappa_t

rho_t   = beta * log pi_theta(y_t | s_t) / pi_ref(y_t | s_t)
kappa_t = beta * KL(pi_theta(. | s_t) || pi_ref(. | s_t))
```

For outcome-only RLVR with `gamma=1`, the paper's operational value loss is

```text
L_V = 1/2 * [ sum_t (rho_t - sg[kappa_t]) - (R - mean_group(R)) ]^2
```

The stop-gradient on the KL term is enabled by default through `vimpo_config.vimpo_detach_kl=True`, matching the paper's `sg[KL]` value-loss term. The reference model is frozen by default (`vimpo_update_ref_freq=0`).

When `use_ppo_actor=True`, VIMPO also constructs a detached PPO actor advantage from the policy-implied TD term. The external reward enters through `L_V`, not directly through the actor advantage. In the final-reward setting above, the reward terms cancel from the actor TD residual, so the PPO actor branch does not need token-level rewards to compute its advantage.

## Data

The sample run expects parquet files with the same schema used by verl RL datasets, including a `prompt` column and reward-compatible metadata for math verification.

The helper scripts in `recipe/vimpo/data_preprocess` follow and are lightly
adapted from the preprocessing pipeline of
[`Shenzhi-Wang/Beyond-the-80-20-Rule-RLVR`](https://github.com/Shenzhi-Wang/Beyond-the-80-20-Rule-RLVR),
specifically `recipe/rlvr_with_high_entropy_tokens_only`. We acknowledge that
our math RLVR data preparation learns from that work.

Default paths are relative to the repo:

```text
data/math__combined_54.4k.parquet
data/math__aime_repeated_32x_960.parquet
data/math__math_500.parquet
```

Override them from the shell instead of editing code:

```bash
export DATA_HOME=/path/to/data
export TRAIN_FILE=/path/to/train.parquet
export VAL_FILES="['/path/to/aime.parquet','/path/to/math500.parquet']"
```

Dataset files are not committed to this repo.

## Example Run

Install the usual verl dependencies, then run:

```bash
bash recipe/vimpo/run_vimpo.sh
```

The default VIMPO run follows the paper recipe:

```text
model: Qwen/Qwen3-4B-Base
beta: 5e-4
actor coefficient c_A: 5e-3
KL estimator: exact KL(pi_theta || pi_ref)
KL reward penalty: disabled
actor KL loss: disabled
reference update: disabled
responses per prompt: 8
actor branch: PPO actor enabled
```

Useful overrides:

```bash
MODEL_PATH=Qwen/Qwen3-4B-Base \
DATA_HOME=/path/to/data \
N_GPUS_PER_NODE=8 \
TRAIN_PROMPT_BSZ=96 \
N_RESP_PER_PROMPT=8 \
bash recipe/vimpo/run_vimpo.sh
```

To run the GRPO baseline through the same script:

```bash
RUN_MODE=grpo bash recipe/vimpo/run_vimpo.sh
```

## Some Config Knobs

The VIMPO config lives under `vimpo_config`:

- `use_vimpo_loss`: enable the VIMPO objective.
- `value_type`: `raw` is the paper method; `soft` is retained only for experimental compatibility.
- `vimpo_beta`: coefficient `beta` in the policy-reference log-ratio and KL terms.
- `vimpo_kl_estimator`: `exact` matches the paper. `candidate_set`, `k2`, and `k3` are implementation variants. `k1` is rejected for raw VIMPO because it makes `log(pi/ref) - KL_sample` cancel to zero.
- `vimpo_detach_kl`: detach the KL term in the differentiable value loss. Defaults to `True`, matching the paper's stop-gradient KL term.
- `use_ppo_actor`: enable the detached PPO actor branch in the VIMPO objective.
- `vimpo_actor_coeff`: actor coefficient `c_A` in `L_VIMPO = L_V + c_A L_A`.
- `vimpo_adaptive_actor_coeff`: adaptively match PPO actor logit-gradient scale to the VIMPO value-gradient scale.
- `vimpo_ppo_whiten_adv`: whiten the detached actor advantage after GAE.
- `vimpo_update_ref_freq`: keep this at `0` for the frozen-reference setting in the paper.
- `vimpo_value_loss_type`: `squared_error` is the paper loss; `huber` is an implementation option.

The default run uses raw VIMPO, exact detached KL, PPO actor enabled, whitened actor advantage, a frozen reference model, and 8 responses per prompt.
