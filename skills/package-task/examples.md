# User input example

Provide everything in this structure in one go when packaging. If any item is missing, stop. Replace numbers, commits, and hashes with the real values from that training run. Packaging runs from the root of an AutoRLBench checkout — fork and clone the repository first, and keep the raw material outside `tasks/`.

When packaging, read the code and launch-script diff between the buggy commit and the baseline commit, write the root cause as one sentence, and include the metric symptoms given here.

The scoring keys in this example (from `metric_key` down) are needed **only when the task gets a solution tier** — the packaging agent decides this, and an issue-only task neither asks for nor uses them. Everything above them is required for every task.

## Evaluation score only

```yaml
source_workspace: /home/gaoyuan/verl
output_dir: tasks/infrastructure/verl_moe_token_normalization
task_id: verl_moe_token_normalization
title: "MoE auxiliary gradients miss token normalization"
domain: infrastructure
model_name: Qwen/Qwen1.5-MoE-A2.7B-Chat
dataset_name: openai/gsm8k
framework: verl
algo: grpo

git_repo: /home/gaoyuan/verl
buggy_commit: aaaa1111
baseline_commit: bbbb2222
ignore_hunks: []

launch_script: scripts/train.sh
launch_args: []
launch_env:
  WANDB_MODE: offline
model_path: /workspace/models/Qwen1.5-MoE-A2.7B-Chat
train_path: /workspace/data/gsm8k/train.parquet
val_path: /workspace/data/gsm8k/validation64.parquet
failure_logs:
  - logs/train.log
include_wandb: true

image: verl-vllm0.8.5-torch2.6.0-flashattn2.7.4-cu124-py3.10
gpus: 2

assets:
  - kind: hf
    repo: Qwen/Qwen1.5-MoE-A2.7B-Chat
    type: model
    revision: ec052fda178e241c7c443468d2fa1db6618996be
    dest: models/Qwen1.5-MoE-A2.7B-Chat
    why_this_path: >-
      scripts/train.sh exports PYTHONPATH from $W and the resolved config
      reads the model from $W/models/Qwen1.5-MoE-A2.7B-Chat, which prepare
      fills.
  - kind: hf
    repo: openai/gsm8k
    type: dataset
    revision: 740312add88f781978c0658806c59bc2815b9866
    dest: data/raw/gsm8k
    why_this_path: >-
      scripts/prepare_gsm8k.py converts the raw parquet into
      data/gsm8k/{train,validation64}.parquet with a fixed 2026-seed split.

metric_key: val-core/openai/gsm8k/acc/mean@1
metric_source: wandb_history
log_sample: '{"_step": 150, "val-core/openai/gsm8k/acc/mean@1": 0.0}'
final_step: 150
historical_final_baseline: "0"
required_final_score: "0.4"

historical_eval_points:
  - { step: 100, eval_score: 0.0 }
  - { step: 150, eval_score: 0.0 }
```

A `failure_reason` written from this diff would look like: `the per-token loss callback returns (loss, metrics) instead of the native (loss sum, token count, metrics), so MCore never normalizes the MoE z-loss gradients; final val-core is 0.0`. The exact wording follows the filtered diff.

Each entry in `assets` becomes one `fetch_hf.py` call in `environment/prepare.sh`, with `$AUTORLBENCH_WORKSPACE/` prefixed to `dest`; `revision` becomes the pinned-sha comment next to the call. `why_this_path` becomes the comment in the script, because a wrong path only blows up midway through training.
## No scoring method given

The example above, up to `required_final_score`, is the default scoring: the score depends entirely on the eval-set key. Per `score_mode`, take that key's highest finite value (default `best`) or the value at `final_step` (`final`). If it reaches `required_final_score`, score is 1; otherwise 0.

Give these two only when you need to narrow it:

```yaml
score_mode: best              # or final; defaults to best
required_final_score_max: "1.0"
```

`score_mode: best` suits evals with high per-point noise (for example, 30-question AIME); `final` suits "run to the end and pass at the end". `required_final_score_max` rejects obviously fabricated values.

## Add when the user specifies a scoring method

Reference scores are not limited to training scores. For each one, give the key name, threshold, weight, where it appears, and one log sample, and state the pass rule.

```yaml
score_components:
  - role: validation
    key: val-core/openai/gsm8k/acc/mean@1
    required: "0.4"
    weight: "0.8"
    source: wandb_history
    log_sample: '{"_step": 150, "val-core/openai/gsm8k/acc/mean@1": 0.0}'
  - role: training_reward
    key: critic/rewards/mean
    required: "0.14"
    weight: "0.2"
    source: text_log
    log_sample: "step:150 - critic/rewards/mean:0.16"
pass_rule: "score = sum over components of weight times whether the component meets its threshold (1 if met, else 0); pass only when score is 1"
```

`role` determines the key names in `grader_spec.json`. Write them as a flat triple; **do not** generate `components` / `pass_rule` fields:

```json
{
  "metric_key": "val-core/openai/gsm8k/acc/mean@1",
  "required_final_score": "0.4",
  "validation_weight": "0.8",
  "training_reward_key": "critic/rewards/mean",
  "required_training_reward": "0.14",
  "training_weight": "0.2"
}
```

`role: learning_rate` maps to `learning_rate_key` / `required_learning_rate` / `learning_rate_weight`. The eval-set side is always `validation_weight`.

Implement `pass_rule` exactly as the user wrote it. If the user did not provide `score_components`, do not apply this weighting.

## Add when the run needs more GPUs

Just declare the count; the docker backend derives `--gpus <n> --shm-size=16g` from it and the harbor backend synthesizes the compose device reservation. Never list device ids.

```yaml
gpus: 8
```

## Add when an asset needs on-site conversion

When no upstream repo exists and the asset can only be converted from downloaded assets (for example HF → Megatron torch_dist), give the conversion command and the arguments it depends on:

```yaml
assets:
  - kind: derive
    from: font-info_qwen3-4b-sft
    dest: font-info_qwen3-4b-sft_torch_dist
    command: tools/convert_hf_to_torch_dist.py
    fixed_args:
      WORLD_SIZE: "1"
      --rotary-base: "5000000"
    skip_without_cuda: true
    guard_in_launch_script: 'if [ ! -d "$DIST_MODEL_DIR" ]'
```

With `skip_without_cuda: true`, `prepare.sh` must probe for CUDA first and, if unavailable, print one line and skip. The harbor backend runs prepare as a build-time layer with no GPU, so the `guard_in_launch_script` guard does the work in the agent stage. If the output is byte-for-byte reproducible, record its sha256 in the script and assert it after building; if it is not, do not pin it.
