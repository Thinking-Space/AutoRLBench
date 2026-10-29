# Seed and group evidence (reviewer reference, not used for grading)

## Runs

- Four 30-step runs of `run_grpo_fd.sh` on one GPU (one vLLM replica): defective and repaired, data seeds 1 and
  2, all with rollout seed 42. The arms differ only in `verl/workers/rollout/vllm_rollout/vllm_async_server.py`:
  the defective arm is verl `6f2aa24` as released, the repaired arm applies `reference.patch`. The workspace
  freezes defective data seed 1; `evaluations/solution/oracle/` holds the repaired runs and defective data seed 2.
- Four 3-step restart runs, same recipe without validation, with `scripts/sitecustomize.py` on `PYTHONPATH`. The
  probe wraps `vLLMHttpServer.generate` and records, per call, the request id, `priority`, the sampling seed that
  `generate` actually set, the temperature, and SHA-1 digests of the prompt and response token ids. Neither arm's
  code is changed. Data is in `restart/`: `repaired-A`, `repaired-B-restart` (A run again in a new process),
  `repaired-D-seed43` (A with `rollout.seed=43`) and `defective-C`. `defective-C` was probed with an earlier probe
  version that did not record `priority`.

## 1. The seed every request gets

`vLLMHttpServer.generate` at `6f2aa24` (line 600):

```python
if self.config.full_determinism:
    sampling_params.setdefault("seed", self.replica_rank + self.config.seed)
```

The defective restart run sent 1536 sampled requests (3 steps × 64 prompts × 8) and all 1536 carried seed 42; the 8
responses of each of the 192 prompts are identical (`restart_check.txt`, line C). Full determinism runs vLLM batch
invariant, so the 8 copies of a prompt see bitwise-identical logits, and a shared per-request seed then gives them
the same tokens.

## 2. Group diversity at every training step

`group_diversity.txt`, computed with `scripts/group_distinct.py` on the full rollout dumps (64 prompts × 8 samples
per step):

| run | groups with 8 identical responses | mean distinct responses per group | groups with any reward spread |
| --- | --- | --- | --- |
| defective, data seed 1 / 2 | 64 of 64 at all 30 steps | 1.00 | 0 |
| repaired, data seed 1 / 2 | 0 of 64 at all 30 steps | 7.95 to 8.00 | 0.44 to 0.91 of the groups |

With no reward spread in any group, GRPO's advantages are 0 for every sample. The defective console history shows
`critic/advantages/max` = `critic/advantages/min` = 0, `actor/pg_loss` = 0, `actor/grad_norm` = 0 and
`actor/kl_loss` = 0 at all 30 steps of both data seeds, and `val-core/openai/gsm8k/acc/mean@1` = 0.08719 at every
validation, equal to the base model's greedy accuracy.

## 3. The repaired seed keeps runs reproducible

`restart_check.txt`, regenerated from the shipped `restart/` data with `scripts/compare_restart.py`:

- Repaired, within one run: in each of the 192 prompt groups the 8 requests have 8 distinct `priority` values and
  8 distinct seeds, and the 8 responses are distinct (1536 distinct seeds in the run).
- Repaired, A against its restart B, matched by `(priority, prompt)` because request ids are fresh uuid4s: 1536 of
  1536 requests have the same seed and the same response tokens; the rollout dumps of steps 1, 2 and 3 are
  identical line for line (512 of 512 each), and the step metrics (score, advantages, `pg_loss`, `grad_norm`,
  `kl_loss`, response length) are equal.
- Repaired, A against D (`rollout.seed` 43 instead of 42): every seed is shifted by exactly 1 and no response is the
  same, so the base seed still controls the stream.

The 30-step runs reproduce the restart runs too: the rollout dumps of steps 1 to 3 of the repaired data-seed-1
30-step run are byte-identical to those of `repaired-A` and `repaired-B-restart`, and the defective data-seed-1
30-step run's are byte-identical to `defective-C`, although the 30-step runs validate at step 0 and the restart runs
do not. The solution rerun relies on this: it repeats the graded command for 3 steps and compares those dumps.

## 4. Why `priority` and not the request id

`AgentLoopManager.generate_sequences` assigns `priority = np.arange(len(batch))` to the expanded batch before it
is dispatched, so each of the `n` copies of a prompt has its own value, and a restart assigns the same ones. The
request id that reaches `generate` cannot be used: `LLMServerClient` sends a fresh `uuid4` per call (the probe
records show uuid hex ids), and the `det-{priority}` id of the single-turn agent loop is only used for sticky
routing. A first version of the repaired arm keyed the seed on the request id (runs not shipped); it produced diverse
groups, but a restart of it gave different rollout dumps from step 1 on, i.e. it broke the reproducibility that
full determinism exists for.
`reference.patch` hashes `priority` together with the prompt tokens, so the same row on a later step, which holds a
different prompt, does not reuse a stream.

Limit of the reference patch: at this commit `ToolAgentLoop` does not forward `priority` to the server (it arrives as
0), so in multi-turn tool rollouts the `n` copies of a prompt would again share a seed on their first turn. This
task's recipe uses the single-turn agent loop. A complete upstream fix would need a per-sample key on that path too.

## 5. Not the uid defect

`verl_grpo_uid_after_repeat` makes every group a singleton, so advantages are non-zero but unnormalized
(`critic/advantages/max` = 0.9999990463 on rewarded steps). Here each prompt's 8 rows share one uid and form a group
of 8; the group is correct and its responses are identical, so advantages are exactly 0.

## Upstream status

Checked on 2026-10-25: verl main still sets the seed to `self.replica_rank + self.config.seed` (latest commit
touching the file: `619c397d`, 2026-10-21). No verl issue or pull request found under `full_determinism` reports
identical samples within a group. verl's determinism tests (`tests/workers/rollout/rollout_vllm/
test_vllm_generation_determinism.py`, `tests/experimental/reward_loop/run_determinism_e2e_with_rm.py`) run with
`rollout.n=1` and compare two runs against each other, which this defect passes.

## Files

- `reference.patch`: the repaired arm's change to `vllm_async_server.py` (written for this task).
- `group_diversity.txt`, `restart_check.txt`: outputs described above.
- `restart/`: probe records of the four restart runs; console logs and rollout dumps of steps 1 to 3 for
  `repaired-A` and `repaired-B-restart`.
- `scripts/`: `group_distinct.py`, `compare_restart.py`, and the probe `sitecustomize.py`. From the task root:

  ```bash
  R=evaluations/issue/reference/restart
  python3 evaluations/issue/reference/scripts/compare_restart.py \
      $R/repaired-A $R/repaired-B-restart $R/repaired-D-seed43 $R/defective-C
  python3 evaluations/issue/reference/scripts/group_distinct.py workspace/outputs/rollouts
  ```
