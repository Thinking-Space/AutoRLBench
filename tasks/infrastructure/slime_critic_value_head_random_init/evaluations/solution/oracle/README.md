# Sealed solution oracle

## Verdict

**unhealthy.** The workspace reproduces a Multi-Turn PPO run in `slime` v0.3.2
(Qwen3-1.7B on Endless Terminals) on an otherwise-faithful setup, but the
critic's value head is initialised with random weights instead of zeros.

## Root cause

`LinearForLastLayer.__init__` — `slime/backends/megatron_utils/model_provider.py`
(bundled in the workspace at `slime/slime/backends/megatron_utils/model_provider.py`).
The constructor
draws the weight from `normal_(mean=0.0, std=init_method_std)` (`init_method_std`
is the backbone's 0.02 LM-head scale). `LinearForLastLayer` is installed
**unconditionally** as the critic's output layer with `output_size=1`, after the
backbone is built, so this constructor is the only thing that sets the value
head's initial weights. A `[1, hidden_size]` value head drawn from `N(0, 0.02²)`
makes `V(s)` a random projection of the backbone's final hidden state, whose
magnitude is set by the activation norm — nothing to do with the task's rewards.

## Required remediation

Zero-initialise the value head: replace the `normal_` draw with
`self.weight.data.zero_()` (the `if bias: self.bias.data.zero_()` line already
zeroes the bias). See `value_head_zero_init.patch` / `apply.sh`. One hunk.
`LinearForLastLayer` is only ever the critic's `output_size=1` head, so zeroing
is minimal and unconditionally correct; the actor's `[vocab, hidden]` LM head is
untouched and still loads from the checkpoint. Do not weaken the check by
suppressing the metric, shortening the warmup to zero steps, or changing the
reward scale, the advantage estimator, or `--value-clip`.

A freshly built critic then predicts exactly 0, and its first reported value
loss is `mean(R²) = pass_rate ≤ 1`, bounded by the binary episode reward.

## Decider and threshold

Metric: Stage B's first critic-optimizer-step `train/critic-value_loss`
(`tools/parse_value_loss.py` → `first_value_loss`). Threshold **1.0**: broken
`> 1.0`, fixed `< 1.0`.

| | Stage B first value loss | warmup-window mean | warmup-window max |
| --- | --- | --- | --- |
| defective | 7.200569 | 6.416253 | 7.645938 |
| fixed | 0.093750 | 0.082262 | 0.152233 |

The gap is ~77× with no overlap: the episode reward is binary so a correctly
zero-initialised critic reports `mean(R²) = pass_rate ≤ 1` on its first step,
while the random `[1, 2048]` head on Qwen3-1.7B (final-hidden RMS ≈ 2.55,
init scale 0.02) reports `E[V²] + pass_rate ≈ 5.3` by derivation and 5.3–8.2 in
practice. The threshold sits an order of magnitude from either side, and above
the fixed side's value at step 0 (`mean(R²) = 6/64 = 0.09375`), so merely
finishing a run without the fix cannot pass it.

**Calibration: 1 run per side** (1×H800; Stage A = 2 rollouts, Stage B = 30
rollouts of which the first 3 are critic-only ⇒ a 6-step warmup window). Margins
are reported above; with one run per side there is no variance estimate.

**Harness rerun protocol** (the one the verifier actually grades): the rerun
`run_spec.json` injects only `OUT_DIR`/`CKPT_ROOT`, so `run.sh` runs its
defaults — the short reproduction: Stage B = 2 rollouts, both critic-only ⇒ a
4-step warmup window, 2×H100. That protocol was measured on both sides
(2026-11-02, devspace): defective `first_value_loss` 6.972 / warmup mean 7.154 /
max 8.154; fixed 0.250 / 0.218 / 0.372. The threshold separates just as
cleanly there (~28× gap, no overlap); the fixed side's step-0 value is the
pass rate of its opening batch (`mean(R²) = 0.25`), still 4× below
the threshold. The long-protocol artifacts in `measured_runs/` are from the
calibration runs, not from the harness rerun.

## Why the reload only exposes it (not the cause)

The random weights are created at construction, not loaded: a saved actor
checkpoint has no `[1, hidden_size]` value head, so on reload the constructor
builds a fresh random head and `load_checkpoint` has nothing to restore into it.
A from-scratch run shows the same inflation. "Training did not improve", "the
checkpoint carries a stale value head", and "optimizer state" are therefore
wrong attributions — see `../../issue/oracle.md`.

## Machine grading

The solution verifier (`../verifier/test.sh`) runs two checks and exits non-zero
on any failure (the harness reads the exit code when no `result.json` is
written): (1) offline — `../verifier/tests/check_value_head.py` (the
verifier's own copy; the checker does not ship inside the workspace, where the
issue-level agent could read the diagnosis off it) asserts a freshly built
value head has an all-zero weight and bias, and that `output_size=1` is still
built for the critic; (2) end-to-end — `run.sh` completes and Stage B reports
`first_value_loss < 1.0` and `warmup_value_loss_mean < 1.0` over ≥1 critic
optimizer step, on rollouts that really generated episodes (non-zero response
length). `AUTORLBENCH_SKIP_TRAIN=1` runs check 1 alone and exits 2 (`SOLUTION_PARTIAL`,
not a pass).

## Measured evidence

`measured_runs/` holds both recorded runs (defective + control), the parsed
value-loss summaries, the gzipped Stage A/B driver logs, the per-episode
held-out dumps, the deterministic split, the Endless Terminals admission report,
the paired-comparison verdicts, and `compare_arms.py`. See
`measured_runs/readme.md` for how to recheck the two claims without a GPU.
