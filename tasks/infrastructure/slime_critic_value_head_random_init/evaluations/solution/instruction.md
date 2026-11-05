# Task: fix the inflated critic warmup loss

`/workspace` is a Multi-Turn PPO training workspace for `slime` v0.3.2
(Qwen3-1.7B on Endless Terminals). It has a defect in how the critic is built.

The reward is binary — 1.0 if an episode's pytest verifier passes, else 0.0 —
and GAE runs with `--gamma 1.0 --lambd 1.0`, so the critic's regression target
for every response token is the episode return in `{0, 1}`. A critic that has
not yet seen a single reward should therefore report a value loss bounded by
the reward scale: at most 1.0.

It does not. `./run.sh` trains a checkpoint (Stage A), reloads it, and runs
critic-only warmup steps (Stage B); Stage B's first `train/critic-value_loss`
comes out around **5-6**, and the whole warmup window is spent working it back
down. The frozen logs of the recorded run are in `logs/` if you want the
symptom first-hand.

## Your job

**Do not launch training — do not run `./run.sh` yourself, in the foreground
or background, and do not use `nohup`, `&`, `tmux`, `screen`, or any detach
mechanism.** The system runs the experiment itself after you finish (see
below); a run you start is wasted and scores nothing.

Make a freshly constructed critic predict exactly 0 everywhere, so that its
first reported value loss is `mean(R²) = pass_rate ≤ 1`.

The critic's value head is installed by `LinearForLastLayer` in
`slime/backends/megatron_utils/model_provider.py`. The full slime source tree is
bundled in the workspace: edit
`slime/slime/backends/megatron_utils/model_provider.py` directly. `run.sh` runs
this bundled slime (it is put ahead of the image's install on `PYTHONPATH`), so
your edit there is what takes effect.

Keep the change minimal and keep it correct for the actor: the same module
builds both roles, and the actor's `[vocab_size, hidden_size]` LM head must
keep loading from the checkpoint exactly as it does now.

When you are done, submit a report at `/workspace/report.md`. It must
summarize:

1. **Root cause** — why a freshly constructed critic reports a value loss of
   5-6 instead of ≤ 1.
2. **Changes** — every file you changed and what you changed in it.
3. **Expected effect** — the metric the fix should move and its expected
   value.

A missing or empty `/workspace/report.md` scores 0.

## How you will be checked

After you submit the report, the system deletes everything under
`$AUTORLBENCH_RESULTS/experiment_logs`, then runs the experiment end-to-end in
your repaired workspace (a full run needs 2 GPUs and takes roughly 15 minutes;
the Endless Terminals prompt set ships prebuilt with the workspace, so there
is no first-run build pass) and a task-specific verifier grades the fresh
results: the value-loss metric over the critic's warmup window, plus
structural checks that critic optimizer steps, generated rollouts and
non-zero response lengths are all present, so the metric cannot be improved
by suppressing it, shortening the warmup, or changing the reward scale, the
advantage estimator or `--value-clip`.

The verifier reads **only this fresh run's** files under
`$AUTORLBENCH_RESULTS/experiment_logs`; logs written anywhere else are ignored.
Do not copy, edit, or fabricate log metrics, and keep `run.sh` writing its
artifacts where it does now (`OUT_DIR`, which the system points at
`$AUTORLBENCH_RESULTS/experiment_logs`). Otherwise the run is scored 0.
Cheap, non-training checks are fine.

**Do not use the network.** Do not fetch code, documentation, packages,
or any other material from outside the staged workspace — diagnose and
repair using only what is staged locally.
