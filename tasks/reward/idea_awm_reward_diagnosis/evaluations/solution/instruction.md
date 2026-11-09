# Diagnose and repair the staged experiment

Review the staged post-training experiment and determine whether it is healthy
or unhealthy. Treat the workspace as a snapshot you may repair: inspect its
source, effective configuration, launch scripts, data processing, reward logic,
logs, tracked metrics, and frozen evaluation evidence. Do not assume that an
unusual setting or implementation is causal without connecting it to observed
behavior.

Use held-out evaluation evidence together with rollout, truncation, runtime,
optimization, and training signals. Training reward alone does not establish
held-out health.

## Your job

If you find a defect, fix it in the staged workspace at `/workspace`. **Do not
launch training.** Do not start the experiment, in the foreground or
background, and do not use `nohup`, `&`, `tmux`, `screen`, or any detach
mechanism. The system evaluates your repaired workspace itself after you finish
(see below); anything you launch is wasted and scores nothing. Cheap,
non-training checks are fine.

**Do not use the network.** Do not fetch code, documentation, packages,
or any other material from outside the staged workspace — diagnose and
repair using only what is staged locally.

When you are done, submit a report at `/workspace/report.md`. It must
summarize:

1. **Verdict** — healthy or unhealthy, with the evidence that decides it.
2. **Root cause** — the defect you found, and the evidence in the workspace
   (logs, metrics, code) that connects it to the observed behavior.
3. **Changes** — every file you changed and what you changed in it.
4. **Expected effect** — which behavior the fix should change and in which
   direction.

A missing or empty `/workspace/report.md` scores 0.

## What the system will run

After you submit the report, the system runs a task-specific verifier against
your repaired workspace. There is no training rerun and no LLM judge — the
verifier itself is the whole evaluation. It selects, for each held-out
episode, the rule the training bridge would select (by scenario and task
index) from

```text
data/AgentWorldModel-1K/gen_verifier.pure_code.jsonl
```

in your repaired workspace, and replays the episode through that rule. The
score is the fraction of episodes classified correctly. Every original entry
in the rule file must remain present and parseable, and a file that was
wholesale-replaced, truncated, or corrupted makes the run ungradeable and
scores 0. Do not modify the preserved evidence under `logs/`, and do not
modify any evaluation, oracle, or verifier file. Otherwise the run is scored 0.
