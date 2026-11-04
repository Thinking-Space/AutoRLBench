# Measured runs

The artifacts behind the Results section of the task `readme.md`: one recorded
run on 1×H800 for each side of the defect — `defective/` (the published
`workspace/` as shipped) and `fixed/` (the same with the oracle patch applied) —
Stage A = 2 rollouts, Stage B = 30 rollouts of which the first 3 are
critic-only, evaluated every 5. One slice of this is an input to the task: the
defective side's Stage A/B logs and value-loss summaries are mirrored
(unpacked) into `workspace/logs/` as the frozen evidence the issue level
diagnoses; everything else — the fixed arm, the per-episode dumps, the
comparison — is oracle-side only, kept so the reported numbers can be checked
without a GPU and three hours. This directory lives under
`evaluations/solution/oracle/`, which the agent never sees.

```
validation_report.json   Endless Terminals admission report, 1489 of 2492 tasks.
splits.json              the deterministic train/val/test split of the admitted set.
                         Both are byte-identical between the two runs (one cached
                         prompt set, built once), hence stored once.
compare_et_test.json     compare_arms.py verdict, held-out test split
compare_et_val.json      the same for the validation split
compare_arms.py          paired held-out comparison of the two recorded runs
                         (moved here from the workspace, where it has no second
                         arm to compare against)
defective/               the published workspace as shipped
  stage_a.log.gz         full slime driver log, Stage A ("previous run")
  stage_b.log.gz         full slime driver log, Stage B (reload + warmup + PPO)
  stage_{a,b}_value_loss.json       tools/parse_value_loss.py output for each
  eval_episodes/eval_rollout_NNNN.jsonl.gz   one row per held-out episode
fixed/                   the same with the oracle patch applied
```

The two claims are reproducible straight from here (paths relative to this
`measured_runs/` directory). The defect measurement:

```bash
python3 -c 'import json;print(json.load(open("defective/stage_b_value_loss.json"))["first_value_loss"])'  # 7.200569
python3 -c 'import json;print(json.load(open("fixed/stage_b_value_loss.json"))["first_value_loss"])'       # 0.093750
```

and the end-to-end one, which reads the gzipped dumps directly:

```bash
python3 compare_arms.py \
    --buggy defective/eval_episodes --fixed fixed/eval_episodes \
    --dataset et_test
```

The solution verifier can be pointed at the archived fixed arm instead of a
fresh run — it reads `${AUTORLBENCH_RESULTS}/experiment_logs/stage_b_value_loss.json`,
not the logs, and runs its offline check with its own copy of the checker
(paths relative to the task root; CPU torch is enough for check 1):

```bash
rm -rf /tmp/cvri_ws /tmp/cvri_res && cp -r workspace /tmp/cvri_ws
evaluations/solution/oracle/apply.sh /tmp/cvri_ws        # oracle patch + offline check
mkdir -p /tmp/cvri_res/experiment_logs
cp evaluations/solution/oracle/measured_runs/fixed/stage_b_value_loss.json \
   /tmp/cvri_res/experiment_logs/
WORKSPACE=/tmp/cvri_ws AUTORLBENCH_RESULTS=/tmp/cvri_res \
  bash evaluations/solution/verifier/test.sh      # -> SOLUTION_VERIFIED
```

Two things about `fixed/` are worth knowing before reading its logs. Its Stage B
was interrupted twice and resumed, so `stage_b.log` is the fresh entry and
covers rollouts 0-4 only; the remaining rollouts are in the run directory's
`stage_b_resume_*.log`, which are not archived here because they add ~20 MB and
no metric the summary JSON does not already carry. Consequently its
`stage_b_value_loss.json` lists 10 critic steps where `buggy/` lists 60 — the
reported warmup window is the first 6 on both sides, which both files contain in
full, and that is the window the Results tables compare.

The `*_worker_logs/` directories that `run.sh` also produces are not kept: the
per-step `critic-step N: {...}` lines they hold are already present in the
driver logs, which is what the parser reads.
