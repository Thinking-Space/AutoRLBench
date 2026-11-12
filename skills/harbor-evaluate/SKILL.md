---
name: harbor-evaluate
description: >-
  Evaluate a model on AutoRLBench tasks (issue diagnosis level + solution repair level):
  environment and credential setup, launching and monitoring `autorlbench harbor-evaluate`,
  trial result validity checks, and score aggregation. Use when the user says "evaluate a
  model / run an issue or solution eval / run an eval round".
---

# harbor-evaluate — full model evaluation flow (issue + solution)

One evaluation = one trial of an agent+model on a task. Issue level: the agent diagnoses
read-only and writes `agent_answer.md`; an LLM judge scores it against a hidden oracle.
Solution level: the agent repairs the workspace and submits a `report.md` (it must not run
training itself); the harness re-runs the experiment in the verifier stage and the task's
grader scores only that fresh run. Everything is driven by a single command, and artifacts
land on disk automatically:

```text
python3 -m autorlbench harbor-evaluate <task-root> --evaluation-level <issue|solution> \
  --agent <agent> --model <model> [other args forwarded to harbor run]
        │
        ├─ prepare: the task's environment/prepare.sh, the only networked stage; fetches pinned models/datasets
        ├─ agent:   instruction.md + a writable copy of the workspace (inside the harbor container)
        ├─ rerun:   (solution only) inside the verifier container: clear log roots, re-run per run_spec.json
        └─ verifier: issue -> judge.py; solution -> the task's grader
Each trial is recorded under outputs/harbor-evaluate/<task_id>/<agent>-<model>/<timestamp>/
```

## 0. Principles (never violate)

1. **At issue level, only scores with `metrics.status == "judged"` count.** `"unavailable"`
   means the judge call failed (API timeout / bad key / network) — that is not a zero; re-run
   or re-judge locally (§2.4). Never let it enter the aggregate.
2. **A trial with non-empty `metrics.infrastructure_errors` is invalid** (harbor non-zero
   exit / trial exception / workspace collection failure); the harness forces its score to
   0.0 — that zero says nothing about model capability. Fix the cause and re-run.
3. **At solution level, a score of 0: check `metrics.failure` first.**
   `"agent did not submit report.md"` = the report gate, a real capability zero (the agent
   produced no deliverable); `"system rerun failed"` = the re-run was killed/crashed, also a
   capability failure; only a non-empty `infrastructure_errors` means an environment problem.
4. **Read each trial's `report.json` yourself before aggregating** (the check script in
   §2.3) — do not trust the process exit code: `harbor-evaluate` exits 1 both for "judged
   but did not pass" and for "something broke".
5. **Do not modify task packages**: evaluation reads `tasks/` only; scores are read only
   from `outputs/`.

## 1. Setup (once per environment)

```bash
git clone <AutoRLBench repo> && cd AutoRLBench

# 1) docker available; task images are pulled on demand at run time (the 8 shared images
#    are listed in ENVIRONMENTS.md — pre-pull with `docker pull <reference>` if you want)
docker images                                  # verify what is already local

# 2) harbor CLI available (AutoRLBench drives agents through it)
harbor --help || pip install harbor            # or --harbor-command 'uv run ... harbor'

# 3) credentials: .env at the repo root (gitignored), auto-loaded by harbor on startup
#    OPENAI_BASE_URL / OPENAI_API_KEY  -> the issue judge (OpenAI-compatible gateway)
#    <the chosen agent's keys>          -> e.g. ANTHROPIC_API_KEY
```

**Agent choice** (forwarded to `harbor run` as `--agent/--model`): claude-family and most
models use `--agent claude-code`; GPT-family uses `--agent codex`; Qwen uses
`--agent qwen-coder`; Grok uses `--agent grok-build`. The agent runs its own CLI inside the
container and takes credentials from the harbor process env / `.env`. Always request the
highest reasoning tier: `--agent-kwarg reasoning_effort=max`.

**Task scope**: 32 tasks in 6 domains under `tasks/`, 23 of which carry
`evaluations/solution/` (full list in reference.md, appendix A). `task_root` accepts a
single task directory, a domain directory (`tasks/reward`), or several paths at once;
`tasks/` itself does not work (its direct children are domains, not tasks).

**Start small**: before a full sweep, run one light task once (e.g.
`tasks/configuration/trl_gsm8k_eos_mismatch`) to confirm the agent works and keys are
valid, then scale up.

## 2. Issue-level evaluation

### 2.1 Launch

```bash
python3 -m autorlbench harbor-evaluate tasks/configuration/trl_gsm8k_eos_mismatch \
  --evaluation-level issue \
  --judge-model <judge-model> \
  --agent claude-code --model <model-id> \
  --agent-kwarg reasoning_effort=max
```

- `--evaluation-level` **must be passed explicitly** (the default is solution).
- `--judge-model`: the model used by the issue judge (OpenAI-compatible, via
  `OPENAI_BASE_URL/OPENAI_API_KEY`).
- Other options: `--huggingface-mirror` (HF mirror, passed to prepare),
  `--models-dir/--datasets-dir` (download caches, default `<repo>/models|datasets`, reused
  across tasks), `--harbor-temp-dir`, `--output-root`.

### 2.2 Monitoring

Run directory = `outputs/harbor-evaluate/<task_id>/<agent>-<model>/<timestamp>/`; once it
exists:

```bash
tail -f outputs/harbor-evaluate/<task_id>/<agent>-<model>/<ts>/prepare.log   # prepare stage (fetching models; can take a while)
tail -f outputs/harbor-evaluate/<task_id>/<agent>-<model>/<ts>/harbor.log     # agent/verifier stages
cat  outputs/harbor-evaluate/<task_id>/<agent>-<model>/<ts>/run.log           # stage lines + final [RESULT] passed=.. score=..
```

Global harbor job progress is in `<run-dir>/harbor-job/job.log`, the per-trial summary in
`<run-dir>/harbor-job/<trial>/trial.log`; the agent's live activity log (streaming JSON —
tail it to confirm it is actually working) is at
`<run-dir>/harbor-job/logs/agent/<agent>.txt`.

### 2.3 Result validation (mandatory before aggregating)

```bash
python3 - <<'EOF'
import json, pathlib
run = pathlib.Path("outputs/harbor-evaluate/<task_id>/<agent>-<model>/<ts>")
report = json.loads((run / "report.json").read_text())
for t in report["trials"]:
    m = t["metrics"]
    print("passed:", t["passed"], "| score:", t["score"])
    print("status:", m.get("status"), "| infra_errors:", m.get("infrastructure_errors"))
    print("verdict/root_cause/evidence:",
          m.get("verdict_score"), m.get("root_cause_score"), m.get("evidence_score"))
    print("judge reason:", m.get("reason"))
EOF
```

Validity condition: `status == "judged"` and `infrastructure_errors` absent/empty. Score
semantics: `score = 0.2×verdict + 0.6×root_cause + 0.2×evidence` (weights vary per task;
check the `WEIGHTS` header of that task's `evaluations/issue/verifier/judge.py`).
**Pass rule (matches the publication standard): `verdict==1.0` AND `root_cause>0.8` AND
`evidence>0.8`.** The most common fail shape is "right diagnosis, insufficient evidence".

### 2.4 Handling judge failures (status=unavailable)

The judge is just a script inside the verifier container, and all materials are local —
**re-judge only; do not re-run the agent**:

```bash
set -a; . ./.env; set +a
T=outputs/harbor-evaluate/<task_id>/<agent>-<model>/<ts>/harbor-job/<trial>
AUTORLBENCH_JUDGE_MODEL=<judge-model> OPENAI_BASE_URL=$OPENAI_BASE_URL OPENAI_API_KEY=$OPENAI_API_KEY \
  python3 tasks/<domain>/<task>/evaluations/issue/verifier/judge.py \
  --answer $T/artifacts/logs/artifacts/agent_answer.md \
  --oracle  tasks/<domain>/<task>/evaluations/issue/oracle.md \
  --prompt  tasks/<domain>/<task>/evaluations/issue/verifier/judge_prompt.md \
  --result  $T/verifier/autorlbench-result-retry.json
```

Once it reports `status: judged`: back up the original
`verifier/autorlbench-result.json`, copy the retry result over it, correct the run
directory's `report.json` accordingly, and note "re-judged locally" in the aggregate.

**Re-run quota**: for `missing or invalid agent_answer.md` (the agent wrote no
deliverable), re-run at most 3 times; before each re-run, check the agent log to confirm it
was actually working (tool_use events, token consumption) — repeatedly "finishes the
analysis cleanly but never writes the file" is a model capability defect: record 0 and stop.
Judge/network/container failures do not consume the quota; fix the cause and re-run.

## 3. Solution-level evaluation

### 3.1 Launch (differences from issue)

```bash
python3 -m autorlbench harbor-evaluate tasks/reward/slime_retool_length_reward_bonus \
  --evaluation-level solution \
  --agent claude-code --model <model-id> \
  --agent-kwarg reasoning_effort=max
```

- **No `--judge-model`**: the solution verifier is the task's own deterministic grader.
- The re-run happens inside the verifier container; timeout = the `run_spec.json` timeout
  + 30 minutes of grading headroom — no need to tune anything yourself.
- GPU count comes from the task's `task.toml` `environment.gpus`; the compose overlay adds
  device reservations automatically. If the machine lacks those GPUs, tasks with a re-run
  cannot run (check with `docker run --gpus all nvidia-smi` first).

### 3.2 Grading semantics (read results by these rules)

- **Report gate**: no non-empty `report.md` at the workspace root ->
  `{"failure": "agent did not submit report.md"}`, score 0, no re-run, no grading. This is
  a protocol zero, not an infrastructure failure.
- **System re-run**: the harness clears the task's declared log roots, re-runs the
  experiment on the agent's repaired workspace, and the grader reads only that fresh
  run's logs (`$AUTORLBENCH_RESULTS/experiment_logs`); `"system rerun failed"` (rerun
  exit≠0, e.g. OOM/crash) = score 0, a capability failure.
- **Protocol**: the final step's validation metric decides the pass
  (`missing finite <metric> at step N` = never reached the declared step count, also 0);
  tampering with pinned files or re-pointing the scoring threshold = 0.
- Validate `report.json` with the same script as §2.3, but look at `metrics.failure` /
  `metrics.errors` / `infrastructure_errors`; there are no judge sub-scores.

### 3.3 Post-run spot checks (recommended)

`diff -r` the task's original `workspace/` against the trial's `artifacts/workspace/` to
confirm the agent's change surface is reasonable (the files it touched match what
`report.md` claims); then scan the agent log for network forensics (`git clone` / `curl` /
`pip install` to external sources) — task protocols generally forbid networking, so fixes
that "pass" by pulling upstream sources are invalid; flag them in the aggregate.

## 4. Batch sweeps and aggregation

One model's round = 1 trial on each of the 32 issue tasks + each of the 23 solution tasks.
Ready-made scripts (copy to the repo root or run directly as
`bash skills/harbor-evaluate/examples/...`):

```bash
# issue level: all 31 tasks
bash skills/harbor-evaluate/examples/run-issue.sh claude-code <model-id> [judge-model]

# solution level: all 23 tasks that have solutions
bash skills/harbor-evaluate/examples/run-solution.sh claude-code <model-id>

# aggregate: scan outputs/harbor-evaluate, print per-task score/validity + totals
python3 skills/harbor-evaluate/examples/collect-results.py [<output-root>] --model <model-id>
```

Both sweep scripts call `harbor-evaluate` serially, one task at a time (prepare caches are
reused; a failing task does not block the rest), and every run's command is stored in its
own `command.sh` for standalone replay. The aggregation script keeps only rows that are
`status=="judged"` (issue) / free of infra errors (solution), and clearly marks invalid
runs for re-work.

## 5. Quick troubleshooting reference

| Symptom | Action |
| --- | --- |
| issue `status: "unavailable"` + judge request failure | re-judge only (§2.4) |
| `infrastructure_errors` non-empty | environment problem (harbor exit≠0 / trial exception / workspace collection failure); fix the cause and re-run |
| `missing or invalid agent_answer.md` | capability class; re-run within quota, record 0 after 3 attempts (§2.4) |
| solution `"agent did not submit report.md"` | report-gate true zero; record as capability failure |
| solution `"system rerun failed"` | re-run crashed (exit 137=OOM etc.); capability failure, record 0 |
| prepare stalls for a long time | normal — fetching models/datasets; `tail prepare.log`; if HF is slow, add `--huggingface-mirror` |
| want to see what the agent is doing | `<run-dir>/harbor-job/logs/agent/<agent>.txt` |
| a task's image is missing | nothing to do — the run pulls the task's digest-pinned reference on demand; to pre-pull manually, `docker pull <reference>` from the task's `task.toml` |

For depth (run directory layout, env contract, task list, anti-leak checks, full debug
table) see [reference.md](reference.md).
