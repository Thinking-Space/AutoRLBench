# Repair the evaluation protocol

Review the staged AIME25 evaluation and determine whether the single-run result
is reliable. If the evaluation protocol is defective, repair it in the staged
workspace at `/workspace`.

The repaired default protocol must:

- keep the 30-question AIME25 benchmark;
- keep stochastic generation at temperature `1.0` and top-p `1.0`;
- use `32` samples per question;
- aggregate repeated samples with the `mean` reducer.

Concretely, update the workspace evaluation configuration
(`config/eval_config.yaml`) that `run_eval.py` consumes so that
`run_eval.py --protocol-only` reports `samples_per_question: 32` and
`expected_generations: 960`.

Do not modify the preserved original evaluation evidence under
`workspace/autorlbench_case/original_eval/`. Do not modify any evaluation,
oracle, or verifier file. Do not use the network.

## Your job

This task is a protocol repair, not a training run — there is nothing to
launch. **Do not start any evaluation or training run**; cheap, non-training
checks are fine.

When you are done, submit a report at `/workspace/report.md`. It must
summarize:

1. **Root cause** — why the single-run protocol was unreliable.
2. **Changes** — every file you changed and what you changed in it.
3. **Expected effect** — how the repaired protocol addresses the
   unreliability.

A missing or empty `/workspace/report.md` scores 0.

## What the system will run

After you submit the report, the system re-runs the experiment in your
repaired workspace as a foreground process:

```bash
python3 run_eval.py --protocol-only > "$AUTORLBENCH_RESULTS/solution_protocol.json"
```

from `/workspace`. This is a CPU-only stdlib call (a few seconds). A
task-specific verifier then reads the JSON it produced and checks the
protocol fields deterministically; it also verifies that the preserved
evidence under `autorlbench_case/original_eval/` is unchanged.
