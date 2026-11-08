# Improve the staged experiment

Review the staged experiment and repair the data-processing defect you find.
Use the source, launch configuration, frozen logs and rollout evidence to explain
the cause. Cheap CPU checks are allowed. **Do not launch training or use the network.**

Submit a nonempty UTF-8 `/workspace/report.md` describing the root cause, the
files you changed, and the expected effect. A missing report scores 0.

## Repair scope

Repair the production GSM8K preprocessing script under `verl/examples/data_preprocess/`.
The system accepts equivalent implementations;it does not require a particular
patch. Preserve all source questions, row ordering, train/test membership,
the established question-plus-instruction prompt format, and correct reward labels.
Do not modify the launcher, framework, reward function, model, source dataset or
frozen evidence. Do not replace or fabricate grading output.

## What the system will run

The system clears this attempt's `experiment_logs`, then runs the following
foreground CPU-only command in a separate rerun sandbox:

```bash
python3 "$AUTORLBENCH_ENVIRONMENT/rerun_inputs.py"
```

The rerun deletes the staged, rebuildable `data/preprocessed/` directory and
completion marker, then calls the submitted production preprocessor through the
same data-generation entry used during prepare. It rebuilds all 7,473 training
and 1,319 test rows from the immutable staged GSM8K source. Existing edited parquet
files alone are not a repair:they are replaced.

Finally,the actual launcher's CPU `--input-config` mode resolves its data path.
The native dataset loader reads every regenerated training row and captures the
policy-visible messages and reward label. No Ray cluster,model inference,
optimizer update or RL training is started. A hidden verifier independently
checks the fresh parquet and consumption capture;candidate-reported metrics
are not scores. The fixed task contract must remain unchanged.

This solution tier grades data-pipeline repair only. There is no learning-curve,
accuracy,response-length or multi-seed requirement. Budget:0 GPU,900-second
CPU rerun timeout. Assets were downloaded by prepare and later stages are offline.
