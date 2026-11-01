# Incident continuation grading

The canonical `test.sh` calls `tests/grade_submission.py`. The wrapper uses only the hidden `resources/incident_audit` modules and `resources/snapshot.json`; it does not import the submitted resolver, a workspace verifier, or a producer summary. It checks the prepared scene receipt pin and invokes:

```python
verify_resume(prepared_scene, results / 'experiment_logs/run-home', fresh=True)
```

The frozen auditor is `verify_resume.py` SHA-256 `db21b6dd204f55efb4e4ce06aa65dfb0a7dd76e9ac35caa87fc31223cbfe4286`, accompanied by `accounting.py` and `scene.py`. These modules work without the repository tree or GPU libraries. The resource snapshot carries the seven measured scene-bound model file pins from the verified asset inventory.

Acceptance requires the exact planned logical IDs and content, native model/Adam/scheduler/RNG restore at checkpoint 3, scheduler progress 64, Adam counters 4→8, all four updates per role, final checkpoint readback and payload rehashing, fixed final evaluation, and unchanged sealed scene inputs. Native `None` defaults for the documented checkpoint skip flags are treated as disabled; true skip flags remain invalid. Transient retries are allowed; replacement samples, omitted updates, and missing output payloads fail. Neither reward improvement nor a fixed difficulty ratio is a threshold.

The wrapper writes `result.json` with `passed` (boolean), `score` (integer 0 or 1), and `metrics` (the original auditor proof). It also writes the proof to `metrics.json`. Success exits 0; rejection exits 1. Missing scene inputs produce a zero-score envelope. The harness owns the preceding nonempty `report.md` gate.

The task-level run specification invokes the prepared continuation driver and consumes only `workspace/loader.py`. It resumes rollouts 4–7 rather than replaying eight updates from base. Historical from-base calibration and its old score-1 receipt are archived under `oracle/history`; they are not acceptance receipts for this revision. Labeled synthetic fixtures validate the packaging interface only. Packaged run 1430440 passed this fresh verifier with score 1. Its unmodified result is archived at `environment/receipts/packaged-reference-grade.json` relative to the task root. No LLM blind test or agent success rate is reported.
