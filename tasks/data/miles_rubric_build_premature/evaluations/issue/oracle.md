# Canonical issue oracle

Reconciled from session-documented evidence.
Expected verdict: **unhealthy**.

## Root cause

- `gen_rubric_tigan10k.py` — `--build` invocation timing
  - kind: `script`
  - observed: `python3 gen_rubric_tigan10k.py --build` was invoked while async
    generation was still running. `load_done()` read 8199 lines from the JSONL
    at that moment and built the parquet immediately. Generation later completed
    to 9716 lines, but the parquet was already frozen at 8199.
  - expected: re-invoke `python3 gen_rubric_tigan10k.py` without `--build` and
    confirm it prints "待生成 0" before calling `--build`.

## Observable impact

- `len(parquet) == 8199`; `wc -l rubric_complete_expected.jsonl == 9716`.
- Difference: 1517 rows (~15.6%) missing from training.
- No error or warning at build time; parquet is structurally valid.

## Trigger conditions

- `--build` called before generation is complete.
- `load_done()` reads a partial JSONL state without checking completion.

## Documentation reconciliation

- "The generation script has a bug" is wrong: generation completed correctly.
- "The training data is too small" is a downstream symptom, not the cause.
- The "待生成 0" gate is the canonical completeness check.
