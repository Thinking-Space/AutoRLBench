# Maintainer reference

The hidden starting verdict is in verifier/resources/expected_verdict.json.
The scored task requires an unhealthy assessment and a corrected query. The
reference resolver is provided here for maintainers only. Any submission,
repaired or not, still requires the same fresh continuation checks.

Original GPU receipts are under task provenance/receipts/. They predate the
bounded query interpreter and are not new security or agent evaluations.

Availability is part of the scored contract: records pin content by `sha256`,
and the verifier probes the submitted resolver with synthetic unavailable
records (see `assess_resolver_contract` in `verifier/tests/grade_submission.py`).
A repair that returns any path for content it cannot locate fails the
availability contract even when the continuation replay itself is clean.
