# Solution oracle

- `repair.patch` — the reference fix: the public 4xx fail-fast rule for
  `miles/utils/http_utils.py::_post` (the two lines added by upstream commit
  108602db, byte-identical modulo context) and the gateway's routing-key
  forwarding (`gateway/whitebox_gateway.py`).
- `calibration.json` — the 2x2 recording matrix (one run per arm, single
  seed) the grading thresholds were set from: training-level evidence
  separates the 0-iteration arms from the 10-iteration arms; the fail-fast
  axis is graded by the verifier's sealed `_post` unit tests.
- `control_run/` — the fully-fixed control arm's logs (train, gateway,
  proxy, launch) for reviewer inspection; never mounted into the agent stage.
