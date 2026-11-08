# Reference repairs

There are two independently accepted repairs for the bounded submission interface.

- Replace `workspace/catalog.py` with `producer/catalog.py`. The selected
  producer then retains the supplied source documents without constructing
  passages from the training targets. The verifier materializes its fresh
  output and builds the effective index before checking production retrieval.
- Alternatively, replace `workspace/config.json` with `selection/config.json`.
  The effective training path then selects the prepared source catalog. The
  unused producer may remain unchanged.

Apply one alternative at a time and submit a nonempty `report.md`. Neither
repair changes scoring, model weights, source passages or the training budget.
The oracle files are hidden maintenance references, never agent-visible copies.

The solution's pass condition is the source-preserving data contract and real
CPU retrieval functionality. It does not claim that every accepted repair
improves held-out EM after retraining. The historical control's held-out margin
is too small to provide such a threshold.
