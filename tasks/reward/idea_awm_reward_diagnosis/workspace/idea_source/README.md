# IDEA

IDEA is the submitter's project for agentic reinforcement learning with on-policy
self-distillation. This source tree contains the IDEA implementation used for
the recorded Agent World Model (AWM) OPSD experiment in this contribution.

The project name is IDEA. Source paths use `idea_source/`, the Python package
is `idea`, and training scripts live in `examples/idea_trainer/`.
The historical `algorithm.seed` configuration namespace, analyzer class names,
prompt-version values and metric keys are retained for runtime compatibility.
Recorded numerical evidence and upstream attribution remain unchanged.

## Recorded task entry point

Run the contribution's `workspace/run_local.sh` wrapper from the workspace root.
It uses `reproduce.py` and the recorded Hydra configuration under
`config/.hydra/`. Training is optional and requires separately provisioned
model weights, dependencies, and hardware; static issue diagnosis does not
require a training rerun.

The release includes only the AWM environment package; unrelated environment
implementations, site documentation and presentation images are omitted. The
AWM, IDEA and veRL execution path is retained. Data inputs and complete
server/trajectory logs are directly readable original files. The optional
preparation check verifies SHA-256 and byte sizes only; it never extracts or
rewrites files. No compressed release assets or network downloads are needed.

## Components

- `agent_system/`: agent environments and multi-turn rollout integration.
- `verl/`: veRL-based training implementation.
- `idea/`: trajectory analysis and distillation modules.
- `examples/idea_trainer/`: training launch scripts, including AWM OPSD.
- `../awm_source/`: the separately supplied Agent World Model runtime.

## License and upstream attribution

Retain `LICENSE`, `Notice.txt`, file-level copyright notices, and nested
third-party licenses. The MIT notice shipped with this source tree remains
unchanged; renaming the project does not change upstream authorship.
The original framework README and its upstream citations are retained in
`../../evaluations/issue/verifier/resources/author_materials/upstream_notices/framework-README.md`. Component-specific
terms are described in `../../THIRD_PARTY_NOTICES.md`.
