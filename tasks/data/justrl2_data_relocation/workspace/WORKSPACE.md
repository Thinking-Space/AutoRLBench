# Interrupted training workspace

Training has recorded an abrupt change in reward and abort metrics. Inspect the
logs and mounted incident. Implement the needed data-loading correction in
`loader.py`, whose API is `resolve(root, record) -> relative_path`, and submit
`report.md`. The resolver may use the files in the incident dataset to locate
records. It must return the requested logical sample, not a substitute.

The harness owns model weights, checkpoint restoration, sampling, generation,
reward and optimization. Only `loader.py` is consumed from your submission;
editing logs or local launch scripts does not change the experiment. The harness
continues the preserved training state; do not launch training yourself.

`incident/` is materialized before you begin and is read-only. The checkpoint,
data files and current storage metadata belong to the same sealed snapshot.
`logs/recorded/` contains the earlier historical fault reproduction; it is labeled
as historical and is not a claim that this new checkpoint produced those logs.
