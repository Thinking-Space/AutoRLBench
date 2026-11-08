# Canonical issue oracle

Expected verdict: **unhealthy** for the intended evidence-retrieval and reasoning
experiment, despite successful training execution and high training reward.

## Essential root cause

There is one essential mechanism: label-derived answer documents are admitted
to the training retrieval catalog. `workspace/catalog.py::build` reads the
training question and `reward_model.ground_truth.target`, then creates a
question-titled passage whose first sentence is the target and whose body
repeats the question and answer. `config.json` selects the resulting catalog;
`serve.py` embeds queries, searches its FAISS index, and returns those document
contents to the Search-R1 generation loop. Searching and copying the returned
answer therefore earns the ordinary generated-answer EM reward without the
intended evidence synthesis. The diagnosis must describe this causal role,
not merely name a suspicious file or say retrieval quality is poor. Exact
symbol names are not required when the same producer-to-consumer mechanism
is explained correctly.

## Required observable evidence

- The materialization code puts training targets into indexed document text;
  the declared training catalog and HTTP retrieval path actually consume it.
- A concrete frozen trajectory and corresponding retrieval record show a
  returned question/answer passage and the model emitting the answer from that
  passage. Relevant evidence is in `workspace/logs/recorded/rollouts/`,
  `retrieval.jsonl`, and the materialized catalog. The generated span, rather
  than merely the presence of an answer somewhere in a log, establishes the link.
- The experiment completes update 32 with high native training reward
  (about 0.95); clean held-out EM is about 0.36. These scalar values support the
  mismatch between apparent training success and the intended task, but the
  train/held-out difference alone is not the root-cause argument.

## Attribution boundaries

The native answer-only EM recipe and observation masking are shared with the
recorded reference, and reward is computed from generated text. This is a
retrieval-corpus provenance error, not a scorer reading the tool's gold-answer
column, a zero-query-embedding failure, held-out test questions inserted into
the training set, or evidence that training never improved. Ordinary legitimate
retrieval of factual answer-bearing source text alone is not a defect: the
relevant fact here is constructing retrieval passages from the training labels
for an experiment intended to learn evidence retrieval and reasoning.

No claim of complete query collapse, guaranteed top-1 retrieval, significant
held-out collapse, or absence of all internal reasoning is required. Private
paired or counterfactual measurements are maintenance calibration and are not
required in the agent's evidence. Do not require unstaged control results or
exact row/document IDs in the diagnosis.
