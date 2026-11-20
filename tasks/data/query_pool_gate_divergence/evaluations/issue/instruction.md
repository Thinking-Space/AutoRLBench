# Launch-gate diagnosis

The workspace holds the data side of an agentic RL setup: the query pools a run trains on, the
builder that produces them, the registry of instances whose sandbox image is unusable, the
launcher whose gate decides whether a pool may be trained on, a stub consumer that reads an
accepted pool exactly as the trainer would, and the logs of one pass of that gate over a list of
launch cases.

Decide whether the gate's decisions mean what they claim to mean. If they do not, find the
concrete root cause or causes in the code and the data, and explain the causal chain from each of
them to what the logs show, with specific file, case and number evidence. Separate root causes
from symptoms and from unrelated noise, and say which launch cases each cause decides wrongly and
in which direction.

You may run the launcher and the pool builder locally; they are CPU-only and take seconds. Do not
use the network, and do not modify the archived logs.

Write one concise UTF-8 diagnosis to `$AUTORLBENCH_RESULTS/agent_answer.md`. Do not propose a fix.
