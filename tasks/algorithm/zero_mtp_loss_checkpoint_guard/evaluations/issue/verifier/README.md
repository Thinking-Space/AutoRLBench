# Issue verifier

The shared LLM judge grades `$AUTORLBENCH_RESULTS/agent_answer.md` against the hidden `oracle.md`. The expected verdict is unhealthy for historical task historical failure.

A diagnosis must connect the Miles unlabeled logits interface (`labels=None`, `position_ids=None`) to the retained MTP1 postprocess path, `_get_embeddings`, packed-sequence rolling, and `None.clone`. It must distinguish the original failure, historical repaired replay corrected smoke, and any new tiny-model validation. It must not require a coupled checkpoint-guard failure or diagnose an invented optional-loss addition. There is no solution metric threshold or full-training comparison.
