# Canonical diagnosis

Expected verdict: unhealthy training pipeline.

The 64-token data.max_prompt_length with left truncation and filtering disabled loses required question context. SplitPromptDataset keeps validation at 2048; async uses dataset raw_prompt_ids. Repair the effective training cap to 2048. Do not claim no learning or held-out degradation: AIME does not underperform the reference.

Root cause must be tied to source/configuration and recorded evidence; curve shape alone is insufficient.
