# Reviewer evidence

These files are hidden from the diagnosis stage. `bf16-*` and `fp16-*` preserve
the two formal runs' native metric rows and completion observations;
`comparison.json` is the derived paired summary. There is one run per precision.
Use the BF16 workspace evidence, rather than knowledge of this comparator, when
assessing an agent's diagnosis. The public task has no solution tier.

The scalar probability maximum (step 252) and logprob maximum (step 326) are
different observations. Full pre-update peak weights and response traces were
not retained. No exact replay claim follows from these aggregate records.
