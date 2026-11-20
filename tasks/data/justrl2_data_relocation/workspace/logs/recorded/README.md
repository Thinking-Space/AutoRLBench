# Historical observations

Structured metrics and events come from an earlier eight-update PPO run.
`stdout.log` is a curated diagnostic excerpt; unfiltered logs remain in the
maintainer's recorded-audit archive. Numerical observations are unchanged.

[curves.svg](curves.svg) plots these recorded metrics and held-out evaluation
values. It adds no new model measurements or current acceptance evidence.

These logs are historical context, not output from the current CPU checker and
not proof that the current implementation is unhealthy. Use them to guide
checks of current code and file state. The current task requires no model,
checkpoint restoration, new training, model evaluation or GPU.

The easy/hard strata and historical accuracy metadata are author-supplied;
they are not newly measured difficulty or current model results. The bundled
original samples retain their IDs, prompts, answers and hashes. Acceptance
uses fresh independent read checks rather than these logs or prewritten JSON.
