# Canonical solution oracle

The current row's extracted solution is appended to the policy-visible question
inside `verl/examples/data_preprocess/gsm8k.py`. Remove that dependency and
regenerate the parquet actually consumed by the launcher. Reward labels and the
native scorer remain correct and must not change.

Equivalent implementations are accepted if they produce all original questions
with the established instruction and correct labels,
and feed those fresh prompts into the native dataset path. The unused sampling
setup may be retained or removed. No requirement is imposed on model immunity to
newly supplied hints or on post-fix learning curves. This is a data-contract
solution tier,not a quantitative RL-quality threshold.

`fix.diff` is the minimal reference repair. It is not mounted into the agent or
rerun sandbox. Qualification results describe actual positive/negative replays;
historical paired training remains evidence for issue attribution only.
