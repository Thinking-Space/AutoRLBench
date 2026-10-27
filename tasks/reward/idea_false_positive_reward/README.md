# Inaccurate rule-based task rewards in IDEA AWM OPSD

Training `locaszzz/IDEA_sft` with OPSD on `locaszzz/IDEA_training_data` optimizes generated completion rules that sometimes award full reward to unsuccessful trajectories.

| | |
| --- | --- |
| **Author** | Jinyang Wu |
| **Domain** | reward |
| **Tags** | `idea` `verl` `awm` `opsd` `issue-only` |
| **Model** | locaszzz/IDEA_sft |
| **Dataset** | locaszzz/IDEA_training_data; locaszzz/AgentWorldModel-1K; 9,872 train / 128 validation tasks |
| **Framework** | IDEA / veRL / Agent World Model, `run_local.sh --train` |
| **Environment** | `idea-issue` · CPU-only diagnosis; no GPU or training required |
| **Evaluations** | issue |
| **Recorded runs** | 1 historical run, 0 corrected/control runs; 200 configured steps, W&B history covers 1-199 |

## Difficulty

The recorded `config/.hydra/config.yaml` selects `env.awm.reward_mode: code`.
In `idea_source/agent_system/environments/env_package/awm/bridge.py`, that mode
selects generated pure-code completion verifiers and maps their labels into rewards.
Some rules accept a different user's pre-existing dashboard, incidental identifier
matches, or seeded entities without checking the requested creation. Merely omitting
an LLM judge is not itself a defect; the incorrect acceptance rules are the cause.

The run reached tracked `critic/score/mean` values of 0.02455 at step 1 and
0.37402 at step 199. The frozen evidence contains 20 complete episodes, including
8 audited full-reward counterexamples across 7 tasks: 1 has archived database
evidence and 7 have recorded tool-history evidence plus separately labelled
seed-only verifier probes. This curated subset does not measure the population
error rate; all 8 cases are from the training split. The final configured
step 200 is absent from W&B history, and the last stdout record is truncated.
Neither rising reward nor that truncated record establishes semantic success or
a separate training failure.

## Reference solution

- **issue**: identify inaccurate rule-based completion rewards, name the configuration/code path, and connect a concrete verifier defect to a rewarded but unsuccessful trajectory. Do not require a patch, an LLM replacement, or a new training run. Detailed reviewer evidence is in [verifier resources](evaluations/issue/verifier/resources/README.md).

## Verification

- **issue**: an LLM judge compares `agent_answer.md` with the hidden oracle on verdict / root cause / evidence, weighted 0.2 / 0.6 / 0.2. Passing requires verdict = 1.0, root cause > 0.8, and evidence > 0.8, matching the current benchmark judge.

Prepare downloads dataset inputs from `locaszzz/AgentWorldModel-1K` at commit
`6f8a98a043c631ba9c88dd1d1d1d1829a1f9dca5`, checking SHA-256 and byte sizes.
Frozen logs and trajectories ship with the task; diagnosis and verification are
offline after prepare. Offline checks replay the 8 cases, verify the 20-episode subset
and check answer isolation. Hosted semantic grading and a corrected/control run
have not been performed. The optional `idea-historical` training image is
**not yet built**. Commands and evidence boundaries are in
[verifier resources](evaluations/issue/verifier/resources/README.md).
Component terms are in [LICENSE](LICENSE) and
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
