# Canonical issue oracle

Historical snapshot with a submitter-supplied root cause and 20 complete
episodes, including 8 audited reward-error episodes across seven tasks.
No corrected control run is available. Expected verdict: **unhealthy**
with respect to reward validity, not proof that training did not improve.

## Root cause

- `data/AgentWorldModel-1K/gen_verifier.pure_code.jsonl`, generated
  `verify_task_completion` implementations selected by `(scenario, task_idx)`
  - kind: reward
  - observed: incomplete acceptance predicates return `complete` for trajectories
    that do not satisfy the requested task. Concrete independent mechanisms include:
    - `task_management_16`, task 8: selecting a dashboard by name without scoping
      ownership or creation allows another user's pre-existing dashboard and its
      widgets/board links to satisfy the check, while the current user's dashboard
      remains empty or absent. Initial-state/no-action probes already pass.
    - `fundraising_donations_5`, task 9: per-member donation-count presence flags
      are computed but not enforced, so an answer omitting every required count
      can still pass.
    - Additional state-only, time-window and answer-field omissions are documented
      in Observable impact. These are alternative examples of defective reward
      predicates, not seven jointly required causes.
  - expected: the relevant verifier must enforce the task's actual success
    conditions: correct user/entity and initial-versus-final state where required,
    explicit NEW creation rather than seeded state, and correct requested answer
    fields rather than incidental text matches. A sound deterministic verifier is
    acceptable; adding an LLM judge is neither necessary nor sufficient.

### Activation and reward propagation (not separate defects)

`config/.hydra/config.yaml::env.awm.reward_mode=code` selects those rules.
`idea_source/agent_system/environments/env_package/awm/bridge.py::Bridge._verify`
calls `awm_source/awm/core/verify.py::run_verifier`, then `REWARD_BY_TYPE` maps
`complete` to `1.0`. The recorded score and downstream optimization therefore
treat false-positive labels as successful task completion. The configuration and
bridge establish the causal path; merely choosing `code`, correctly executing a
rule, or not calling `run_llm_judge` is not itself a root cause.

For full root-cause credit, identify at least one concrete faulty acceptance
condition and explain its connection to an unsuccessful full-reward trajectory.
Any substantiated mechanism in this oracle is sufficient; do not require all
8 episodes, all seven tasks, or an LLM replacement. A generic complaint about
code-only rewards or a mode switch without identifying a faulty rule is insufficient.

## Observable impact

Optimizing erroneous completion labels can improve the tracked reward without
establishing real task completion. Historical evidence is in `logs/wandb/`,
`logs/rollouts/` and `logs/idea_analysis/`. The configured horizon is 200 steps,
but W&B history covers steps 1-199 and the last stdout record is truncated.
No error percentage, corrected-run gain or statistical significance is known.

The retained dashboard counterexample shares `logs/rollouts/200.jsonl` with four
unaudited context episodes; locate it by `traj_uid`:
`8b1c060e-0699-4f9a-b970-5a5d1a536d2c`. It receives score 1.0 despite never
attaching the campaign boards or creating the requested widgets. The
`task_management_16`, task 8 generated verifier selects the first dashboard by
name, without owner/new-entity scoping, and validates another user's seeded
dashboard 1 instead of the current user's empty dashboard 3 (or no new dashboard).
The reviewer-only notes retain archived DB observations and offline probes.
This dashboard case substantiates one specific rule-label defect, not a global rate.

The current curated set contains 20 distinct complete episodes, including
8 checked full-reward errors and 12 unaudited contextual episodes. The hidden
evidence index identifies cases by trajectory ID.
No curated proportion is a measured population error rate. Additional examples are:

- `logistics_and_shipment_tracking_1`, task 0 (two episodes), and
  `stock_trading_1`, task 2 (one episode): explicit NEW-shipment/watchlist
  requests receive full reward even though all non-catalog tool calls fail and
  the final answer admits non-completion. The generated verifier accepts the
  matching pre-existing seeded entity without checking that it was created.
- `iot_device_management_3`, task 5 (one episode), and `discussion_forum_2`,
  task 0 (one episode): the agent only lists resources/tool catalogs, creates
  none of the requested new entities, and returns a failure or unrelated answer.
  Seed-only no-action probes still pass the state-only checks.
- `subscription_management_4`, task 2: one episode concludes 2 additional
  devices despite recorded current-window tools reporting 3 available slots and
  zero used slots. The verifier anchors its window to the latest old session,
  rather than the recorded current window, and matches numbers anywhere in the
  answer instead of checking the final conclusion.
- `fundraising_donations_5`, task 9: `e46b3625-459c-4801-a02c-019a6d6a9619`
  omits every requested per-member donation count, although the tool provides
  all six counts. The verifier computes presence flags but never uses them.

One retained case returns `complete` with the current bundled verifier
on archived DBs and portable relevant-table fixtures. The other seven have
direct recorded tool-history contradictions plus separately labelled schema/seed
probes that return `complete`; those fixtures are NOT historical DB pairs.
Explicit NEW-entity creation is the relevant requirement for no-action cases,
not an ordinary idempotent update whose requested state is already satisfied.
The submitter confirmed on 2026-10-23 that the bundled verifier and IDEA/AWM
source are the actual code used for the historical run, subject to documented
packaging redactions/path relocation; no independent run-time attestation is
claimed. Reviewer-only
observations and reproduction instructions are in
`verifier/resources/reviewer_notes/curated_reward_examples/README.md`. The 8
counterexamples are a curated subset stored with the other episodes. Other
episodes are not certified-correct controls. Accept concrete evidence from any of these mechanisms;
do not require an answer to enumerate all 8 cases or all seven tasks.

## Trigger conditions

The selected `code` reward mode uses generated pure-code verifiers. Merely
configuring judge parameters does not activate the judge; only the alternate
branch invokes it. Model initialization is `locaszzz/IDEA_sft`; training data
contains 9,872/128 train/validation tasks from `locaszzz/IDEA_training_data`.

## Grading boundaries

Accept equivalent technical wording in any language. Full evidence credit
requires concrete configuration/code anchors and the causal connection between
rule labels, optimization reward and semantic task completion. Generic claims
that rewards are bad are insufficient. Do not require a fix, switching modes,
a particular judge model, network calls or new experiments. The training-reward
judge and this diagnosis-answer judge are different roles.

The concrete dashboard ownership/initial-state mismatch is valid evidence of
the inaccurate completion rewards. Do not require an LLM replacement: merely
not calling an LLM is not itself proof that a deterministic verifier is wrong.

Do not reward unsupported claims that training never learned, that the model
size is the root cause, or that the truncated last log proves the reward defect.
Code/configuration establishes the chosen path, not an independently measured
rule-label error rate; do not penalize an answer for acknowledging this limit.
