# Sealed implementation oracle

## Verdict

**unhealthy.** The staged Search-R1 recipe reproduces the canonical
retrieval-augmented GRPO setup on an unmodified upstream `slime` snapshot,
but two defects have been injected into
`examples/search-r1/generate_with_search.py`: an observation-mask defect in
`generate` and a title-only retrieval rendering defect in `_passages2string`.

## Root cause

Two independent causes; a complete attribution names both.

- `observation_loss_mask` — `generate` appends retrieved
  `<information>…</information>` tokens with
  `loss_mask += [1] * len(obs_tokens_ids)` and assigns that local list to
  `sample.loss_mask`. Search-engine results are therefore trained on. The
  paired `append_response_tokens(..., trainable=False)` call still marks those
  tokens non-trainable, but the local list is the one that wins.
- `title_only_retrieval_rendering` — `_passages2string` renders each
  retrieved document as `Doc {idx+1}(Title: {title})` and never appends the
  passage body `text` split off the document `contents` two lines above, so
  every `<information>` observation carries document titles only and the
  policy never sees retrieved passage text.

## Required remediation

Set observation tokens out of the loss:
`loss_mask += [0] * len(obs_tokens_ids)`, matching
`append_response_tokens(..., trainable=False)`. Only model-generated tokens
should carry `loss_mask=1`. And include the passage body in the rendered
observation: `Doc {idx+1}(Title: {title}) {text}`. Fixing only one of the two
leaves the run below the passing bar (measured single-fix states: `0.18533`
loss-mask-only-fixed, `0.1543`/`0.2237` rendering-only-fixed).

## Historical evidence

Evidence kind: `historical_run`. The frozen window is train step 200; the last
held-out eval is step 199 (`eval_interval=10`). Machine-readable copies:

- `oracle/historical_evidence_manifest.json`
- `oracle/dirty_reference/historical_metrics.json`
- Agent-visible `workspace/outputs/metrics.json`,
  `workspace/outputs/historical_eval_trace.log`, and the slimmed W&B
  `workspace/wandb/*/files/output.log`

Primary metric `eval/nq_test` at step 199 is `0.1673` (16.7%), up from
`0.033` (3.3%) at step 0, with a peak of `0.1683` (16.8%) at step 179. 21
eval points are frozen. A healthy masked run reaches about 40.8% (`0.4083`)
at the same horizon. Train response length at step 190 is about 14 tokens
with a 0% truncate ratio. Training `rollout/raw_reward` at step 199 is
0.1680 and does not establish held-out health.

## Why it is unhealthy

- `examples/search-r1/generate_with_search.py::generate`: retrieved
  observation tokens are appended with `loss_mask=1`. Dummy
  `rollout_log_probs` of `0.0` on those tokens mean the GRPO objective is
  optimized against meaningless reference log probs. This is the injected
  defect that keeps held-out NQ accuracy far below the healthy protocol.
- `examples/search-r1/generate_with_search.py::_passages2string`: the
  passage body is computed and then dropped, so the `<information>` block
  the policy sees — and that the loss-mask defect trains on — is titles
  only. This starves the search channel on the input side independently of
  the loss-side defect.

The surrounding recipe is otherwise correct and must not be reported as the
cause:

- `examples/search-r1/qa_em_format.py`: exact-match reward with a
  `<think>→<search>→<information>→…→<answer>` format state machine.
- `examples/search-r1/run_qwen2.5_3B.sh`: standard GRPO
  (`--advantage-estimator grpo`, `--use-kl-loss --kl-loss-coef 0.001
  --kl-loss-type low_var_kl`, `--eps-clip 0.2 --eps-clip-high 0.28`,
  `--entropy-coef 0.00`), Adam `--lr 1e-6` constant schedule, and the recipe is
  wired only through the documented `--custom-generate-function-path` /
  `--custom-rm-path` hooks.

## Distractors that must not be reported as the defect

- `max_turns=2`, `topk=3`, `format_score=0.2`, `search_concurrency=256`.
- Commented-out eval and W&B argument blocks.
- Local retrieval server address `http://127.0.0.1:8000/retrieve` and runtime
  asset paths under `/root/…`.
- Falling response length / truncation, or training `rollout/raw_reward`.

## Machine grading

Machine grading reads only the new experiment logs under
`$AUTORLBENCH_RESULTS/experiment_logs`. The required held-out metric is
`eval/nq_test`, exact-match accuracy in `[0, 1]` — the defect distorts the
policy, not the metric, so no channel correction is needed.

The instruction fixes `--num-rollout 100` / `--eval-interval 10`, giving eval
points 0, 9, …, 99. `score_mode` is `final`: the graded value is the
`eval/nq_test` at the run's final eval point (step 99), and it must land in
`[0.28, 1.0]`. Pass is score `1`; otherwise `0`. `0.28` is the same bar the
healthy twin `slime_search_r1` uses — same recipe, same metric, same horizon —
so the threshold is not a tell for which side of the pair a task is on. The
historical dual-defect run's step-99 value is `0.15333`.

Calibrated on measured runs at the graded horizon (step 99; the two older
defect rows cite best over steps `0..99`, devspace 1107990,
`/user/user2/harbor-runs/`):

| run | code | value at 99 |
| --- | --- | --- |
| `slime_search_r1__stage__env` 2026-10-10 | repaired (`loss_mask += [0]`) | 0.3733 |
| `slime_search_r1__stage__env` 2026-10-09 | repaired (best `0..99`) | 0.3570 |
| `calib2-nomask-bonly` 2026-11-04 | loss mask fixed, title-only rendering left | 0.18533 |
| `slime_search_r1_nomask__stage__env` 2026-10-09 | loss-mask defect only (best `0..99`) | 0.2237 |
| `calib2-nomask-ab` 2026-11-04 | both defects (= the frozen evidence) | 0.15333 |
| `slime_search_r1_nomask__stage__env` 2026-10-10 | loss-mask defect only (= old frozen evidence) | 0.1543 |

Thinnest margin is `0.056` (`0.28 − 0.2237`), on the failing side; the
closest passing run clears the bar by `0.077`; an untrained step 0 (about
`0.033`–`0.048` across runs) fails by a wide margin.

The frozen historical evidence runs to step 199, past the rerun horizon — that
run simply used 200 rollouts. Grading is calibrated only on the `0..99` window
every measured run reached. Do not move the graded horizon to 199: the failing
states are high-variance there and censored past 99 — only the fully
dual-defect run is observed through step 199 (`0.16733`), while the
partial-repair states are unmeasured beyond 99 and one loss-mask-only
defective run was still climbing (`0.2713` at step 119) when it was killed,
so their 200-step ceilings are unmeasured. A healthy masked protocol reaches
about `0.3733` by step 99 and about `0.4083` at step 199.
