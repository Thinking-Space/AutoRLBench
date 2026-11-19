# Canonical issue oracle

Expected verdict: **unhealthy**.

## Diagnosis target and grading scope

As defined in `workspace/diagnosis_scope.md`, this task asks the agent to explain the dominant systematic same-token log-probability shift between rollout and actor in the recorded continuation and to separate it from residual outliers. It does not require one root cause for every historical learning, scoring, or runtime anomaly. Additional valid findings supported by workspace evidence must not be penalized; grade them according to the specific evidence they explain.

Full root-cause credit requires tracing the cross-backend output-support inconsistency below and its normalization and optimization consequences. Full evidence credit is achievable from the public source files, configuration, and `experiment_logs/same_token_observations.json`. Author-only counterfactual results are provided here as validation context and are not quantities the agent must quote.

## Root cause

A single output-support inconsistency spans two essential code locations:

- `framework/verl/workers/rollout/vllm_rollout/utils.py::monkey_patch_compute_logits`, invoked by `vllm_async_server.py` with `len(self.model_config.tokenizer)`:
  - **Observed**: the rollout sets logits at token IDs >= 151,665 to `-inf` before sampling and probability computation.
  - **Expected**: the actor likelihood used to optimize those sampled actions must use the same valid-token support.
- `framework/verl/workers/engine/fsdp/transformer_impl.py::FSDPEngineWithLMHead.prepare_model_outputs`, together with `framework/verl/utils/torch_functional.py::logprobs_from_logits` and entropy helpers:
  - **Observed**: the actor normalizes log-probabilities and entropy over all 152,064 LM head rows, including 399 IDs unreachable by the tokenizer.
  - **Expected**: normalize over the same 151,665 valid IDs as the rollout across every active probability and entropy path while preserving padded parameter tensor shapes.

Naming both files without connecting the support mismatch is incomplete, and merely noting padded vocabulary rows merits only partial credit because padded rows are benign when both backends share the same support.

For a given prefix state $s$ and a tokenizer-valid token $y$ ($y < 151{,}665$), let $p_{\text{valid}}(y \mid s)$ be the softmax probability over the 151,665 valid IDs (the rollout distribution) and let $M(s)$ be the total probability mass that the actor's full 152,064-row softmax assigns to the 399 unreachable rows. Because the actor normalizes over both sets, $p_{\text{train}}(y \mid s) = (1 - M(s))\, p_{\text{valid}}(y \mid s)$. This produces a shared shift across all valid-token log-probabilities at state $s$ and couples policy gradients to the unreachable logits. Under negative advantages, updates can increase unreachable logits even though the rollout never samples them, introducing an optimization direction absent from the sampling policy.

## Public observable impact

The frozen historical log covers 275 updates (steps 726 through 1000). In those logs, `K3` denotes the rollout-versus-recomputed-old-policy estimator, gradient norms are recorded prior to clipping (clipping threshold `1.0` was active), and probability correlation is computed in probability space after exponentiation, which masks large relative log-probability errors.

Selected public rows 509, 187, and 401 each contain 16,384 generated response tokens and show mean absolute rollout-versus-actor log-probability differences of approximately `4.873`, `4.869`, and `4.768`. Row 0 is a 637-token short response for comparison. Accept an agent's own accurate per-token or aggregate calculations from the public logs rather than requiring these exact rounded values.

## Author validation (not required agent evidence)

In an offline counterfactual check using the restored `checkpoint-975` and the three 16,384-token trajectories, restricting only the normalization support reduced the mean absolute rollout-versus-actor log-probability difference by `98.4%–99.1%` within each trajectory. This figure applies to those selected long trajectories rather than a token-weighted run-wide average and isolates output support on frozen trajectories; later corrected training and kernel diagnostics have separate evidence.

A separate seven-position replay of row 509 (`experiment_logs/shared_raw_positions.json` and `shared_same_token_observations.json`) provides raw target and top logits, full and block `logsumexp`, and every logit in the final block at fixed response positions (`0`, `511`, and `2047` through `16383`), enabling checkpoint-free reconstruction. Positions `0` and `511` have negligible tail mass, whereas at positions `2047` through `16383` the reconstructed unreachable-token mass $M(s)$ is `0.9955–0.9974`, and the shift predicted from $M(s)$ matches the recorded rollout-minus-actor log-probability difference within `0.0101` across all seven positions. Accept independent valid reconstructions without requiring these exact figures.

## Residual long-tail mismatch

In a separate 16,384-token same-token comparison after aligning output support, a heavy-tailed numerical residual remains: median absolute log-probability difference `0.008171`, p90 `0.040448`, p95 `0.068760`, p99 `0.253907`, and p99.9 `14.3299`, with 82 positions exceeding `1` and 28 exceeding `10`. At position `10760`, the actor log-probability is `-35.093754` and the rollout log-probability is `-9.369534` (difference `25.7242`) while the unreachable-token mass is zero. Support alignment explains the dominant systematic shift, leaving these sparse outliers as a separate numerical discrepancy. Credit accurate descriptions of this residual without requiring unprovided kernel-level attributions.

## Trigger conditions

- The LM head output dimension exceeds the number of tokenizer-reachable token IDs (here `152,064` vs. `151,665`).
- Rollout sampling masks the extra IDs while actor likelihood evaluation normalizes over all rows.
- Policy updates can produce negative advantages; this configuration has neither a reference KL loss nor a reward KL penalty.
- Active path flags in this run: `remove-padding` enabled, `fused kernels` disabled.

## Wrong attributions

Blaming learning rate alone, missing importance sampling correction alone, faulty GRPO advantage standardization alone, or an abrupt dataset change misses the confirmed support defect. Backend numerical discrepancies explain isolated token outliers and are valid secondary observations when supported, but they do not account for the systematic shared normalization shift. In the counterfactual comparison after support alignment, the three long trajectories still retained maximum absolute outliers of approximately `25.74`, `33.40`, and `27.53`, while short row 0 showed virtually no support-dependent shift. Reward diagnoses that distinguish the systematic support shift from residual numerical outliers.

## Reference repair scope

Align the active padded and `remove-padding` likelihood and entropy paths (and the reference-policy path if later enabled) to normalize only over tokenizer-valid IDs; if fused loss kernels are enabled later, enforce the same support mask inside the kernel. Keep the padded checkpoint and optimizer tensor shapes intact rather than truncating LM head rows. This issue grades diagnosis of the frozen historical failure; subsequent repaired training runs have separate source and evaluation artifacts.

## Other defects and historical source identity

The packaged `MathVerify` module matches the original immutable runtime (SHA-256 `1242011cddc041ea6cb19fc9a1872ebab742d8ae7e8ebd3342f6b3b8b6e2ba15`) and contains a real broken-process-pool recovery defect: if a worker dies, subsequent calls can reuse the dead pool and convert service failures into zero scores. Recognizing this conditional bug is valid, but the preserved main log for this run records no `MathVerify` failure, and scorer behavior does not change the actor's softmax denominator.

Separate diagnostics also investigated RoPE and RMSNorm numerics. Do not require the agent to cite unprovided kernel fixes, and do not penalize a supported residual-numerics finding. Credit the required support-mismatch diagnosis and valid independent findings separately.

## Shared-image training qualification

A separate two-step qualification run on the shared image (8 prompts, 64 responses, 16,384-token limit, native V1 sync) started a fresh optimizer from private `checkpoint-975` weights. Step 1 ran with zero warmup learning rate (leaving weights unchanged) and saved `CP1`. A native resume from `CP1` completed step 2 at learning rate `0.0000001` with gradient norm `27.8426`, `43` positive and `21` negative rewards, `5` mixed groups out of `8`, Adam step `2` on both ranks, and verified updates across all four checked parameter slices. Scalar logs and selected rollout samples for both steps are in `workspace/experiment_logs/shared_image/`. This qualification is distinct from the 64-prompt, 512-response historical continuation.
