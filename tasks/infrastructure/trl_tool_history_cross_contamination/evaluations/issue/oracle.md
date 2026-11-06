# Canonical issue oracle

Expected verdict: **unhealthy**.

## Root cause

`vendor/trl/trl/trainer/grpo_trainer.py::GRPOTrainer._generate_single_turn` selects `self.num_generations` (four) for post-tool continuations as well as initial generation. `vendor/trl/trl/generation/vllm_generation.py::VLLMGeneration.generate` assumes that its input consists of repeated identical prompts, subsamples with `all_prompts[::num_generations]`, and requests `n=num_generations` outputs from each retained prompt. After independent tool calls the histories are no longer interchangeable. Returned outputs are assigned to trajectory rows whose own histories were never sent to the server.

Initial generation legitimately has four samples of a shared prompt. The correct continuation path requests one output for every sample's full history, preserving initial group size four. The root cause is this dispatch error, not a global setting that makes four initial samples inherently incorrect.

## Observable impact

The diagnostic evidence is the cross-assigned continuation histories and their signatures; quoting specific rows, spans, or values is sufficient, and any equivalent decomposition of the totals counts.

- `observed_run/request_traces.jsonl` records expected token IDs, client input IDs, real server-echoed prompt IDs, multiplicity, row order, mode, step, and trajectory IDs. Expanding each returned server prompt by `client_n` reconstructs the history a continuation was actually conditioned on: 338 of 844 post-tool continuations reconstruct to a history other than their own row's, while initial shared-prompt generation is intact (0/160 mismatches). Quoting the counts in any equivalent decomposition, or stating the intact-initial versus mismatched-continuation contrast by mechanism, is sufficient.
- `observed_run/metrics.jsonl` shows nonzero training repair rewards and a nonzero gradient on at least one update. Development reward is 0.25 (4/16 trajectories; one of four bugs) at steps 0, 1, 2, and 3. Flat accuracy alone is not evidence of this root cause.
- `observed_run/weight_sync.jsonl` records completed weight updates. Post-tool sampling log-probability discrepancies are consistent with evaluating a continuation under the receiving sample's different context; these discrepancies are symptoms, not a separate root cause.

## Trigger conditions

Server mode, four initial generations per prompt, multi-turn tools, and naturally distinct histories. No per-sample nonce, hidden task difference, or fabricated observation creates the divergence.

## Attribution boundaries

A complete diagnosis must identify the retained multiplicity and subsampling after tool use and tie that mechanism to raw correspondence evidence. Generic statements about lost context, weak coding ability, or incorrect rewards are insufficient. Workspace isolation is implemented; the tests themselves do not explain server/history mismatches. The workload is difficult, but difficulty cannot make the server receive another sample's token history. Token truncation, parse errors, and normal shared initial prompts must not be misidentified as the causal defect.

The paired control has zero continuation mismatches, and real-server replay isolates the code path. These hidden controls support attribution but are not required evidence from the agent. The continuation group-instance census (219 observed continuation group instances, 159 affected, and 337/615 rows within the 158 naturally divergent groups) and the tool-trace inventory (`observed_run/tool_traces.jsonl` records unique workspace identities, identical starting-source hashes for samples of one case, edits, fixed test results, and clean-checkout grades, connecting a receiving sample to what it actually inspected or changed) likewise corroborate attribution but are not required evidence items. Neither arm improves development success in this three-update pilot; do not require or credit claims of a demonstrated learning-performance gap. There is one framework root cause; naming the source project's individual repair defects instead does not diagnose the RL system.
