# Reference repair

The vLLM logits mask makes IDs from 151,665 through 152,063 unreachable to the rollout policy. The actor's likelihood and entropy on those same trajectories still include the logits for those actions. `actor.patch` restricts the effective support in the padded and remove-padding branches of `FSDPEngineWithLMHead.prepare_model_outputs`, after any distributed vocabulary gather and before normalization. It leaves the original parameter, checkpoint and optimizer tensors intact. The guarded fused path is disabled in this configuration.

The patch isolates the actor support change. Sparse differences from numerical backend behavior remain and are outside this repair's acceptance metric. The fresh run must retain the training data and budget, checkpoint state, reward, and full terminal evaluation. It must meet the final threshold in `../verifier/resources/grader_spec.json` using the recorded 32 answers per problem across 100 held-out problems.
