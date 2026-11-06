# Canonical solution oracle

## Root cause

The first-turn `gen_text = prefix_text + search_content` lacks the terminator assumed by `new_gen_tokens[:-1] + gen_ids[-1:]`. The slice removes the closing search tag's final `>` token, corrupting the action text.

## Observable impact

The recorded step-200 training mean score is -1.0; all trajectories in the original 128-row batch hit the response-length limit. The one-line control has score 0.6796875 and 128/128 final-answer trajectories. Full-batch counts are preserved in `rollout_full_batch_summary.json`; the current shipped rollout files contain three sampled rows per retained step. Diagnose the collapse by connecting the reward curve and malformed, length-exhausted trajectories to the search-token reconstruction code.

Comparison logs are losslessly compressed as `.gz`, with original/compressed byte sizes and SHA-256 hashes in `recorded_comparison/control_run/log_archive_manifest.json`. They remain hidden from normal prepare and the agent. To replay the historical grader, copy that run to a writable directory and restore the copy with `environment/restore_logs.py --run-dir <copied-control-run>`. The normal prepare script restores only the agent-visible recorded run. Fresh solution grading does not consume historical comparison logs.

## Trigger conditions

The first-turn search reconstruction branch re-encodes a complete search block and replaces its final token with the original EOS. See the issue oracle for the causal chain and tokenizer-level reproduction.

## Required remediation

Apply `fix.patch`: append `"<|im_end|>"` to the reconstructed first-turn action. Do not alter reward, training data, held-out data, model or the 200-step budget. The historical native training script is byte-identical between source commits; packaging portability edits only affect output routing and interpreter selection and are shared by both arms.

Machine grading reads only fresh experiment logs. `critic/score/mean` must be finite and at least 0.5 at step 200, with maximum 1.0 (`score_mode=final`). This training-collapse regression criterion has margins 1.5 to the defective final and 0.1796875 to the control final. Calibration is one run per arm. No step-0 training score or variance estimate exists.
