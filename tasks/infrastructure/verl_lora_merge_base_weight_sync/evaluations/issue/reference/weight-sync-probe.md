# Weight-sync probe (reviewer evidence, not used for grading)

Two extra 6-step runs (`probe7014_defect`, `probe7014_control`), identical to the recorded recipe
except that `probe/sitecustomize.py` was put on `PYTHONPATH`; neither arm's code was changed. The
probe hooks two points:

- actor side, `ActorRolloutRefWorker.update_weights`: at the start of every sync, for three LoRA
  target layers (`model.layers.0.self_attn.q_proj`, `model.layers.14.mlp.down_proj`,
  `model.layers.27.self_attn.v_proj`) it records `W_base` and `dW = scaling * B @ A` from the live
  PEFT parameters (`scaling = alpha / rank = 2`), i.e. what should reach vLLM;
- consumer side, `vLLMColocateWorkerExtension._update_weights`: the tensors vLLM's `load_weights`
  actually receives for those layers at the same sync.

1. **The actor updates.** `|dW|_F` for `q_proj` grows 0.0112, 0.0158, 0.0196, 0.0233, 0.0268,
   0.0302 over syncs 1–6 in the defective run (0.0110 … 0.0316 in the repaired one).
2. **What the consumer receives.** The merge happens in bf16 and most per-element `dW` are below
   half a bf16 ulp of `W`, so a correct merge differs from `bf16(W_base)` only where rounding flips.
   The informative set is `S = {i : bf16(W_base + dW)_i != bf16(W_base)_i}` (20k–1.3M elements per
   layer and sync). Defective: at every sync from 1 to 6 and on all three layers the received tensor
   is bit-identical to `bf16(W_base)` over the whole tensor, and equals `bf16(W_base + dW)` on 0% of
   `S`. Repaired: the received tensor equals `bf16(W_base + dW)` on 98.98% of `S` (7,166,555
   elements over syncs 1–6), differs by exactly one bf16 ulp on 0.97% and by two or more on 0.05%,
   and equals `bf16(W_base)` on about 0.2%. Our reference rounds the fp32 sum once; the residual
   one-ulp differences are consistent with the merge accumulating `B @ A * scaling` in a different
   order and precision before rounding. Tolerance used: exact bf16 bit equality, no epsilon.
3. **Output distribution after the sync, before the update.** verl logs, on each step's rollout
   tokens with the same response mask and temperature 1.0, `rollout_corr/kl` = masked mean of
   `rollout_log_prob - old_log_prob`, where `old_log_prob` is the actor's log-prob computed right after
   the sync and before the optimizer step. Defective probe run: 0.00037, 0.0013, 0.0066, 0.013,
   0.029, 0.047 at steps 1–6 (81.5 at step 60 in the recorded run). Repaired: 0.00052, 0.00045,
   0.00079, 0.00083, 0.0010, 0.0012 — the numerical floor for this stack.
   `training/rollout_probs_diff_*` in the same logs are probability-space differences
   (`|exp(a) - exp(b)|`), not log-prob differences.

Files: `probe/sitecustomize.py` (the hooks), `probe/probe_report.py` and `probe/probe_ulp.py`
(analysis), `probe/*.txt` (their outputs on both arms).
