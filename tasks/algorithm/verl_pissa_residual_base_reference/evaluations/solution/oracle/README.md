# Canonical solution oracle

Repair `_compute_base_log_prob_from_actor` so a PiSSA residual-base model
computes its base reference under the frozen initial adapter. That reconstructs
`W_pretrained = W_residual + A_init B_init`; ordinary `disable_adapter()` is
valid only for standard LoRA checkpoints whose underlying weights are already
the pretrained base.

The matched runs change only this reference path. At step 4 the defective run
has consistency `0.045778327704709`; the repaired run has `1.0`. Grading
requires `>= 0.95` at exactly step 4. One run per side supports the observed
margin but not a variance claim. Step-zero health is unavailable.
