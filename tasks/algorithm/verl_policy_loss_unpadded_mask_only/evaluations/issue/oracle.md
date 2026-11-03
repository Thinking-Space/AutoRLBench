# Reference diagnosis

The loss masks valid tokens but sums them per response and averages over responses; it omits native dual clipping. These are objective-design differences, not intrinsically invalid algorithms. Its task status is a candidate relative to the explicitly chosen standard-GRPO contract.

The forward remove-padding setting is True. Unpadding does not remove the mask requirement because log probabilities are repadded before policy loss. Locate case/loss.py and actor policy_loss.loss_mode. Explain loss scaling and clipping; do not claim zero gradients or that the reward callback broadcasts scores. For mask-only cases, explicitly distinguish a nonstandard objective from a universally erroneous implementation.

When attributing the composite difference from the standard reference, naming only one difference is partial rather than full attribution. The shared reference also uses remove_padding=true; this is an additional configuration difference for padded cases.
