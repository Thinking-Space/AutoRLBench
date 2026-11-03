# Reference diagnosis

The loss multiplies all token positions by response length and averages without masking, so padded positions affect the objective. It also differs from standard token-mean normalization and native dual clipping.

The forward remove-padding setting is True. Unpadding does not remove the mask requirement because log probabilities are repadded before policy loss. Locate case/loss.py and actor policy_loss.loss_mode. Explain loss scaling and clipping; do not claim zero gradients or that the reward callback broadcasts scores. For mask-only cases, explicitly distinguish a nonstandard objective from a universally erroneous implementation.

When attributing the composite difference from the standard reference, naming only one difference is partial rather than full attribution. The shared reference also uses remove_padding=true; this is an additional configuration difference for padded cases.
