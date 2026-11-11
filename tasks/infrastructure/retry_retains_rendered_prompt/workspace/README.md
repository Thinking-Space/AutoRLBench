# Retry lifecycle reproduction

This is a sanitized, standard-library snapshot of a generation retry path. It
contains no training data, model weights, credentials, network addresses, or
private prompt text. The visible system and user messages are static examples.

The runnable reproduction is:

```bash
python3 scripts/show_failure.py
```

It completes one generation, resets the same sample for retry, and demonstrates
the repeated input-contract failure without contacting a model service.
