# MTP zero-loss training smoke reproduction

This is a sanitized, standard-library reproduction of an RL training interface.
It contains no training data, model weights, credentials, hostnames, or private
prompt text.

The runnable reproduction is:

```bash
python3 scripts/show_failure.py
```

It checks the one-rollout checkpoint schedule first, then exercises the zero
auxiliary-loss training path so the two failures remain visible in one small
smoke.
