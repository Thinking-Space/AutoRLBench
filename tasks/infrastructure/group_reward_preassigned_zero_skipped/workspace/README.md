# Group reward adapter reproduction

This is a sanitized, standard-library snapshot of a group-reward interface.
It contains no training data, model weights, credentials, hostnames, or
private prompt text.

The runnable reproduction is:

```bash
python3 scripts/show_failure.py
```

It builds one valid eight-sample group, applies the group adapter, and prints
the reason returned by the downstream validity filter.
