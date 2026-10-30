# SPDX-License-Identifier: Apache-2.0
"""This function is deliberately unreachable when the native replay path is active."""
def unused_rollout(*args, **kwargs):
    raise RuntimeError("This case requires --load-debug-rollout-data; no online rollout is configured")
