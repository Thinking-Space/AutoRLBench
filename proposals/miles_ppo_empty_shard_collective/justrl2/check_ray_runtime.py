"""Build smoke: start a private Ray cluster, execute a task, then shut it down."""
import os
from pathlib import Path
import secrets
import tempfile

from opentelemetry.sdk.metrics import export, view  # noqa: F401

with tempfile.TemporaryDirectory(prefix='justrl2-ray-build-') as directory:
    root = Path(directory)
    token = root / 'auth'
    token.write_text(secrets.token_hex(32))
    token.chmod(0o600)
    os.environ.pop('RAY_AUTH_TOKEN', None)
    os.environ.update(RAY_AUTH_MODE='token', RAY_AUTH_TOKEN_PATH=str(token), RAY_USAGE_STATS_ENABLED='0')
    import ray
    ray.init(num_cpus=1, num_gpus=0, include_dashboard=False,
             object_store_memory=100 * 1024**2, _temp_dir=str(root / 'ray'))
    try:
        @ray.remote
        def probe():
            return 42
        assert ray.get(probe.remote(), timeout=60) == 42
        assert len([node for node in ray.nodes() if node.get('Alive')]) == 1
    finally:
        ray.shutdown()
print('Complete Ray startup, task execution and shutdown passed.')
