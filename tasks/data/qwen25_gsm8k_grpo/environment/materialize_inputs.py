"""Place the pinned source dataset at the launcher's portable input path."""
import os
from pathlib import Path

from fetch_hf import materialize

workspace = Path(os.environ['AUTORLBENCH_WORKSPACE'])
materialize(workspace / 'data/snapshot/gsm8k', workspace / 'data/gsm8k')
