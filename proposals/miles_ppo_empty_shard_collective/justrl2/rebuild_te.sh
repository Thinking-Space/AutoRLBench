#!/usr/bin/env bash
# Rebuild the TE torch binding against the exact torch in the pinned base.
set -euo pipefail
export NVTE_PYTORCH_FORCE_BUILD=TRUE NVTE_NO_LOCAL_VERSION=1 MAX_JOBS=2
CPATH=$(python3 - <<'PY'
from pathlib import Path
import site
paths=[p for base in site.getsitepackages() for p in (Path(base)/'nvidia').glob('*/include')]
assert any((p/'cudnn.h').is_file() for p in paths)
assert any((p/'nccl.h').is_file() for p in paths)
print(':'.join(map(str,paths)))
PY
)
export CPATH
WORK=$(mktemp -d /tmp/justrl2-te-build-XXXXXXXX)
trap 'rm -rf "$WORK"' EXIT
python3 - "$WORK" <<'PY'
from pathlib import Path
import hashlib,sys,urllib.request,torch
assert torch.__version__=='2.13.0+cu129'
url='https://files.pythonhosted.org/packages/9d/f1/ad613157261ac71262f5f260acf07fe22ae5ee0bb0675622c4f19a3e8afb/transformer_engine_torch-2.17.0.tar.gz'
with urllib.request.urlopen(url,timeout=120) as response:raw=response.read()
assert hashlib.sha256(raw).hexdigest()=='52ca109cdbc987ca02f87735fa375da8a02e117acb87e25dcb7e27dca37aa87f'
(Path(sys.argv[1])/'transformer_engine_torch-2.17.0.tar.gz').write_bytes(raw)
PY
python3 -m pip wheel --no-index --no-build-isolation --no-deps --no-cache-dir \
  "$WORK/transformer_engine_torch-2.17.0.tar.gz" -w "$WORK/wheels"
python3 -m pip install --no-index --no-deps --force-reinstall "$WORK"/wheels/*.whl
# The replay selects cuDNN fused attention. These optional prebuilt packages
# target a different torch ABI and are imported eagerly even when disabled.
python3 -m pip uninstall -y flash-attn flash-attn-3
python3 - <<'PY'
from pathlib import Path
import importlib.metadata as metadata
import subprocess,torch
assert torch.__version__=='2.13.0+cu129'
assert metadata.version('transformer-engine-torch')=='2.17.0'
for name in ('flash-attn','flash-attn-3'):
    try:metadata.version(name)
    except metadata.PackageNotFoundError:pass
    else:raise AssertionError('Incompatible optional package remains: '+name)
root=Path(metadata.distribution('transformer-engine-torch').locate_file(''))
extensions=list((root/'transformer_engine/wheel_lib').glob('transformer_engine_torch*.so'))
if not extensions:extensions=list(root.glob('transformer_engine_torch*.so'))
assert len(extensions)==1,extensions
undefined=subprocess.check_output(['nm','-D','--undefined-only',str(extensions[0])],text=True)
assert '_ZN3c104impl3cow23materialize_cow_storageERNS_11StorageImplE' not in undefined
print('Pinned TE binding rebuilt for '+torch.__version__)
PY
