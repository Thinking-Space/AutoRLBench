#!/usr/bin/env bash
# Install AutoRLBench's prepare-stage fetch/dataset tooling into the image's
# /usr/local at runtime, from this directory's copies.
#
# These files used to be COPY'd into every docker/images/*.Dockerfile at build
# time. That baked a snapshot: a fix to autorlbench_fetch.py / autorlbench_dataset.py
# never reached a running job (prebuilt images are not rebuilt on
# demand), and even locally it needed a full image rebuild. Instead the
# Dockerfiles now ship only the activate hook, and each prepare run invokes this
# installer first so the shims exec the repo's CURRENT tooling, not a stale bake.
#
# Run it with this script sitting next to autorlbench_fetch.py / autorlbench_dataset.py
# / autorlbench_datasets (i.e. the repo's docker/ dir, mounted RO in the local
# prepare container). Overwrites
# unconditionally and is idempotent.
set -eu

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p /usr/local/libexec /usr/local/bin

# --- disable hf_xet --------------------------------------------------------
# huggingface_hub's HF_HUB_DISABLE_XET flag is unreliable (huggingface_hub#3266):
# in 0.34.x the download path keys off `find_spec("hf_xet")`, not the flag, so a
# xet-backed repo still routes through cas-*.xethub.hf.co -- which the cluster's
# whitelist egress proxy does not allow, so every HF fetch hangs/fails even with
# HF_HUB_DISABLE_XET=1 exported (see autorlbench_fetch.py). This hits both the model
# fetch (autorlbench_fetch_hf -> huggingface_hub) and the dataset recipes
# (autorlbench_dataset -> `datasets.load_dataset` -> huggingface_hub).
#
# The only reliable fix is to make `find_spec("hf_xet")` return None. `pip
# uninstall` is the clean way, but it cannot remove an hf_xet that was installed
# by conda/system packaging or that lives in a site-packages `python3 -m pip`
# does not own -- and on those images the earlier pip-only attempt silently left
# hf_xet in place. So: try pip, then if it is STILL importable, delete its files
# directly (these are throwaway prepare containers), then verify. The whole block
# is a no-op once hf_xet is gone, so it costs nothing on later prepares.
if python3 -c 'import importlib.util, sys; sys.exit(0 if importlib.util.find_spec("hf_xet") else 1)' 2>/dev/null; then
    python3 -m pip uninstall -y hf-xet >/dev/null 2>&1 || true
    # Fallback: nuke the files wherever the import system still finds them.
    python3 - <<'PY' || true
import importlib.util, shutil
from pathlib import Path

spec = importlib.util.find_spec("hf_xet")
if spec is not None:
    dirs = {str(p) for p in (spec.submodule_search_locations or [])}
    if spec.origin and spec.origin not in ("namespace", "built-in"):
        dirs.add(str(Path(spec.origin).parent))
    for d in dirs:
        p = Path(d)
        if p.name == "hf_xet" and p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
        # drop its *.dist-info / *.egg-info so pip/importlib stop advertising it
        for meta in list(p.parent.glob("hf_xet-*.dist-info")) + list(
            p.parent.glob("hf_xet-*.egg-info")
        ):
            shutil.rmtree(meta, ignore_errors=True)
PY
    if python3 -c 'import importlib.util, sys; sys.exit(0 if importlib.util.find_spec("hf_xet") else 1)' 2>/dev/null; then
        echo "[install-autorlbench-tooling] WARN: hf_xet still importable after pip+file removal; xet-backed HF fetches may fail" >&2
    else
        echo "[install-autorlbench-tooling] removed hf_xet (hf#3266: HF_HUB_DISABLE_XET unreliable)"
    fi
fi

# --- fetch -----------------------------------------------------------------
# The shims resolve python3 at call time, so they run under the image's own
# interpreter and its huggingface_hub / datasets (same as the old Dockerfile).
cp "$DIR/autorlbench_fetch.py" /usr/local/libexec/autorlbench_fetch.py
for kind in hf url; do
    printf '#!/bin/sh\nexec python3 /usr/local/libexec/autorlbench_fetch.py %s "$@"\n' \
        "${kind}" > "/usr/local/bin/autorlbench_fetch_${kind}"
    chmod 0755 "/usr/local/bin/autorlbench_fetch_${kind}"
done

# --- dataset ---------------------------------------------------------------
cp "$DIR/autorlbench_dataset.py" /usr/local/libexec/autorlbench_dataset.py
rm -rf /usr/local/libexec/autorlbench_datasets
cp -a "$DIR/autorlbench_datasets" /usr/local/libexec/autorlbench_datasets
printf '#!/bin/sh\nexec python3 /usr/local/libexec/autorlbench_dataset.py "$@"\n' \
    > /usr/local/bin/autorlbench_dataset
chmod 0755 /usr/local/bin/autorlbench_dataset

echo "[install-autorlbench-tooling] installed fetch/dataset tooling from $DIR"
