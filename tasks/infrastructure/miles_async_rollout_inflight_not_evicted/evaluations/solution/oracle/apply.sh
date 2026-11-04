#!/usr/bin/env bash
# Apply the one-hunk fix to a copy of workspace/, producing the control side the
# measured runs were generated from. Run from the task root.
set -euo pipefail
here="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
target="${1:-workspace}"
patch -p1 -d "$target" --fuzz=0 < "$here/abort_mode.patch"
echo "applied abort_mode.patch to $target"
