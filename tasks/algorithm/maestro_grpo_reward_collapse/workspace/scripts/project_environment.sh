#!/usr/bin/env bash
# Sourced from the project root by the training entry point.
export PYTHONNOUSERSITE=1
# Keep whatever PYTHONPATH the stage inherited (the benchmark's prepared/env
# hands the recorded numpy 1.26.0 shadow this way); prepending the project
# root and the vendored verl must not discard it.
export PYTHONPATH="$PWD:$PWD/verl${PYTHONPATH:+:$PYTHONPATH}"
if [[ -z "${PYTHON:-}" ]]; then
    if [[ ! -x runtime/conda-env/bin/python || ! -f runtime/conda-env/.maestro-prefix ]]; then
        echo 'Bundled environment is not ready. Run: bash scripts/restore_environment.sh' >&2
        return 1
    fi
    IFS= read -r _maestro_prefix < runtime/conda-env/.maestro-prefix
    if [[ "$_maestro_prefix" != "$PWD/runtime/conda-env" ]]; then
        echo 'Project location changed. Run: bash scripts/restore_environment.sh' >&2
        unset _maestro_prefix
        return 1
    fi
    unset _maestro_prefix
    export CONDA_PREFIX="$PWD/runtime/conda-env"
    export CONDA_DEFAULT_ENV=maestro-project
    export PATH="$CONDA_PREFIX/bin:$PATH"
    export PYTHON=runtime/conda-env/bin/python
fi
export VLLM_PYTHON="${VLLM_PYTHON:-$PYTHON}"
