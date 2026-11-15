"""``torch.compile`` backend for fused logprob forward/backward tile operations.

This is the ``torch_compile`` backend: the *same* pure-tensor tile math as the
:mod:`...fused_kernels.torch_eager_fused_logprob` backend, wrapped in
``torch.compile(dynamic=True)`` with an eager fallback latch.

Both torch backends share one tile body (imported from the eager module); the
only difference is whether it is ``torch.compile``-d. Keeping them as sibling
modules means the dispatcher in ``fused_logprob.py`` treats torch the same way
it treats triton / tilelang / cute_dsl -- a lazily-loaded backend exporting
``fwd_tile_impl`` / ``bwd_tile_impl`` matching the eager signature documented in
:mod:`...fused_kernels.__init__`.

The latch is the *single* module-level ``FUSED_KERNEL_STATE.disabled`` owned by
:mod:`...fused_kernels.__init__` -- the same flag the top-level dispatcher and
the triton/tilelang/cute_dsl backends trip on failure. A ``torch.compile`` trace
blowup sets that same flag, so once tripped every subsequent tile call (from any
backend) runs plain eager and the dispatcher short-circuits before even reaching
the compiled callable. This keeps ``FUSED_KERNEL_STATE`` the one source of truth
for "is the kernel path latched off": there is no second private ``disabled``
that could drift out of sync (e.g. ``backend == "torch_compile"`` looking healthy
while actually serving eager). The only thing this module caches privately is
the compiled callables themselves (``compiled_fwd`` / ``compiled_bwd``), cleared
by :func:`reset_compile_state` which :meth:`_BackendState.reset` invokes.
"""

from __future__ import annotations

import torch

from .torch_eager_fused_logprob import bwd_tile_impl as _eager_bwd_tile
from .torch_eager_fused_logprob import fwd_tile_impl as _eager_fwd_tile


# Compile callable cache only -- NOT a latch. The latch is the single
# ``FUSED_KERNEL_STATE.disabled`` flag owned by ``fused_kernels.__init__``; a
# torch.compile trace failure sets that flag (see :func:`_compiled`), so the
# dispatcher short-circuits to eager on the next tile and never re-enters the
# compiled path. ``compiled_fwd`` / ``compiled_bwd`` cache the compiled
# callables so the torch.compile cost is paid once per (fn, dynamic-shape key).
_COMPILE_STATE: dict[str, object | None] = {
    "compiled_fwd": None,
    "compiled_bwd": None,
}


def reset_compile_state() -> None:
    """Clear the compile cache (tests / re-select backend use this).

    Does NOT touch the latch -- that lives on ``FUSED_KERNEL_STATE`` and is
    cleared by :meth:`_BackendState.reset`, which also calls this.
    """
    _COMPILE_STATE.update({"compiled_fwd": None, "compiled_bwd": None})


def _fused_kernel_state():
    """Lazy access to the single latch; avoids the ``__init__`` <-> backend
    module circular import (``fused_kernels/__init__`` imports this module to
    resolve the ``torch_compile`` backend, so it is not fully populated when
    this module is first imported)."""
    from . import FUSED_KERNEL_STATE
    return FUSED_KERNEL_STATE


def _compiled(fn, key: str):
    """Wrap ``fn`` in ``torch.compile(dynamic=True)`` with eager fallback.

    Lazily compiles on first call and, on any exception, trips the single
    ``FUSED_KERNEL_STATE.disabled`` latch + falls back to eager, so one bad
    trace never kills the run. Subsequent calls see the latched flag (either
    here or, more cheaply, in the top-level dispatcher) and skip compile.
    """

    def wrapper(*args, **kwargs):
        if _fused_kernel_state().disabled:
            return fn(*args, **kwargs)
        compiled = _COMPILE_STATE[key]
        if compiled is None:
            compiled = _COMPILE_STATE[key] = torch.compile(fn, dynamic=True)
        try:
            return compiled(*args, **kwargs)
        except Exception:
            # Latch: the rest of the process runs eager. The first call already
            # raised, so the caller sees the failure once (logged by the caller's
            # own try/except if it cares); subsequent calls are silent eager.
            _fused_kernel_state().disabled = True
            return fn(*args, **kwargs)

    return wrapper


def fwd_tile_impl(*args, **kwargs):
    """torch_compile fwd tile: compile the eager impl, fall back to eager on failure."""
    if not _fused_kernel_state().disabled:
        return _compiled(_eager_fwd_tile, "compiled_fwd")(*args, **kwargs)
    return _eager_fwd_tile(*args, **kwargs)


def bwd_tile_impl(*args, **kwargs):
    """torch_compile bwd tile: compile the eager impl, fall back to eager on failure."""
    if not _fused_kernel_state().disabled:
        return _compiled(_eager_bwd_tile, "compiled_bwd")(*args, **kwargs)
    return _eager_bwd_tile(*args, **kwargs)
