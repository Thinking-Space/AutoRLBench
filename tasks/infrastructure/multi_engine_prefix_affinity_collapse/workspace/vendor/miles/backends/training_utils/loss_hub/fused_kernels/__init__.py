"""Kernel backend selection + dispatch state for fused logprob.

This package owns the fused-logprob backend state machine: which backend is active
(``--fused-logprob-kernel``), its lazily-cached tile impls, and the failure latch.
``fused_logprob.py`` calls :func:`forward_tile` / :func:`backward_tile` here and
does not know about backend selection.

Backends live as sibling modules (torch_eager / torch_compile / triton /
tilelang / cute_dsl). Each exports ``fwd_tile_impl`` / ``bwd_tile_impl`` matching
the eager signature in :mod:`torch_eager_fused_logprob`. All are loaded lazily
via :func:`get_backend_impls`; an import failure or runtime exception latches
``disabled=True`` and every subsequent call falls back to ``torch_eager`` (pure
PyTorch, zero optional deps) so one bad trace never kills the run.
"""

from __future__ import annotations

import importlib
from typing import Callable


# Backend registry: maps backend name to (module_path, fwd_symbol, bwd_symbol).
# Keys must match the ``--fused-logprob-kernel`` choices.
_BACKEND_MODULES: dict[str, tuple[str, str, str]] = {
    "torch_eager": (
        "miles.backends.training_utils.loss_hub.fused_kernels.torch_eager_fused_logprob",
        "fwd_tile_impl",
        "bwd_tile_impl",
    ),
    "torch_compile": (
        "miles.backends.training_utils.loss_hub.fused_kernels.torch_compile_fused_logprob",
        "fwd_tile_impl",
        "bwd_tile_impl",
    ),
    "triton": (
        "miles.backends.training_utils.loss_hub.fused_kernels.triton_fused_logprob",
        "fwd_tile_impl",
        "bwd_tile_impl",
    ),
    "tilelang": (
        "miles.backends.training_utils.loss_hub.fused_kernels.tilelang_fused_logprob",
        "tilelang_fwd_tile",
        "tilelang_bwd_tile",
    ),
    "cute_dsl": (
        "miles.backends.training_utils.loss_hub.fused_kernels.cute_fused_logprob",
        "cute_fused_logprob_forward_tile",
        "cute_fused_logprob_backward_tile",
    ),
}

VALID_BACKENDS: tuple[str, ...] = tuple(_BACKEND_MODULES)
DEFAULT_BACKEND = "torch_compile"
# Zero-dependency baseline used when a backend import fails or the latch trips.
FALLBACK_BACKEND = "torch_eager"


def get_backend_impls(backend: str) -> tuple[Callable, Callable] | None:
    """Load and return (fwd_tile_impl, bwd_tile_impl) for the given backend.

    Returns None if the backend is unknown or unavailable (import failure,
    missing deps, ...). The caller should fall back to ``torch_eager`` in that
    case (it always imports -- pure PyTorch, no optional deps).
    """
    entry = _BACKEND_MODULES.get(backend)
    if entry is None:
        return None
    module_path, fwd_sym, bwd_sym = entry
    try:
        module = importlib.import_module(module_path)
    except (ImportError, AttributeError):
        return None
    fwd = getattr(module, fwd_sym, None)
    bwd = getattr(module, bwd_sym, None)
    if fwd is None or bwd is None:
        return None
    return fwd, bwd


def list_available_backends() -> list[str]:
    """Return list of backend names that can be successfully imported."""
    return [b for b in _BACKEND_MODULES if get_backend_impls(b) is not None]


# ---------------------------------------------------------------------------
# Dispatch state. One module-level instance (``STATE``); ``fused_logprob.py``
# imports the re-export shims (``FUSED_KERNEL_STATE`` / ``set_fused_logprob_kernel``
# / ...) for back-compat with tests that poke ``disabled`` directly.
# ---------------------------------------------------------------------------
class _BackendState:
    """Active backend + cached impls + failure latch.

    ``disabled`` latches True after any dispatch exception; once tripped, every
    subsequent tile call runs ``torch_eager`` bare (no compile, no kernel) so a
    single bad trace never kills the run. ``set`` / ``reset`` clear it.
    """

    __slots__ = ("backend", "disabled", "_fwd_impl", "_bwd_impl")

    def __init__(self) -> None:
        self.backend: str = DEFAULT_BACKEND
        self.disabled: bool = False
        self._fwd_impl: Callable | None = None
        self._bwd_impl: Callable | None = None

    def reset(self, backend: str = DEFAULT_BACKEND) -> None:
        """Clear cached impls + re-enable the latch (tests use this)."""
        self.backend = backend
        self.disabled = False
        self._fwd_impl = None
        self._bwd_impl = None
        # Also drop the torch_compile backend's compiled-callable cache so a
        # re-selected backend never reuses a stale (possibly failed) compiled
        # fn. Local import: ``torch_compile_fused_logprob`` imports this module
        # (via ``fused_kernels.__init__``), so a top-level import would cycle.
        from . import torch_compile_fused_logprob
        torch_compile_fused_logprob.reset_compile_state()

    def set(self, backend: str) -> None:
        """Select the backend; eagerly resolve + cache its impls.

        Resolving at set-time (not per-call) keeps the selection stable and
        queryable. If the backend import fails, silently fall back to
        ``torch_eager`` (not torch_compile -- if the requested backend failed,
        torch_compile would just be eager with extra compile cost).
        """
        if backend not in _BACKEND_MODULES:
            raise ValueError(
                f"Invalid kernel backend: {backend!r}. Choose from {VALID_BACKENDS}."
            )
        self.reset(backend)
        impls = get_backend_impls(backend)
        if impls is None:
            self.backend = FALLBACK_BACKEND
        else:
            self._fwd_impl, self._bwd_impl = impls

    def select_fwd(self) -> Callable | None:
        """Return the cached fwd impl, lazily resolving from the registry."""
        if self._fwd_impl is not None:
            return self._fwd_impl
        impls = get_backend_impls(self.backend)
        impl = impls[0] if impls is not None else None
        self._fwd_impl = impl
        return impl

    def select_bwd(self) -> Callable | None:
        """Return the cached bwd impl, lazily resolving from the registry."""
        if self._bwd_impl is not None:
            return self._bwd_impl
        impls = get_backend_impls(self.backend)
        impl = impls[1] if impls is not None else None
        self._bwd_impl = impl
        return impl


FUSED_KERNEL_STATE = _BackendState()


def _fallback_tile(fwd: bool, *args, **kwargs):
    """Run torch_eager directly (no compile, no latch) as the disabled-state
    fallback, so a latched-off process never re-enters the failed path."""
    impls = get_backend_impls(FALLBACK_BACKEND)
    impl = impls[0] if fwd else impls[1]
    return impl(*args, **kwargs)


def fused_logprob_forward_tile(*args, **kwargs):
    """Dispatch one forward vocab tile to the active backend with latch fallback.

    The kernel backends (triton/tilelang/cute_dsl) do not implement the entropy
    accumulator (``weighted_num``); entropy is an observed metric, so when it is
    requested (``weighted_num`` kwarg present) we short-circuit to torch_eager
    (no kernel benefit to lose on a metric). Otherwise: latch off -> torch_eager;
    cache miss -> resolve + cache; on exception -> latch + torch_eager.
    """
    if kwargs.get("weighted_num") is not None:
        impls = get_backend_impls(FALLBACK_BACKEND)
        return impls[0](*args, **kwargs)
    if FUSED_KERNEL_STATE.disabled:
        return _fallback_tile(True, *args, **kwargs)
    impl = FUSED_KERNEL_STATE.select_fwd()
    if impl is None:
        return _fallback_tile(True, *args, **kwargs)
    try:
        return impl(*args, **kwargs)
    except Exception:
        FUSED_KERNEL_STATE.disabled = True
        return _fallback_tile(True, *args, **kwargs)


def fused_logprob_backward_tile(*args, **kwargs):
    """Dispatch one backward vocab tile. Mirrors :func:`forward_tile`; backward
    has no entropy path so there is no ``weighted_num`` short-circuit."""
    if FUSED_KERNEL_STATE.disabled:
        return _fallback_tile(False, *args, **kwargs)
    impl = FUSED_KERNEL_STATE.select_bwd()
    if impl is None:
        return _fallback_tile(False, *args, **kwargs)
    try:
        return impl(*args, **kwargs)
    except Exception:
        FUSED_KERNEL_STATE.disabled = True
        return _fallback_tile(False, *args, **kwargs)
