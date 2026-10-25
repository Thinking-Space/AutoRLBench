from __future__ import annotations

from typing import Callable, Dict, Type

from .base import SandboxRuntime


_RUNTIME_REGISTRY: Dict[str, Type[SandboxRuntime]] = {}


def register_runtime(name: str) -> Callable[[Type[SandboxRuntime]], Type[SandboxRuntime]]:
    """Register a runtime provider so the evaluator need not branch per platform."""
    def decorator(runtime_cls: Type[SandboxRuntime]) -> Type[SandboxRuntime]:
        _RUNTIME_REGISTRY[name] = runtime_cls
        return runtime_cls

    return decorator


def create_runtime(name: str) -> SandboxRuntime:
    """Create a runtime by name. Built-in providers register themselves on module import."""
    runtime_cls = _RUNTIME_REGISTRY.get(name)
    if runtime_cls is None:
        raise ValueError(f"unsupported runtime backend: {name}")
    return runtime_cls()


# Import built-in providers to trigger decorator registration.
from . import providers as _providers  # noqa: E402,F401
