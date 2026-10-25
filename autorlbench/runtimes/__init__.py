"""Evaluation execution platforms."""

from .base import (
    CommandSpec,
    EnvironmentArtifact,
    SandboxInstance,
    SandboxRequest,
    SandboxRuntime,
)
from .factory import create_runtime, register_runtime
from .providers import DockerRuntime, LocalRuntime

__all__ = [
    "CommandSpec",
    "EnvironmentArtifact",
    "SandboxInstance",
    "SandboxRequest",
    "SandboxRuntime",
    "create_runtime",
    "register_runtime",
    "DockerRuntime",
    "LocalRuntime",
]
