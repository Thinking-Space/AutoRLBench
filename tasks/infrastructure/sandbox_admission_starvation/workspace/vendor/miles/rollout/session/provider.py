"""Generic extension contracts for session trajectory providers.

The default registry contains only the built-in linear trajectory. Extensions are
registered explicitly in the session-server process; importing an extension is
never a prerequisite for ordinary sessions.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Protocol, runtime_checkable

from miles.rollout.session.types import SessionRecord
from miles.utils.chat_template_utils.tito_tokenizer import TITOTokenizer


PROVIDER_API_VERSION = "1.0"
CAPABILITY_DISCOVERY = "capability_discovery"
CAPABILITY_ABORT_REQUEST = "abort_request"
CAPABILITY_RESUMABLE_ROLLOUT = "resumable_rollout"


@dataclass(frozen=True)
class SessionProfile:
    """Selection and opaque metadata for one session trajectory."""

    name: str = "linear"
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ProviderCapabilities:
    """Generic capabilities advertised by a trajectory provider."""

    api_version: str = PROVIDER_API_VERSION
    names: frozenset[str] = frozenset({CAPABILITY_DISCOVERY})


@dataclass(frozen=True)
class ProviderCloseResult:
    """Provider-owned result returned when a session is closed."""

    status: str = "closed"
    metadata: Mapping[str, Any] = field(default_factory=dict)


@runtime_checkable
class TrajectoryProvider(Protocol):
    """Trajectory state contract consumed by :class:`SessionCore`.

    Implementations own trajectory data while the core owns HTTP, the
    prepare/proxy/commit lock boundary, response CAS, and error translation.
    The mutable lifecycle fields are intentionally part of the public contract:
    they let providers participate without inheriting ``LinearTrajectory`` or
    reaching into core internals.
    """

    lock: Any
    closing: bool
    messages: list[dict[str, Any]]
    records: list[SessionRecord]
    num_assistant: int

    @classmethod
    def create(cls, *, session_id: str, profile: SessionProfile) -> "TrajectoryProvider": ...

    @classmethod
    def capabilities(cls) -> ProviderCapabilities: ...

    @property
    def token_ids(self) -> list[int]: ...

    def prepare_pretokenized(
        self,
        request_messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        *,
        tito_tokenizer: TITOTokenizer,
    ) -> list[int]: ...

    def update_pretokenized_state(
        self,
        request_messages: list[dict[str, Any]],
        assistant_message: dict[str, Any],
        prompt_token_ids: list[int],
        completion_token_ids: list[int],
        max_trim_tokens: int,
        **kwargs: Any,
    ) -> None: ...

    def append_record(self, record: SessionRecord) -> None: ...

    def close(self) -> ProviderCloseResult: ...


# ---------------------------------------------------------------------------
# Optional provider hooks
# ---------------------------------------------------------------------------
# The methods below are NOT part of the ``TrajectoryProvider`` Protocol on
# purpose. ``SessionCore`` probes each one with ``getattr(session, name, None)``
# and falls back to its own hardcoded behaviour when the attribute is missing.
# Adding them to the Protocol would retroactively invalidate every provider that
# already satisfies the contract, which is exactly what this indirection avoids.
#
#   def create_result(self) -> Mapping[str, Any]:
#       Extra keys merged into the ``POST /sessions`` response. Returning
#       ``None`` or ``{}`` adds nothing. Overriding ``session_id`` or
#       ``profile`` is rejected: both are parsed by existing clients.
#
#   def extra_request_fields(self) -> Mapping[str, Any]:
#       Fields merged into the upstream request body. Applied after the core's
#       own replay switches so a provider can override them, but before the
#       hardcoded ``no_stop_trim``/``logprobs`` invariants.
#
#   def consume_meta(self, response: Mapping[str, Any]) -> tuple[str, ...] | None:
#       Called with the full upstream response. The provider stores whatever it
#       needs and returns the meta_info keys to strip from the client copy.
#       Returning ``None`` keeps the default strip list.
#
#   def sequence_token_ids(self, sequence_id: str) -> list[int]:
#       Multi-sequence providers only. The ``token_ids`` property keeps its
#       current meaning (the primary sequence) and is not repurposed.
#
#   def export(self, view: str) -> Mapping[str, Any]:
#       Extra metadata for ``GET /sessions/{id}?view=summary``.
#
#   def acknowledge_export(self, body: Mapping[str, Any]) -> None:
#       Called by ``POST /sessions/{id}/export-ack``. Raise
#       ``HistoryPrefixMismatchError`` (409) when the reported counts disagree.
#
# ``session_token`` (a plain attribute) lets a provider own the per-rollout bearer
# token checked under ``--session-auth-mode enforce``. Providers that do not set
# it fall back to the admin key, which is the only credential that exists then.

OPTIONAL_PROVIDER_HOOKS = (
    "create_result",
    "extra_request_fields",
    "consume_meta",
    "sequence_token_ids",
    "export",
    "acknowledge_export",
)


LifecycleCallback = Callable[[str, str, Mapping[str, Any]], Any]
ProviderFactory = Callable[..., Any]


class ProviderConfigurationError(ValueError):
    """Raised when an explicitly selected provider cannot be resolved."""


class SessionProviderRegistry:
    """Explicit profile-to-factory registry local to one server process."""

    def __init__(self, *, lifecycle_callback: LifecycleCallback | None = None):
        self._factories: dict[str, ProviderFactory] = {}
        self._capabilities: dict[str, ProviderCapabilities] = {}
        self.lifecycle_callback = lifecycle_callback

    def register(self, name: str, factory: ProviderFactory, *, capabilities: ProviderCapabilities | None = None) -> None:
        if not name or name == "linear":
            raise ProviderConfigurationError("provider name must be non-empty and not 'linear'")
        if not callable(factory):
            raise ProviderConfigurationError(f"provider factory for {name!r} is not callable")
        self._factories[name] = factory
        self._capabilities[name] = capabilities or self._read_capabilities(factory)

    def register_dotted_path(self, name: str, dotted_path: str) -> None:
        module_name, separator, attribute = dotted_path.rpartition(".")
        if not separator:
            raise ProviderConfigurationError(f"provider path must be module.attribute: {dotted_path!r}")
        try:
            factory = getattr(importlib.import_module(module_name), attribute)
        except (ImportError, AttributeError) as exc:
            raise ProviderConfigurationError(f"failed to import provider {dotted_path!r}: {exc}") from exc
        self.register(name, factory)

    def create(self, profile: SessionProfile, *, session_id: str) -> Any:
        if profile.name == "linear":
            from miles.rollout.session.linear_trajectory import LinearTrajectory

            return LinearTrajectory()
        factory = self._factories.get(profile.name)
        if factory is None:
            raise ProviderConfigurationError(f"unknown session profile: {profile.name}")
        try:
            create = getattr(factory, "create", None)
            if callable(create):
                return create(session_id=session_id, profile=profile)
            return factory(session_id=session_id, profile=profile)
        except Exception as exc:
            raise ProviderConfigurationError(f"failed to create session profile {profile.name!r}: {exc}") from exc

    def capabilities(self, profile_name: str | None = None) -> dict[str, Any]:
        if profile_name == "linear":
            from miles.rollout.session.linear_trajectory import LinearTrajectory

            capability = self._read_capabilities(LinearTrajectory)
            profiles = {"linear": sorted(capability.names)}
            return {"api_version": capability.api_version, "profiles": profiles}
        if profile_name is None:
            from miles.rollout.session.linear_trajectory import LinearTrajectory

            capabilities = {"linear": self._read_capabilities(LinearTrajectory), **self._capabilities}
            versions = {capability.api_version for capability in capabilities.values()}
            if len(versions) != 1:
                raise ProviderConfigurationError("registered session providers use incompatible API versions")
            profiles = {name: sorted(capability.names) for name, capability in capabilities.items()}
            return {"api_version": versions.pop(), "profiles": profiles}
        capability = self._capabilities.get(profile_name)
        if capability is None:
            raise ProviderConfigurationError(f"unknown session profile: {profile_name}")
        return {"api_version": capability.api_version, "profiles": {profile_name: sorted(capability.names)}}

    @staticmethod
    def _read_capabilities(factory: Any) -> ProviderCapabilities:
        method = getattr(factory, "capabilities", None)
        if callable(method):
            value = method()
            if isinstance(value, ProviderCapabilities):
                return value
        return ProviderCapabilities()

    def emit(self, event: str, session_id: str, metadata: Mapping[str, Any] | None = None) -> None:
        callback = self.lifecycle_callback
        if callback is None:
            return
        try:
            callback(event, session_id, metadata or {})
        except Exception:
            # Observability must not change session correctness or cleanup.
            return

    @property
    def profiles(self) -> tuple[str, ...]:
        return ("linear", *self._factories.keys())
