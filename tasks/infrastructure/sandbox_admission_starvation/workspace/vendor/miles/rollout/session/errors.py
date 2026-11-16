"""Error types for the session module.

Hierarchy
---------
SessionError (base)
├── SessionNotFoundError       → 404  session does not exist
├── MessageValidationError     → 400  messages structure/content invalid
├── HistoryPrefixMismatchError → 409  client rewrote history the server owns
├── TokenizationError          → 500  TITO tokenizer / prefix mismatch
├── UpstreamResponseError      → 502  SGLang response invalid or unexpected
└── EngineDrainingError        → 503  shared EngineClient is draining
"""


class SessionError(Exception):
    """Base class for all session-related errors."""

    status_code: int = 500


class SessionNotFoundError(SessionError):
    """Raised when the requested session ID does not exist."""

    status_code: int = 404


class SessionConfigurationError(SessionError):
    """Raised when an explicitly requested session profile is unavailable."""

    status_code: int = 400


class SessionLifecycleError(SessionError):
    """Raised after session state is removed but provider cleanup fails."""

    status_code: int = 500


class UnsupportedControlError(SessionError):
    """Raised when an iteration-two control is requested."""

    status_code: int = 501


class MessageValidationError(SessionError):
    """Raised when request messages fail structural validation.

    Examples: user message after assistant, messages not append-only,
    rollback failed (no assistant checkpoint in matched prefix).
    """

    status_code: int = 400


class HistoryPrefixMismatchError(SessionError):
    """Raised when the history replayed by the agent contradicts server state.

    The server owns the token-level trajectory, so a client that edits, drops or
    re-writes an earlier turn cannot be reconciled: re-tokenizing would silently
    produce log-probabilities that never belonged to the recorded generation.
    Not retryable — the caller must start a new session.
    """

    status_code: int = 409


class TokenizationError(SessionError):
    """Raised when TITO tokenization invariants are violated.

    Examples: pretokenized prefix mismatch between stored and new token IDs.
    """

    status_code: int = 500


class UpstreamResponseError(SessionError):
    """Raised when the upstream SGLang response is invalid or unexpected.

    Examples: missing meta_info, assistant content is None,
    output_token_logprobs length mismatch.
    """

    status_code: int = 502


class EngineDrainingError(SessionError):
    """Raised when the shared EngineClient is draining and refuses to resume
    an aborted black-box generation call."""

    status_code: int = 503
