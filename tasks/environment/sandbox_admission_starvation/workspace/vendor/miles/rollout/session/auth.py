"""Two keys for the session server: one for the trainer, one for the agent.

The split is by *caller*, not by data-versus-control:

- The trainer creates, reads, acknowledges and deletes sessions, and drives the
  weight-update endpoints. It holds the admin key.
- The agent inside the sandbox only generates. It holds a token minted for its
  one rollout, returned in the ``POST /sessions`` response.

Both checks are disabled by default (``--session-auth-mode off``). The existing
callers send no ``Authorization`` header at all, so turning this on by default
would break every one of them. ``off`` short-circuits before any comparison, so
it costs one boolean per request and changes no response.
"""

import hmac
import logging
import secrets

from fastapi import Request

from miles.rollout.session.errors import SessionError

logger = logging.getLogger(__name__)

AUTH_MODE_OFF = "off"
AUTH_MODE_ENFORCE = "enforce"


class AuthenticationError(SessionError):
    """No usable credential was presented."""

    status_code: int = 401


class AuthorizationError(SessionError):
    """A credential was presented but does not grant this operation."""

    status_code: int = 403


def bearer_token(request: Request) -> str:
    """Read the credential from ``Authorization: Bearer`` or ``X-API-Key``."""
    header = request.headers.get("authorization") or ""
    if header.lower().startswith("bearer "):
        return header[7:].strip()
    return (request.headers.get("x-api-key") or "").strip()


class SessionAuth:
    """Credential checks shared by every authenticated route."""

    def __init__(self, mode: str = AUTH_MODE_OFF, admin_key: str | None = None):
        if mode not in (AUTH_MODE_OFF, AUTH_MODE_ENFORCE):
            raise ValueError(f"unknown session auth mode: {mode!r}")
        self.mode = mode
        # Minting a key even when enforcement is off keeps the enabled path from
        # having a separate "is there a key yet" state to get wrong.
        self.admin_key = admin_key or secrets.token_hex(32)
        if mode == AUTH_MODE_ENFORCE and not admin_key:
            logger.warning(
                "[session] auth mode is 'enforce' but no admin key was configured; "
                "a random one was generated and only this process knows it"
            )

    @property
    def enabled(self) -> bool:
        return self.mode == AUTH_MODE_ENFORCE

    def check_admin(self, request: Request) -> None:
        if not self.enabled:
            return
        supplied = bearer_token(request)
        if not supplied:
            raise AuthenticationError("admin key required")
        if not hmac.compare_digest(supplied, self.admin_key):
            raise AuthorizationError("invalid admin key")

    def check_session(self, request: Request, session_token: str | None) -> None:
        """Accept either this session's own token or the admin key."""
        if not self.enabled:
            return
        supplied = bearer_token(request)
        if not supplied:
            raise AuthenticationError("session token required")
        if session_token and hmac.compare_digest(supplied, session_token):
            return
        if hmac.compare_digest(supplied, self.admin_key):
            return
        raise AuthorizationError("invalid session token")
