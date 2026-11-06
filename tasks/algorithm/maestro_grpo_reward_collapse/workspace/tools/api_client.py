"""Resolve portable model names against existing OpenAI-compatible services.

Remote model IDs are server-owned identifiers, not local checkpoint paths. Query
the service rather than baking another machine's checkpoint directory into code.
"""
from functools import wraps
from pathlib import PurePosixPath
import logging
import os
import time
import urllib.request

from openai import APIConnectionError, APIStatusError, OpenAI as _OpenAI


class OpenAI(_OpenAI):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._portable_model_ids = {}
        create = self.chat.completions.create

        @wraps(create)
        def portable_create(*args, **kwargs):
            requested = kwargs.get("model", "")
            deadline = None
            while True:
                try:
                    if requested.startswith("models/"):
                        kwargs["model"] = self.resolve_model_id(requested)
                    return create(*args, **kwargs)
                except (APIConnectionError, APIStatusError) as error:
                    if isinstance(error, APIStatusError) and error.status_code not in {429, 500, 502, 503, 504}:
                        raise
                    if deadline is None:
                        grace = max(0, float(os.environ.get("MODEL_RECOVERY_TIMEOUT", "0")))
                        deadline = time.monotonic() + grace
                        if grace:
                            logging.warning("Model service %s unavailable; waiting up to %.0fs for recovery",
                                            self.base_url, grace)
                    if not self._wait_for_recovery(deadline):
                        raise
                    self._portable_model_ids.clear()

        self.chat.completions.create = portable_create

    def _wait_for_recovery(self, deadline):
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        health_url = str(self.base_url).rstrip('/').removesuffix('/v1') + '/health'
        while time.monotonic() < deadline:
            remaining = deadline - time.monotonic()
            time.sleep(min(2, remaining))
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            try:
                with opener.open(health_url, timeout=min(3, remaining)) as response:
                    if response.status == 200:
                        return True
            except OSError:
                pass
        return False

    def resolve_model_id(self, requested):
        if requested not in self._portable_model_ids:
            model_ids = [model.id for model in self.models.list().data]
            name = PurePosixPath(requested).name.lower()
            matches = [model for model in model_ids if PurePosixPath(model).name.lower() == name]
            if len(matches) == 1:
                actual = matches[0]
            elif len(model_ids) == 1:
                # Existing routing ports each serve one specialized model.
                actual = model_ids[0]
            elif requested in model_ids:
                actual = requested
            else:
                raise ValueError(
                    f"Cannot resolve {requested!r} at {self.base_url}; "
                    f"service exposes {model_ids!r}"
                )
            self._portable_model_ids[requested] = actual
        return self._portable_model_ids[requested]
