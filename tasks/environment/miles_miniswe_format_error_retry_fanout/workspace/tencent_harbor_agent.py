#!/usr/bin/env python3
"""Route the public Harbor agent function to the Tencent AGS environment."""

from __future__ import annotations

import os
import re
import base64
from pathlib import Path
from typing import Any

import harbor_agent_function
import harbor.agents.installed.mini_swe_agent as mini_swe_agent
from harbor.agents.installed.mini_swe_agent import MiniSweAgent


_original_environment_config = harbor_agent_function._environment_config
_INSTALL_TIMEOUT = int(os.getenv("TENCENT_AGS_AGENT_INSTALL_TIMEOUT", "1800"))
_UPLOAD_CHUNK_BYTES = int(os.getenv("TENCENT_AGS_UPLOAD_CHUNK_BYTES", str(32 * 1024 * 1024)))


async def _install_mini_swe_agent(self: MiniSweAgent, environment: Any) -> None:
    wheelhouse_tar = Path(os.environ["TENCENT_AGS_AGENT_WHEELHOUSE_TAR"]).expanduser()
    if not wheelhouse_tar.is_file():
        raise FileNotFoundError(
            "set TENCENT_AGS_AGENT_WHEELHOUSE_TAR to a Python 3.10 wheelhouse tar.gz"
        )

    sandbox = environment._sandbox
    with wheelhouse_tar.open("rb") as wheelhouse_file:
        part_index = 0
        while chunk := wheelhouse_file.read(_UPLOAD_CHUNK_BYTES):
            await sandbox.files.write(
                f"/tmp/agent-wheelhouse.tar.gz.part-{part_index:04d}",
                chunk,
                user="root",
                request_timeout=_INSTALL_TIMEOUT,
                use_octet_stream=True,
            )
            part_index += 1

    version_spec = f"=={self._version}" if self._version else ""
    output_cap_b64 = base64.b64encode(
        mini_swe_agent._OUTPUT_CAP_PATCH.format(
            cap_bytes=mini_swe_agent.MAX_COMMAND_OUTPUT_BYTES
        ).encode()
    ).decode()
    step_timing_b64 = base64.b64encode(
        mini_swe_agent._STEP_TIMING_PATCH.encode()
    ).decode()
    await self.exec_as_root(
        environment,
        command=(
            "set -euo pipefail; "
            "cat /tmp/agent-wheelhouse.tar.gz.part-* > /tmp/agent-wheelhouse.tar.gz; "
            "tar -xzf /tmp/agent-wheelhouse.tar.gz -C /tmp; "
            "/usr/bin/python3.10 -m pip install --no-index "
            "--find-links /tmp/agent-wheelhouse uv==0.7.13; "
            "export UV_OFFLINE=1; "
            'export PATH="/usr/local/bin:/root/.local/bin:$PATH"; '
            f"uv tool install mini-swe-agent{version_spec} "
            "--with 'litellm[proxy]' "
            "--find-links /tmp/agent-wheelhouse "
            "--python /usr/bin/python3.10; "
            'agent_python="$(uv tool dir)/mini-swe-agent/bin/python"; '
            f'echo {output_cap_b64} | base64 -d | "$agent_python" - && '
            f'echo {step_timing_b64} | base64 -d | "$agent_python" - && '
            "mini-swe-agent --help && "
            "/usr/bin/python3.10 -m pip install --no-index "
            "--find-links /tmp/agent-wheelhouse "
            "swebench==4.0.3 datasets==2.16.1 'fastcore<1.11'"
        ),
        env={"UV_OFFLINE": "1"},
        timeout_sec=_INSTALL_TIMEOUT,
    )


MiniSweAgent.install = _install_mini_swe_agent


def _environment_config():
    config = _original_environment_config()
    template = os.getenv("TENCENT_AGS_TEMPLATE", "").strip()
    if template:
        config.kwargs.setdefault("prebuilt_template_id", template)
    config.import_path = "tencent_ags_e2b:E2BEnvironment"
    return config


harbor_agent_function._environment_config = _environment_config


async def run(
    base_url: str,
    prompt: Any,
    request_kwargs: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    **kwargs,
) -> dict[str, Any]:
    public_session_url = os.getenv("SESSION_SERVER_PUBLIC_URL", "").strip().rstrip("/")
    if public_session_url:
        base_url = re.sub(r"^https?://[^/]+", public_session_url, base_url)
    return await harbor_agent_function.run(
        base_url,
        prompt,
        request_kwargs=request_kwargs,
        metadata=metadata,
        **kwargs,
    )
