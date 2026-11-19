#!/usr/bin/env python3
"""Tencent AGS adapter for Harbor's public E2B environment."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from pathlib import Path
from typing import Any

from harbor.environments.e2b import E2BEnvironment as PublicE2BEnvironment

logger = logging.getLogger(__name__)

_OCI_DIGEST_SUFFIX = re.compile(r"@sha256:[0-9a-fA-F]{64}$")
_CANCEL_KILL_TIMEOUT_SEC = float(os.getenv("TENCENT_AGS_CANCEL_KILL_TIMEOUT", "20"))


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "t", "yes", "on"}


def _parse_image_rewrites(raw: str) -> list[tuple[str, str]]:
    rules: list[tuple[str, str]] = []
    for item in (raw or "").split(","):
        if "=" not in item:
            continue
        prefix, replacement = item.split("=", 1)
        if prefix.strip():
            rules.append((prefix.strip(), replacement.strip()))
    return rules


def _rewrite_image(image: str, rules: list[tuple[str, str]]) -> str:
    for prefix, replacement in rules:
        if image.startswith(prefix):
            return replacement + image[len(prefix) :]
    return image


def _normalize_image_ref(image: str) -> str:
    """Convert an OCI digest reference to the tagged form AGS accepts."""
    match = _OCI_DIGEST_SUFFIX.search(image)
    if match is None:
        return image
    repository = image[: match.start()]
    if ":" in repository.rsplit("/", 1)[-1]:
        return repository
    tag = os.getenv("TENCENT_AGS_DIGEST_TAG", "latest").strip() or "latest"
    return f"{repository}:{tag}"


def resolve_task_image(image: str | None, environment_dir: Path) -> str:
    if not image:
        dockerfile = environment_dir / "Dockerfile"
        match = re.search(r"^FROM\s+(\S+)", dockerfile.read_text(encoding="utf-8"), re.MULTILINE)
        if match is None:
            raise ValueError(f"no docker_image or Dockerfile FROM in {dockerfile}")
        image = match.group(1)
    image = _rewrite_image(
        image,
        _parse_image_rewrites(os.getenv("TENCENT_AGS_IMAGE_REWRITE", "")),
    )
    if _env_flag("TENCENT_AGS_STRIP_IMAGE_DIGEST", default=True):
        image = _normalize_image_ref(image)
    return image


def build_custom_config(image: str, cpus: int, memory_mb: int) -> dict[str, Any]:
    envd_port = int(os.getenv("TENCENT_AGS_ENVD_PORT", "49983"))
    registry_type = os.getenv("TENCENT_AGS_IMAGE_REGISTRY_TYPE", "enterprise").strip()
    if not registry_type:
        raise ValueError("TENCENT_AGS_IMAGE_REGISTRY_TYPE cannot be empty")
    return {
        "image": image,
        "imageRegistryType": registry_type,
        "command": ["/bin/sh", "-c"],
        "args": [f"/mnt/envd -port {envd_port}"],
        "resources": {
            "cpu": str(cpus),
            "memory": f"{memory_mb}Mi",
        },
    }


def build_create_kwargs(
    *,
    template: str,
    metadata: dict[str, str],
    timeout: int,
    envs: dict[str, str] | None = None,
    allow_internet_access: bool = True,
    network: Any = None,
) -> dict[str, Any]:
    if not template.strip():
        raise ValueError("Tencent AGS requires a non-empty fixed template")
    create_kwargs: dict[str, Any] = {
        "template": template,
        "metadata": metadata,
        "timeout": timeout,
        "api_key": os.environ["E2B_API_KEY"],
    }
    if envs:
        create_kwargs["envs"] = envs
    if allow_internet_access is not None:
        create_kwargs["allow_internet_access"] = allow_internet_access
    if network is not None:
        create_kwargs["network"] = network

    domain = os.getenv("E2B_DOMAIN", "").strip()
    api_url = os.getenv("E2B_API_URL", "").strip()
    if domain:
        create_kwargs["domain"] = domain
    if api_url:
        create_kwargs["api_url"] = api_url
    if proxy := os.getenv("E2B_PROXY", "").strip():
        create_kwargs["proxy"] = proxy
    if sandbox_url := os.getenv("E2B_SANDBOX_URL", "").strip():
        create_kwargs["sandbox_url"] = sandbox_url
    if _env_flag("TENCENT_AGS_FORCE_HTTP1", default=False):
        create_kwargs["http_version"] = "1.1"
    if request_timeout := os.getenv("TENCENT_AGS_REQUEST_TIMEOUT", "").strip():
        create_kwargs["request_timeout"] = float(request_timeout)
    if retries := os.getenv("TENCENT_AGS_CREATE_RETRIES", "").strip():
        create_kwargs["retries"] = int(retries)
    return create_kwargs


class E2BEnvironment(PublicE2BEnvironment):
    """Create AGS sandboxes directly from a task image and a fixed template."""

    def _file_user(self) -> str:
        return str(self.default_user or "root")

    async def _sandbox_upload_file(self, source_path: Path | str, target_path: str):
        if not self._sandbox:
            raise RuntimeError("Sandbox not found. Please start the environment first.")
        await self._sandbox.files.write(
            target_path,
            Path(source_path).read_bytes(),
            user=self._file_user(),
        )

    async def _sandbox_upload_dir(self, source_dir: Path | str, target_dir: str):
        if not self._sandbox:
            raise RuntimeError("Sandbox not found. Please start the environment first.")
        from e2b.sandbox.filesystem.filesystem import WriteEntry

        files: list[WriteEntry] = []
        for file_path in Path(source_dir).rglob("*"):
            if file_path.is_file():
                files.append(
                    WriteEntry(
                        path=str(
                            Path(target_dir) / file_path.relative_to(Path(source_dir))
                        ),
                        data=file_path.read_bytes(),
                    )
                )
        for index in range(0, len(files), self._UPLOAD_BATCH_SIZE):
            batch = files[index : index + self._UPLOAD_BATCH_SIZE]
            await self._sandbox.files.write_files(
                batch,
                user=self._file_user(),
            )

    async def _sandbox_download_file(self, source_path: str, target_path: Path | str):
        if not self._sandbox:
            raise RuntimeError("Sandbox not found. Please start the environment first.")
        Path(target_path).write_bytes(
            await self._sandbox.files.read(
                source_path,
                format="bytes",
                user=self._file_user(),
            )
        )

    async def _sandbox_download_dir(self, source_dir: str, target_dir: Path | str):
        if not self._sandbox:
            raise RuntimeError("Sandbox not found. Please start the environment first.")
        from e2b.sandbox.filesystem.filesystem import FileType

        results = await self._sandbox.files.list(
            source_dir,
            user=self._file_user(),
        )
        for result in results:
            if result.type == FileType.DIR:
                sub_target_dir = Path(target_dir) / Path(result.path).relative_to(
                    Path(source_dir)
                )
                sub_target_dir.mkdir(parents=True, exist_ok=True)
                await self._sandbox_download_dir(
                    source_dir=result.path,
                    target_dir=sub_target_dir,
                )
            if result.type == FileType.FILE:
                target_path = Path(target_dir) / Path(result.path).relative_to(
                    Path(source_dir)
                )
                target_path.parent.mkdir(parents=True, exist_ok=True)
                await self._sandbox_download_file(
                    source_path=result.path,
                    target_path=target_path,
                )

    async def _create_sandbox(self) -> None:
        template = self._prebuilt_template_id
        if not template:
            raise RuntimeError(
                "Tencent AGS requires prebuilt_template_id; set TENCENT_AGS_TEMPLATE"
            )

        image = resolve_task_image(
            self.task_env_config.docker_image,
            self.environment_dir,
        )
        cpus = self._effective_cpus or self.task_env_config.cpus or 1
        memory_mb = self._effective_memory_mb or self.task_env_config.memory_mb or 4096
        metadata = {
            "environment_name": self.environment_name,
            "session_id": self.session_id,
            "purpose": "miles-harbor-mini-swe-rollout",
            "x-custom-config": json.dumps(
                build_custom_config(image, int(cpus), int(memory_mb)),
                separators=(",", ":"),
            ),
        }
        timeout = self._sandbox_timeout_secs
        if self._max_lifetime_secs is not None:
            metadata["max_lifetime_secs"] = str(self._max_lifetime_secs)
            timeout = min(timeout, self._max_lifetime_secs)

        create_kwargs = build_create_kwargs(
            template=template,
            metadata=metadata,
            timeout=timeout,
            envs=self._startup_env(),
            allow_internet_access=self.network_policy.network_mode.value != "no-network",
            network=self._sandbox_create_network_options(),
        )
        create_task = asyncio.ensure_future(_async_sandbox_create(**create_kwargs))
        try:
            self._sandbox = await asyncio.shield(create_task)
            return
        except asyncio.CancelledError:
            sandbox = None
            try:
                sandbox = await asyncio.wait_for(
                    asyncio.shield(create_task),
                    timeout=_CANCEL_KILL_TIMEOUT_SEC,
                )
            except BaseException:
                create_task.cancel()
            if sandbox is not None:
                self._sandbox = sandbox
                try:
                    await asyncio.wait_for(
                        asyncio.shield(sandbox.kill()),
                        timeout=_CANCEL_KILL_TIMEOUT_SEC,
                    )
                    self._sandbox = None
                    logger.info(
                        "killed AGS sandbox allocated during cancellation: %s",
                        getattr(sandbox, "sandbox_id", "<unknown>"),
                    )
                except BaseException:
                    logger.warning(
                        "could not kill AGS sandbox %s allocated during cancellation",
                        getattr(sandbox, "sandbox_id", "<unknown>"),
                        exc_info=True,
                    )
            raise


async def _async_sandbox_create(**kwargs: Any):
    from e2b import AsyncSandbox

    return await AsyncSandbox.create(**kwargs)
