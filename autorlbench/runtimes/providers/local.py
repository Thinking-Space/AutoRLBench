from __future__ import annotations

import os
import subprocess
import time
from dataclasses import replace
from pathlib import Path
from typing import Optional

from ...core.types import CommandResult, EnvironmentImage
from ..base import CommandSpec, EnvironmentArtifact, SandboxInstance, SandboxRequest, SandboxRuntime
from ..factory import register_runtime


@register_runtime("local")
class LocalRuntime(SandboxRuntime):
    """Local runtime for development and smoke tests only; provides no security isolation."""

    name = "local"

    @property
    def needs_environment_image(self) -> bool:
        return False

    def ensure_environment(
        self, image: EnvironmentImage, *, rebuild: bool = False
    ) -> EnvironmentArtifact:
        # Local mode builds no image and uses the host environment; the task validator still checks environment.
        return EnvironmentArtifact(backend=self.name)

    def path(self, request: SandboxRequest, name: str) -> str:
        oracle = request.host_oracle or request.host_task_root / "oracle"
        verifier = request.host_verifier or request.host_task_root / "verifier"
        paths = {
            "task_root": request.host_task_root,
            "visible_task": request.host_visible_task,
            "workspace": request.host_workspace,
            "results": request.host_results,
            "oracle": oracle,
            "verifier": verifier,
            # Local mode has no mounts, so logical paths are host paths. The three optional directories exist only
            # when the caller actually provided them; a missing one must raise unknown path, not return None.
            "environment": request.host_environment,
            "prepared": request.host_prepared,
            "models": request.host_models,
            "datasets": request.host_datasets,
        }
        selected = paths.get(name)
        if selected is None:
            raise ValueError(f"unknown sandbox path: {name}")
        return str(selected.resolve())

    def run(
        self,
        request: SandboxRequest,
        spec: CommandSpec,
        *,
        artifact: Optional[EnvironmentArtifact] = None,
    ) -> CommandResult:
        spec = self._merged_spec(request, spec)
        spec = replace(spec, cwd=Path(self.path(request, request.workdir)))
        argv = spec.to_argv()
        env = os.environ.copy()
        env.update({str(key): str(value) for key, value in spec.env.items()})
        start = time.monotonic()
        timed_out = False
        process = subprocess.Popen(
            argv,
            cwd=str(spec.cwd or Path.cwd()),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            stdout, stderr = process.communicate(timeout=spec.timeout_s)
        except subprocess.TimeoutExpired:
            timed_out = True
            process.kill()
            stdout, stderr = process.communicate()
        return CommandResult(
            argv=argv,
            returncode=process.returncode if process.returncode is not None else 124,
            stdout=stdout,
            stderr=stderr,
            timed_out=timed_out,
            duration_s=time.monotonic() - start,
            cwd=str(spec.cwd or Path.cwd()),
        )

    def create_sandbox(
        self,
        request: SandboxRequest,
        *,
        artifact: EnvironmentArtifact,
    ) -> SandboxInstance:
        return SandboxInstance(self.name, str(request.host_workspace), request)

    def exec(self, sandbox: SandboxInstance, spec: CommandSpec) -> CommandResult:
        return self.run(sandbox.request, spec)

    def close_sandbox(self, sandbox: SandboxInstance) -> None:
        return None
