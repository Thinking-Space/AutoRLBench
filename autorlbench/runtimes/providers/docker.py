from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Sequence

from ...core.types import (
    DATASETS_DIR,
    MODELS_DIR,
    PREPARED_DIR,
    AUTORLBENCH_ENVIRONMENT_DIR,
    TOOLING_DIR,
    CommandResult,
    EnvironmentImage,
)
from ..base import (
    CommandSpec,
    EnvironmentArtifact,
    SandboxInstance,
    SandboxRequest,
    SandboxRuntime,
)
from ..factory import register_runtime


# ray + sglang workers pass tensors through /dev/shm, and NCCL allocates buffers there too. Docker's
# default is 64MB, which blows up on the first rollout with an unhelpful bus error.
DEFAULT_SHM_SIZE = "16g"


@dataclass(frozen=True)
class DockerMount:
    """One host-to-container directory mount."""

    host_path: Path
    container_path: str
    read_only: bool = False

    def to_arg(self) -> str:
        suffix = ":ro" if self.read_only else ""
        return f"{self.host_path}:{self.container_path}{suffix}"


@register_runtime("docker")
class DockerRuntime(SandboxRuntime):
    """Docker provider, handling images, mounts, and networking in one place."""

    name = "docker"

    @property
    def needs_environment_image(self) -> bool:
        return True

    def ensure_environment(
        self, image: EnvironmentImage, *, rebuild: bool = False
    ) -> EnvironmentArtifact:
        """Use the image if it exists; otherwise build it from docker/images/<name>.Dockerfile.

        Tags are fixed, so "exists" does not mean "matches the Dockerfile": after changing the Dockerfile or
        tooling scripts under docker/, rebuild (`autorlbench build-images --rebuild`).
        Rebuilds use docker's layer cache, so unchanged layers are not rerun.
        """
        if rebuild or not self._image_exists(image.tag):
            if not image.dockerfile.is_file():
                raise RuntimeError(
                    f"image {image.tag!r} not found locally and {image.dockerfile} "
                    "does not exist to build it from"
                )
            # Build logs are progress for humans, not command results; send them to stderr and keep stdout for CLI output.
            subprocess.run(
                [
                    "docker",
                    "build",
                    "-f",
                    str(image.dockerfile),
                    "-t",
                    image.tag,
                    str(image.context),
                ],
                stdout=sys.stderr,
                check=True,
            )
        return EnvironmentArtifact(backend=self.name, reference=image.tag)

    def path(self, request: SandboxRequest, name: str) -> str:
        paths = {
            "task_root": "/task",
            "visible_task": "/task",
            "workspace": "/workspace",
            "results": "/results",
            "environment": AUTORLBENCH_ENVIRONMENT_DIR,
            "prepared": PREPARED_DIR,
            "models": MODELS_DIR,
            "datasets": DATASETS_DIR,
        }
        selected_paths = {
            "oracle": request.host_oracle,
            "verifier": request.host_verifier,
        }
        if name in selected_paths and selected_paths[name] is not None:
            selected = selected_paths[name]
            try:
                relative = selected.resolve().relative_to(request.host_task_root.resolve())
            except ValueError as exc:
                raise ValueError(f"{name} must stay under the selected task root") from exc
            return f"/task/{relative.as_posix()}"
        paths.update({"oracle": "/task/oracle", "verifier": "/task/verifier"})
        try:
            return paths[name]
        except KeyError as exc:
            raise ValueError(f"unknown sandbox path: {name}") from exc

    def run(
        self,
        request: SandboxRequest,
        spec: CommandSpec,
        *,
        artifact: Optional[EnvironmentArtifact] = None,
    ) -> CommandResult:
        if artifact is None or artifact.backend != self.name or not artifact.reference:
            raise ValueError("docker runtime needs a built environment artifact")
        sandbox = self.create_sandbox(request, artifact=artifact)
        try:
            return self.exec(sandbox, spec)
        finally:
            self.close_sandbox(sandbox)

    def create_sandbox(
        self,
        request: SandboxRequest,
        *,
        artifact: EnvironmentArtifact,
    ) -> SandboxInstance:
        if artifact.backend != self.name or not artifact.reference:
            raise ValueError("docker runtime needs a built environment artifact")
        container_name = f"autorlbench-{request.stage}-{uuid.uuid4().hex[:12]}"
        env = self._proxy_env(request, self._merged_spec(request, CommandSpec()).env)
        argv = [
            "docker",
            "create",
            "--name",
            container_name,
            "-w",
            self.path(request, request.workdir),
        ]
        argv.extend(self._gpu_args(request))
        if request.network:
            argv.append(f"--network={request.network}")
        for mount in self._mounts(request):
            argv.extend(["-v", mount.to_arg()])
        env_args, process_env = self._docker_env(env)
        argv.extend(env_args)
        argv.extend(
            [
                "--entrypoint",
                "sh",
                artifact.reference,
                "-lc",
                "while :; do sleep 3600; done",
            ]
        )
        subprocess.run(
            argv,
            check=True,
            env=process_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            subprocess.run(
                ["docker", "start", container_name],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        except Exception:
            subprocess.run(["docker", "rm", "-f", container_name], check=False)
            raise
        return SandboxInstance(self.name, container_name, request)

    def exec(self, sandbox: SandboxInstance, spec: CommandSpec) -> CommandResult:
        if sandbox.backend != self.name:
            raise ValueError("sandbox belongs to a different runtime")
        request = sandbox.request
        merged = self._merged_spec(request, spec)
        merged = merged if request.stage != "agent" else self._with_env(
            merged,
            self._proxy_env(request, merged.env),
        )
        cwd = self.path(request, request.workdir)
        argv = ["docker", "exec", "-w", cwd]
        if merged.user is not None:
            argv.extend(["-u", str(merged.user)])
        env_args, process_env = self._docker_env(merged.env)
        argv.extend(env_args)
        argv.extend([sandbox.reference, *merged.to_argv()])
        return self._communicate(
            argv,
            timeout_s=merged.timeout_s,
            cwd=cwd,
            process_env=process_env,
            live_log_name=merged.live_log_name,
            live_log_to_stderr=merged.live_log_to_stderr,
            results_dir=request.host_results,
        )

    def close_sandbox(self, sandbox: SandboxInstance) -> None:
        if sandbox.backend != self.name:
            return
        subprocess.run(["docker", "rm", "-f", sandbox.reference], check=False)

    @staticmethod
    def _communicate(
        argv: list[str],
        *,
        timeout_s: Optional[int],
        cwd: str,
        process_env: Optional[dict[str, str]] = None,
        live_log_name: Optional[str] = None,
        live_log_to_stderr: bool = False,
        results_dir: Optional[Path] = None,
    ) -> CommandResult:
        start = time.monotonic()
        timed_out = False
        process = subprocess.Popen(
            argv,
            env=process_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if live_log_name and results_dir is not None:
            stdout, stderr, timed_out = DockerRuntime._stream_process(
                process,
                timeout_s=timeout_s,
                results_dir=results_dir,
                log_name=live_log_name,
                to_stderr=live_log_to_stderr,
            )
        else:
            try:
                stdout, stderr = process.communicate(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                timed_out = True
                process.kill()
                stdout, stderr = process.communicate()
        return CommandResult(
            argv=DockerRuntime._redact_argv(argv),
            returncode=process.returncode if process.returncode is not None else 124,
            stdout=stdout,
            stderr=stderr,
            timed_out=timed_out,
            duration_s=time.monotonic() - start,
            cwd=cwd,
        )

    @staticmethod
    def _stream_process(
        process: subprocess.Popen,
        *,
        timeout_s: Optional[int],
        results_dir: Path,
        log_name: str,
        to_stderr: bool = False,
    ) -> tuple[str, str, bool]:
        """Capture a process while making long setup commands observable."""
        results_dir.mkdir(parents=True, exist_ok=True)
        stdout_path = results_dir / f"{log_name}.stdout.log"
        stderr_path = results_dir / f"{log_name}.stderr.log"
        captured = {"stdout": [], "stderr": []}

        def drain(stream, path: Path, key: str) -> None:
            with open(path, "w", encoding="utf-8") as log_file:
                for line in iter(stream.readline, ""):
                    captured[key].append(line)
                    log_file.write(line)
                    log_file.flush()
                    if to_stderr:
                        sys.stderr.write(line)
                        sys.stderr.flush()
            stream.close()

        threads = [
            threading.Thread(
                target=drain,
                args=(process.stdout, stdout_path, "stdout"),
                daemon=True,
            ),
            threading.Thread(
                target=drain,
                args=(process.stderr, stderr_path, "stderr"),
                daemon=True,
            ),
        ]
        for thread in threads:
            thread.start()

        timed_out = False
        try:
            process.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            timed_out = True
            process.kill()
            process.wait()
        for thread in threads:
            thread.join()
        return "".join(captured["stdout"]), "".join(captured["stderr"]), timed_out

    @staticmethod
    def _docker_env(env: Mapping[str, str]) -> tuple[list[str], dict[str, str]]:
        argv: list[str] = []
        process_env = os.environ.copy()
        sensitive = ("KEY", "TOKEN", "SECRET", "PASSWORD")
        for key, value in sorted(env.items()):
            process_env[key] = value
            if any(token in key.upper() for token in sensitive):
                argv.extend(["-e", key])
            else:
                argv.extend(["-e", f"{key}={value}"])
        return argv, process_env

    @staticmethod
    def _with_env(spec: CommandSpec, env: Mapping[str, str]) -> CommandSpec:
        """Return a command with the supplied environment merged in."""
        return CommandSpec(
            argv=spec.argv,
            shell=spec.shell,
            cwd=spec.cwd,
            env=dict(env),
            timeout_s=spec.timeout_s,
            user=spec.user,
            live_log_name=spec.live_log_name,
            live_log_to_stderr=spec.live_log_to_stderr,
        )

    @staticmethod
    def _proxy_env(request: SandboxRequest, env: Mapping[str, str]) -> dict[str, str]:
        """Optionally expose a host-reachable proxy to task containers.

        Docker does not inherit the host's GUI/system proxy settings. The
        explicit AUTORLBENCH_DOCKER_PROXY opt-in keeps direct networking as the
        default and avoids forwarding a host localhost address by accident.
        """
        result = dict(env)
        proxy = os.environ.get("AUTORLBENCH_DOCKER_PROXY", "").strip()
        if proxy:
            for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
                result.setdefault(key, proxy)
        no_proxy = os.environ.get("AUTORLBENCH_DOCKER_NO_PROXY", "").strip()
        if no_proxy:
            result.setdefault("NO_PROXY", no_proxy)
        return result

    @staticmethod
    def _gpu_args(request: SandboxRequest) -> list[str]:
        """Translate the task-declared NVIDIA_VISIBLE_DEVICES into docker's `--gpus`.

        The image side (CUDA base image) has always assumed libcuda.so.1 is injected by `--gpus`; only
        this half was missing. Passing the variable via `-e`
        into the container does not make GPUs visible: the host's default runtime is runc, and only
        `--gpus` makes the nvidia hook actually inject device nodes and libcuda.so.1.

        With no declaration, no flag is added, so CPU-only tasks and stages (grading almost always is) behave
        exactly as before this wiring. `none` / `void` are NVIDIA_VISIBLE_DEVICES's own "no
        GPU" values and are treated the same way.

        AUTORLBENCH_DOCKER_GPUS is a host-side override, following the opt-in convention of _proxy_env:
        a machine with only 2 GPUs should still run a task whose card says 8.

        Prepare only downloads models and datasets and needs no GPU. `docker create`
        accepts --gpus, but on some hosts the NVIDIA runtime hook fails at `docker start` when it
        tries to bind the actual device nodes.

        The rerun has to run real training, so it takes the cards the task declares, exactly as the
        agent stage does. The agent gets them too, but the protocol forbids it from launching
        training -- there the GPU is headroom for cheap offline checks; the rerun is what actually
        fills it.
        """
        if request.stage == "prepare":
            return []
        spec = os.environ.get("AUTORLBENCH_DOCKER_GPUS", "").strip()
        if not spec:
            spec = str(request.task_env.get("NVIDIA_VISIBLE_DEVICES", "")).strip()
        if not spec or spec in {"none", "void"}:
            return []
        # `all` and `device=...` are already --gpus syntax; task cards hold a bare device list
        # ("0,1,2,3"), which must be wrapped for docker to recognize it.
        if spec != "all" and not spec.startswith("device="):
            spec = f"device={spec}"
        shm = os.environ.get("AUTORLBENCH_DOCKER_SHM_SIZE", "").strip() or DEFAULT_SHM_SIZE
        return ["--gpus", spec, f"--shm-size={shm}"]

    @staticmethod
    def _redact_argv(argv: list[str]) -> list[str]:
        redacted = list(argv)
        sensitive = ("KEY", "TOKEN", "SECRET", "PASSWORD")
        for index, value in enumerate(redacted[:-1]):
            if value != "-e":
                continue
            assignment = redacted[index + 1]
            key = assignment.split("=", 1)[0].upper()
            if any(token in key for token in sensitive) and "=" in assignment:
                redacted[index + 1] = f"{assignment.split('=', 1)[0]}=<redacted>"
        return redacted

    def _mounts(self, request: SandboxRequest) -> Sequence[DockerMount]:
        """What each stage mounts, and who can write.

        The differences between the tables are this design's access control, so each is spelled out:

        * **workspace**: writable by prepare, agent and rerun, read-only for verifier. Prepare writes because
          weights are downloaded to the relative path the task script references (`./Qwen2.5-0.5B-Instruct`);
          rerun writes because training itself writes checkpoints;
          the verifier is read-only because grading must not alter the scene.
        * **/task**: prepare, agent and rerun mount visible_task (instruction only);
          the verifier mounts the whole evaluation directory (with oracle and grading scripts). This is "no spoilers":
          prepare runs a script shipped with the task and has no reason to read the answer, and neither does a rerun.
        * **prepared**: writable by prepare, read-only for every later stage. The verifier reads the model
          path prepare delivered, but must not read a copy the agent modified, or the grading basis would be editable by the examinee.
          The rerun reads it too, read-only: that is what makes "the experiment ran on the assets prepare delivered" hold.
        * **models / datasets**: writable by prepare, read-only for every later stage. See
          SandboxRequest.host_models.
        """
        if request.stage == "prepare":
            mounts = [
                DockerMount(request.host_workspace, "/workspace"),
                DockerMount(request.host_visible_task, "/task", read_only=True),
                DockerMount(request.host_prepared, PREPARED_DIR),
            ]
            # The repo's docker/ mounted read-only: before running the script, prepare uses it to install the latest fetch/dataset
            # tools (see TOOLING_DIR / install-autorlbench-tooling.sh). Only prepare needs it.
            if request.host_tooling is not None:
                mounts.append(
                    DockerMount(request.host_tooling, TOOLING_DIR, read_only=True)
                )
        elif request.stage in {"agent", "rerun"}:
            # Rerun comes after the agent: the workspace stays writable, because the training script
            # writes checkpoints and ray writes scratch directories, and results stays writable
            # because that is where the logs grading reads land. The only difference from the agent
            # is nothing at all on /task -- a rerun runs the command the task declared and has no
            # more reason to see the oracle than the agent does.
            mounts = [
                DockerMount(request.host_workspace, "/workspace"),
                DockerMount(request.host_visible_task, "/task", read_only=True),
                DockerMount(request.host_results, "/results"),
            ]
        else:
            mounts = [
                DockerMount(request.host_workspace, "/workspace", read_only=True),
                DockerMount(request.host_task_root, "/task", read_only=True),
                DockerMount(request.host_results, "/results"),
            ]
        if request.host_environment is not None:
            # Read-only in all three stages: no stage has a reason to modify the task definition in the repo.
            mounts.append(
                DockerMount(request.host_environment, AUTORLBENCH_ENVIRONMENT_DIR, read_only=True)
            )
        if request.stage != "prepare" and request.host_prepared is not None:
            mounts.append(DockerMount(request.host_prepared, PREPARED_DIR, read_only=True))
        # Download caches: only prepare needs to write; the later two stages mount the same copy read-only.
        read_only = request.stage != "prepare"
        for host, container in (
            (request.host_models, MODELS_DIR),
            (request.host_datasets, DATASETS_DIR),
        ):
            if host is not None:
                mounts.append(DockerMount(host, container, read_only=read_only))
        return tuple(mounts)

    @staticmethod
    def _image_exists(tag: str) -> bool:
        result = subprocess.run(
            ["docker", "image", "inspect", tag],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return result.returncode == 0
