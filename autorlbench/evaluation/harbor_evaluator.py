from __future__ import annotations

import dataclasses
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from ..core.harbor_types import (
    HarborEvaluationConfig,
    HarborEvaluationReport,
    HarborRun,
    HarborTaskUnit,
    PreparedHarborTask,
    harbor_task_name,
)
from ..core.types import (
    DATASETS_DIR,
    MODELS_DIR,
    PREPARED_DIR,
    AUTORLBENCH_ENVIRONMENT_DIR,
    CommandResult,
    EnvironmentImage,
    EvaluationConfig,
    EvaluationPaths,
    EvaluationResult,
    TaskPackage,
)
from ..tasks import SchemaError, load_task_package
from .evaluator import TaskEvaluator
from .run_spec import (
    REPORT_FILENAME,
    RunSpec,
    RunSpecError,
    load_run_spec,
)
from ..utils.filesystem import copy_tree, ensure_dir, tree_digest, write_text


# Location of the activate hook in the image; see the tooling layer in docker/images/*.Dockerfile.
ACTIVATE_HOOK = "/etc/profile.d/autorlbench-activate.sh"


class HarborAdapterError(RuntimeError):
    """Configuration or result error at the boundary between AutoRLBench and Harbor."""


class HarborTaskMaterializer:
    """Materialize the selected AutoRLBench evaluations on the fly as a batch of Harbor tasks."""

    def materialize(
        self,
        units: Sequence[HarborTaskUnit],
        destination: Path,
        config: HarborEvaluationConfig,
    ) -> PreparedHarborTask:
        if not units:
            raise HarborAdapterError("at least one task is required")
        root = destination.resolve()
        # Harbor's -p/--path accepts "each subdirectory of a parent is a task", so batch and single-task share one shape.
        tasks_dir = ensure_dir(root / "tasks")
        prepared = PreparedHarborTask(root=root, tasks_dir=tasks_dir, units=units)
        self._reject_duplicate_task_names(units)
        for unit in units:
            self._materialize_one(unit, prepared.task_dir(unit), config)
        return prepared

    @staticmethod
    def _reject_duplicate_task_names(units: Sequence[HarborTaskUnit]) -> None:
        """Tasks with the same name would overwrite each other's output and could not be traced back from trials."""
        seen: Dict[str, Path] = {}
        for unit in units:
            name = unit.harbor_task_name
            previous = seen.get(name)
            if previous is not None:
                raise HarborAdapterError(
                    f"duplicate Harbor task name {name!r} from "
                    f"{previous} and {unit.package.paths.root}"
                )
            seen[name] = unit.package.paths.root

    def _materialize_one(
        self,
        unit: HarborTaskUnit,
        task_dir: Path,
        config: HarborEvaluationConfig,
    ) -> None:
        package = unit.package
        evaluation = unit.evaluation
        ensure_dir(task_dir)

        self._write_instruction(evaluation, task_dir)
        self._write_environment(unit, task_dir)
        self._write_tests(unit, task_dir)
        self._write_solution(evaluation, task_dir)
        self._write_task_config(unit, task_dir, config)
        self._write_manifest(package, evaluation, task_dir, config)

    @staticmethod
    def _write_instruction(evaluation: EvaluationPaths, task_dir: Path) -> None:
        instruction = evaluation.instruction_md.read_text(encoding="utf-8")
        write_text(task_dir / "instruction.md", instruction)

    def _write_environment(self, unit: HarborTaskUnit, task_dir: Path) -> None:
        """environment/ holds only the workspace contents, no Dockerfile.

        task.toml declares `docker_image`, so Harbor starts the prebuilt image directly; and when environment/
        has no Dockerfile / docker-compose.yaml, Harbor uploads the whole directory to the workdir
        (/workspace) after the container starts. That is exactly the copy the agent modifies.

        With staging, use the copy prepare wrote (weights may have been downloaded into it); copying via hard links
        means tens of GB of weights are not duplicated on disk.
        """
        source = unit.workspace or unit.package.paths.workspace
        environment_dir = task_dir / "environment"
        for name in ("Dockerfile", "docker-compose.yaml"):
            if (source / name).exists():
                # Anything in environment/ would be treated by Harbor as a build definition, not an uploaded file.
                raise HarborAdapterError(
                    f"workspace of {unit.package.task_id!r} contains {name}; Harbor "
                    "would build it instead of using the prebuilt image"
                )
        self._link_tree(source, environment_dir)
        if not any(environment_dir.iterdir()):
            # Harbor does not upload an empty directory, so /workspace would not exist; leave a placeholder file.
            write_text(environment_dir / ".autorlbench-workspace", "")

    @staticmethod
    def _link_tree(source: Path, destination: Path) -> None:
        """Hard link where possible, falling back to a regular copy across filesystems."""

        def link_or_copy(src: str, dst: str) -> None:
            try:
                os.link(src, dst)
            except OSError:
                shutil.copy2(src, dst)

        shutil.copytree(source, destination, symlinks=True, copy_function=link_or_copy)

    def _write_tests(self, unit: HarborTaskUnit, task_dir: Path) -> None:
        evaluation = unit.evaluation
        tests_dir = ensure_dir(task_dir / "tests")
        copy_tree(evaluation.verifier, tests_dir / "verifier")
        if evaluation.oracle.is_dir():
            copy_tree(evaluation.oracle, tests_dir / "oracle")
        else:
            shutil.copy2(evaluation.oracle, tests_dir / "oracle.md")

        test_script = tests_dir / "test.sh"
        write_text(test_script, self._verifier_wrapper(self._load_run_spec(unit)))
        test_script.chmod(0o755)

    @staticmethod
    def _load_run_spec(unit: HarborTaskUnit) -> Optional[RunSpec]:
        """Read the task's run_spec.json, turning a bad spec into a HarborAdapterError right here.

        There is no second chance to load it inside Harbor's verifier container -- the wrapper is a
        static script -- so all the validation happens at materialize time.
        """
        try:
            return load_run_spec(unit.evaluation.root, unit.package.task_id)
        except RunSpecError as exc:
            raise HarborAdapterError(str(exc)) from exc

    @staticmethod
    def _write_solution(evaluation: EvaluationPaths, task_dir: Path) -> None:
        """evaluations/<level>/solution/ becomes Harbor's solution/ as-is.

        Only `harbor run -a oracle` uploads it to the container and runs solve.sh, to verify end to end
        the "image + GPU + prepare outputs + verifier" chain; regular agents never see it.
        """
        source = evaluation.root / "solution"
        if not (source / "solve.sh").is_file():
            return
        copy_tree(source, task_dir / "solution")
        (task_dir / "solution" / "solve.sh").chmod(0o755)

    # These paths are constants inside Harbor's verifier container (see shared_env in
    # _write_task_config); a run_spec's $AUTORLBENCH_* references are expanded to literals here, so
    # the wrapper needs no substitution at run time.
    _HARBOR_SPEC_PATHS = {
        "AUTORLBENCH_WORKSPACE": "/workspace",
        "AUTORLBENCH_RESULTS": "/logs/artifacts",
    }

    @classmethod
    def _verifier_wrapper(cls, spec: Optional[RunSpec]) -> str:
        """The solution level's wrapper: report gate -> system rerun -> test.sh.

        Harbor only has agent and verifier stages, so the rerun folds into the verifier container:
        same machine, same workspace (environment_mode = "shared", writable), with the report.md the
        agent submitted and whatever it installed into PYTHONUSERBASE both still there. The only
        difference from `_run_rerun` on the docker backend is that there the rerun is its own
        sandbox; here the script pushes AUTORLBENCH_STAGE to rerun itself, runs the experiment, and
        restores it to verifier afterwards.

        The report gate applies to **every** solution task, including those with no run_spec: no
        report, no rerun and no grading, score 0. The issue level never reaches it (the level is read
        from AUTORLBENCH_EVALUATION_LEVEL at run time; the wrapper is one script for both).
        """
        rerun_section = ""
        if spec is not None:
            log_roots = " ".join(shlex.quote(root) for root in spec.log_roots)
            expanded_env = spec.expand_env(cls._HARBOR_SPEC_PATHS)
            env_exports = "\n".join(
                f"export {key}={shlex.quote(value)}"
                for key, value in expanded_env.items()
            )
            command_line = f"bash -c {shlex.quote(spec.command)}"
            # Enforce the timeout where it can be enforced: on the odd slimmed-down image without
            # `timeout`, run bare and let Harbor's own verifier timeout be the backstop.
            if spec.timeout_s is not None:
                command_line = (
                    f"command -v timeout >/dev/null 2>&1 "
                    f"&& exec timeout {spec.timeout_s}s {command_line}; "
                    f"exec {command_line}"
                )
            env_note = (
                "# run_spec's env ($AUTORLBENCH_* already expanded at materialize time). It goes on "
                "top of\n# the task's metadata env: run_spec is the value the protocol requires."
                if env_exports
                else ""
            )
            rerun_section = f'''# --- AutoRLBench rerun: re-run the experiment on the agent's workspace ----
# command / env come from the task's run_spec.json, expanded and inlined at materialize time.
# Clear the logs first: even if an agent broke protocol and trained, those logs do not count --
# grading only ever sees this pass.
for _root in {log_roots}; do
  rm -rf "$AUTORLBENCH_RESULTS/$_root"
done
unset _root
{env_note}
{env_exports}

# The activate hook's once-only guard (AUTORLBENCH_ACTIVATED) does not re-run per stage, so set the
# three variables its rerun branch would have set, explicitly, and clear them afterwards -- grading
# must not depend on what the agent installed into the workspace user base.
AUTORLBENCH_STAGE=rerun
PYTHONUSERBASE=/workspace/.autorlbench-user
PIP_USER=1
PIP_BREAK_SYSTEM_PACKAGES=1
export AUTORLBENCH_STAGE PYTHONUSERBASE PIP_USER PIP_BREAK_SYSTEM_PACKAGES
( cd /workspace && {command_line} )
_rerun_status=$?
unset AUTORLBENCH_STAGE PYTHONUSERBASE PIP_USER PIP_BREAK_SYSTEM_PACKAGES
export AUTORLBENCH_STAGE=verifier
echo "autorlbench: rerun exit=$_rerun_status" >&2
# Grade even when the rerun failed: against empty or partial logs the grading script scores 0 by
# itself, which matches the docker backend, where the verifier always runs.

'''
        return f'''#!/usr/bin/env bash
set +e

# Harbor runs commands with `bash -c`, not a login shell, so /etc/profile.d does not take effect;
# variables prepare.sh wrote to $AUTORLBENCH_PREPARED/env are exported by this line.
if [ -f /etc/profile.d/autorlbench-activate.sh ]; then
  . /etc/profile.d/autorlbench-activate.sh
fi

# --- AutoRLBench report gate: no report.md, no rerun, no grading ----------
if [ "$AUTORLBENCH_EVALUATION_LEVEL" = "solution" ] && [ ! -s "$AUTORLBENCH_WORKSPACE/{REPORT_FILENAME}" ]; then
  echo "autorlbench: agent did not submit {REPORT_FILENAME} in the workspace; skipping rerun and verifier" >&2
  echo '{{"passed": false, "score": 0.0, "metrics": {{"failure": "agent did not submit {REPORT_FILENAME}"}}}}' > "$AUTORLBENCH_RESULTS/result.json"
  cp "$AUTORLBENCH_RESULTS/result.json" /logs/verifier/autorlbench-result.json
  printf '0\\n' > /logs/verifier/reward.txt
  exit 1
fi
if [ "$AUTORLBENCH_EVALUATION_LEVEL" = "solution" ]; then
  # Archive the report into results, same as the docker backend: the workspace is collected after the
  # run, and reviewing "what did the agent actually change" needs something to point at.
  cp "$AUTORLBENCH_WORKSPACE/{REPORT_FILENAME}" "$AUTORLBENCH_RESULTS/{REPORT_FILENAME}" 2>/dev/null || true
fi
{rerun_section}bash "$AUTORLBENCH_VERIFIER/test.sh"
verifier_status=$?

# Put the structured result in the verifier log so remote Harbor environments download it too.
result_file="$AUTORLBENCH_RESULTS/result.json"
if [ -f "$result_file" ]; then
  cp "$result_file" /logs/verifier/autorlbench-result.json
fi

# Harbor only consumes the reward file; AutoRLBench's full metrics still live in result.json.
score=""
if [ -f "$result_file" ]; then
  score="$(tr '\\n' ' ' < "$result_file" | sed -nE 's/.*"score"[[:space:]]*:[[:space:]]*(-?([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][+-]?[0-9]+)?).*/\\1/p')"
fi
if ! printf '%s' "$score" | grep -Eq '^-?([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][+-]?[0-9]+)?$'; then
  if [ "$verifier_status" -eq 0 ]; then
    score="1"
  else
    score="0"
  fi
fi

printf '%s\\n' "$score" > /logs/verifier/reward.txt
exit "$verifier_status"
'''

    @staticmethod
    def _docker_image(package: TaskPackage) -> str:
        if package.environment.image is None:
            raise HarborAdapterError(
                f"task {package.task_id!r} declares no environment.image; Harbor needs "
                "a prebuilt image"
            )
        return EnvironmentImage(package.environment.image).tag

    # Offline switches: same as SandboxRuntime._network_policy_env gives agent / verifier.
    # Only prepare has network, and it finishes before `harbor run`.
    _OFFLINE_ENV = {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "WANDB_MODE": "offline",
        "WANDB_API_KEY": "autorlbench-offline",
        "WANDB_SILENT": "true",
    }

    # /workspace is collected only for archiving and workspace_after_hash; grading in the shared container reads
    # the real /workspace. Weights and training checkpoints are often tens to hundreds of GB, and a download with
    # excludes is a single `tar czf` in the container (Harbor allows 120s), so filter by suffix.
    _WORKSPACE_ARTIFACT_EXCLUDE = (
        "*.safetensors",
        "*.bin",
        "*.pt",
        "*.pth",
        "*.ckpt",
        "*.distcp",
        "*.gguf",
        "__pycache__",
        # Retrieval corpora that prepare.sh stages into /workspace (e.g. search_r1
        # lands ~74GB of wiki-18 plus a large arrow cache). Collecting these as
        # artifacts pushes the 120s tar budget past its limit and discards the
        # trial's score, so exclude them by suffix / directory name.
        "*.arrow",
        ".hf_datasets_cache",
        "wiki-18",
        "*.index",
        "*.faiss",
    )

    def _write_task_config(
        self,
        unit: HarborTaskUnit,
        task_dir: Path,
        config: HarborEvaluationConfig,
    ) -> None:
        package, evaluation = unit.package, unit.evaluation
        oracle_path = "/tests/oracle.md" if evaluation.oracle.is_file() else "/tests/oracle"
        task_name = harbor_task_name(package.task_id, evaluation.level)
        title = str(package.metadata.get("title") or package.task_id)
        docker_image = self._docker_image(package)

        # Shared by both stages: mount paths, offline switches, and BASH_ENV so `bash -c` also loads
        # the activate hook (Harbor does not start a login shell).
        shared_env = {
            "AUTORLBENCH_TASK_ID": package.task_id,
            "AUTORLBENCH_EVALUATION_LEVEL": evaluation.level,
            "AUTORLBENCH_WORKSPACE": "/workspace",
            "AUTORLBENCH_RESULTS": "/logs/artifacts",
            "AUTORLBENCH_VISIBLE_TASK": "/autorlbench",
            "AUTORLBENCH_PREPARED": PREPARED_DIR,
            "AUTORLBENCH_MODELS": MODELS_DIR,
            "AUTORLBENCH_DATASETS": DATASETS_DIR,
            "AUTORLBENCH_ENVIRONMENT": (
                AUTORLBENCH_ENVIRONMENT_DIR if package.paths.environment.is_dir() else ""
            ),
            "BASH_ENV": ACTIVATE_HOOK,
            **self._OFFLINE_ENV,
        }
        agent_env = {
            **shared_env,
            "AUTORLBENCH_TASK_ROOT": "/autorlbench",
            "AUTORLBENCH_STAGE": "agent",
        }
        verifier_env = {
            **shared_env,
            "AUTORLBENCH_TASK_ROOT": "/tests",
            "AUTORLBENCH_ORACLE": oracle_path,
            "AUTORLBENCH_VERIFIER": "/tests/verifier",
            "AUTORLBENCH_STAGE": "verifier",
            "AUTORLBENCH_JUDGE_PROVIDER": "openai-compatible",
            "OPENAI_BASE_URL": "${OPENAI_BASE_URL:-https://api.openai.com/v1}",
            "OPENAI_API_KEY": "${OPENAI_API_KEY:-}",
        }
        if config.judge_model:
            verifier_env["AUTORLBENCH_JUDGE_MODEL"] = config.judge_model

        # Task-declared env (CUDA_VISIBLE_DEVICES and the like) goes to both stages. AUTORLBENCH_*
        # cannot get in; EnvironmentPlan already filtered it.
        for key, value in package.environment.env.items():
            agent_env[key] = value
            verifier_env[key] = value

        artifact_exclude = ", ".join(
            self._toml_string(pattern) for pattern in self._WORKSPACE_ARTIFACT_EXCLUDE
        )
        task_toml = f'''schema_version = "1.4"

artifacts = [{{ source = "/workspace", exclude = [{artifact_exclude}] }}]

[task]
name = {self._toml_string(task_name)}
description = {self._toml_string(title)}

[metadata]
autorlbench_task_id = {self._toml_string(package.task_id)}
autorlbench_evaluation_level = {self._toml_string(evaluation.level)}

[agent]
network_mode = {self._toml_string(config.agent_network)}

[verifier]
environment_mode = "shared"
network_mode = {self._toml_string(config.verifier_network)}

[environment]
docker_image = {self._toml_string(docker_image)}
workdir = "/workspace"
build_timeout_sec = {self._environment_start_timeout(task_dir / "environment")}

[environment.env]
{self._toml_env(agent_env)}

[verifier.env]
{self._toml_env(verifier_env)}
'''
        write_text(task_dir / "task.toml", task_toml)

    # Harbor's default; prebuilt images skip the build, so this time goes almost entirely to uploading workspace.
    _ENVIRONMENT_START_BASE_SEC = 600
    # Conservative `docker compose cp` throughput: 60s per GB.
    _ENVIRONMENT_START_SEC_PER_GB = 60

    @classmethod
    def _environment_start_timeout(cls, environment_dir: Path) -> int:
        """In Harbor, build_timeout_sec covers the whole environment startup, including the environment/ upload.

        Tasks where prepare downloads weights into workspace (slime_retool, about 16GB) would time out under the
        default 600s before the agent even starts, so it is relaxed according to the actual upload size.
        """
        size = sum(
            path.stat().st_size
            for path in environment_dir.rglob("*")
            if path.is_file() and not path.is_symlink()
        )
        gigabytes = -(-size // (1 << 30))
        return cls._ENVIRONMENT_START_BASE_SEC + cls._ENVIRONMENT_START_SEC_PER_GB * gigabytes

    @classmethod
    def _toml_env(cls, env: Dict[str, str]) -> str:
        return "\n".join(
            f"{key} = {cls._toml_string(value)}" for key, value in env.items()
        )

    @staticmethod
    def _toml_string(value: str) -> str:
        return json.dumps(value, ensure_ascii=False)

    @staticmethod
    def _write_manifest(
        package: TaskPackage,
        evaluation: EvaluationPaths,
        task_dir: Path,
        config: HarborEvaluationConfig,
    ) -> None:
        payload = {
            "adapter": "autorlbench-harbor",
            "task_id": package.task_id,
            "evaluation_level": evaluation.level,
            "verifier_mode": "shared",
            "docker_image": HarborTaskMaterializer._docker_image(package),
            "source_task": str(package.paths.root),
        }
        write_text(
            task_dir / "adapter-manifest.json",
            json.dumps(payload, indent=2, ensure_ascii=False),
        )


class HarborRunner:
    """Launch Harbor via its CLI to avoid coupling to Harbor's Python version and internal API."""

    _TASK_PATH_ARGS = {"-p", "--path"}
    # AutoRLBench decides mounts from prepare outputs, while Harbor's --mounts applies to the whole job,
    # so each unit gets its own job; a user-supplied copy would only override the prepared mount.
    _MOUNT_ARGS = {"--mounts", "--mounts-json"}

    def run(
        self,
        prepared: PreparedHarborTask,
        config: HarborEvaluationConfig,
        job_name_suffix: Optional[str] = None,
    ) -> HarborRun:
        harbor_args = list(config.harbor_args)
        self._validate_harbor_args(harbor_args)
        mounts = self._mounts(prepared, config)

        job_name = self._option_value(harbor_args, "--job-name")
        jobs_dir_value = self._option_value(harbor_args, "-o", "--jobs-dir")
        config_path = self._option_value(harbor_args, "-c", "--config")
        injected_args = []
        # Without a job config, fill in the result location; with one, fully respect Harbor's config.
        if job_name is not None and job_name_suffix:
            # In a batch, the user's job name must be distinguished per unit, or the second job collides with the first.
            job_name = f"{job_name}-{job_name_suffix}"
            harbor_args = self._replace_option(harbor_args, "--job-name", job_name)
        if config_path is None:
            if job_name is None:
                job_name = self._job_name()
                injected_args.extend(["--job-name", job_name])
            if jobs_dir_value is None:
                jobs_dir_value = "jobs"
                injected_args.extend(["--jobs-dir", jobs_dir_value])

        caller_cwd = Path.cwd().resolve()
        argv = [
            *config.harbor_command,
            "run",
            "--path",
            str(prepared.tasks_dir),
            *injected_args,
            "--mounts-json",
            json.dumps(mounts),
            *self._compose_overlay_args(prepared),
            *harbor_args,
        ]

        started = time.monotonic()
        display_argv = shlex.join(self._redact_argv(argv))
        print(f"[autorlbench] Starting Harbor: {display_argv}", file=sys.stderr, flush=True)
        try:
            completed = self._run_streaming(argv, caller_cwd)
            command = CommandResult(
                argv=self._redact_argv(argv),
                returncode=completed.returncode,
                stdout=completed.stdout,
                stderr=completed.stderr,
                timed_out=False,
                duration_s=time.monotonic() - started,
                cwd=str(caller_cwd),
            )
        except FileNotFoundError as exc:
            executable = config.harbor_command[0]
            raise HarborAdapterError(f"Harbor command not found: {executable}") from exc

        job_dir = self._reported_job_dir(completed.stdout, caller_cwd)
        if job_dir is None and jobs_dir_value is not None and job_name is not None:
            jobs_dir = Path(jobs_dir_value).expanduser()
            if not jobs_dir.is_absolute():
                jobs_dir = caller_cwd / jobs_dir
            job_dir = jobs_dir.resolve() / job_name
        if job_dir is None:
            raise HarborAdapterError(
                "Harbor did not report its result path; pass --jobs-dir and "
                "--job-name explicitly or use a Harbor version that prints it"
            )

        return HarborRun(command=command, job_dir=job_dir)

    @staticmethod
    def _run_streaming(
        argv: Sequence[str],
        cwd: Path,
    ) -> subprocess.CompletedProcess:
        """Stream Harbor output live while keeping a copy for the evaluation report and locating results."""
        with subprocess.Popen(
            argv,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        ) as process:
            stdout_chunks: list[str] = []
            stderr_chunks: list[str] = []

            def forward(source, target, chunks: list[str]) -> None:
                for line in iter(source.readline, ""):
                    chunks.append(line)
                    target.write(line)
                    target.flush()

            assert process.stdout is not None
            assert process.stderr is not None
            threads = (
                threading.Thread(
                    target=forward,
                    args=(process.stdout, sys.stdout, stdout_chunks),
                    daemon=True,
                ),
                threading.Thread(
                    target=forward,
                    args=(process.stderr, sys.stderr, stderr_chunks),
                    daemon=True,
                ),
            )
            for thread in threads:
                thread.start()
            returncode = process.wait()
            for thread in threads:
                thread.join()

        return subprocess.CompletedProcess(
            args=list(argv),
            returncode=returncode,
            stdout="".join(stdout_chunks),
            stderr="".join(stderr_chunks),
        )

    @classmethod
    def _validate_harbor_args(cls, args: Sequence[str]) -> None:
        """--path must point to the generated tasks; other arguments keep their Harbor meaning."""
        for arg in args:
            name = arg.split("=", 1)[0]
            if name in cls._TASK_PATH_ARGS:
                raise ValueError(
                    f"{name} is managed by the AutoRLBench task materializer; "
                    "pass the AutoRLBench task as task_root"
                )
            if name in cls._MOUNT_ARGS:
                raise ValueError(
                    f"{name} is managed by AutoRLBench (prepared / models / datasets "
                    "mounts); use --models-dir / --datasets-dir instead"
                )

    @staticmethod
    def _mounts(
        prepared: PreparedHarborTask,
        config: HarborEvaluationConfig,
    ) -> list[Dict[str, Any]]:
        """Prepare outputs, the task environment/, and both download caches, all read-only.

        Matches the mounts agent / verifier see in docker evaluation; the only writable one is the
        /workspace Harbor uploads.
        """
        if len(prepared.units) != 1:
            raise HarborAdapterError(
                "Harbor mounts apply to the whole job; run one AutoRLBench task per job"
            )
        unit = prepared.units[0]
        pairs = []
        if unit.prepared is not None:
            pairs.append((unit.prepared, PREPARED_DIR))
        if unit.package.paths.environment.is_dir():
            pairs.append((unit.package.paths.environment, AUTORLBENCH_ENVIRONMENT_DIR))
        pairs.append((config.models_dir, MODELS_DIR))
        pairs.append((config.datasets_dir, DATASETS_DIR))
        return [
            {
                "type": "bind",
                "source": str(Path(source).resolve()),
                "target": target,
                "read_only": True,
            }
            for source, target in pairs
        ]

    @staticmethod
    def _compose_overlay_args(prepared: PreparedHarborTask) -> list[str]:
        """Pass the task's environment/docker-compose.yaml to Harbor as a compose overlay.

        GPU tasks get device reservations and shm_size from it: Harbor's docker environment declaration does not
        support GPUs, `gpus` in task.toml is rejected, and only a compose override on the `main` service
        takes effect. It cannot go in the generated task's environment/ (with a compose file there, Harbor
        stops uploading workspace), so it goes through `--extra-docker-compose`, one per job.
        """
        overlays = []
        for unit in prepared.units:
            compose = unit.package.paths.environment / "docker-compose.yaml"
            if compose.is_file():
                overlays.extend(["--extra-docker-compose", str(compose.resolve())])
        return overlays

    @staticmethod
    def _replace_option(args: Sequence[str], name: str, value: str) -> list[str]:
        result = []
        skip = False
        for index, arg in enumerate(args):
            if skip:
                skip = False
                continue
            if arg == name:
                result.extend([name, value])
                skip = True
            elif arg.startswith(name + "="):
                result.append(f"{name}={value}")
            else:
                result.append(arg)
        return result

    @staticmethod
    def _option_value(args: Sequence[str], *names: str) -> Optional[str]:
        value = None
        for index, arg in enumerate(args):
            for name in names:
                if arg == name:
                    if index + 1 >= len(args):
                        raise ValueError(f"missing value for Harbor argument {name}")
                    value = args[index + 1]
                elif arg.startswith(name + "="):
                    value = arg.split("=", 1)[1]
        return value

    @staticmethod
    def _redact_argv(argv: Sequence[str]) -> list[str]:
        sensitive = re.compile(
            r"(?i)([A-Z0-9_]*(?:KEY|TOKEN|SECRET|PASSWORD)[A-Z0-9_]*)=([^\s]+)"
        )
        return [sensitive.sub(r"\1=<redacted>", arg) for arg in argv]

    @staticmethod
    def _reported_job_dir(stdout: str, cwd: Path) -> Optional[Path]:
        """Read the actual result directory from Harbor's stable completion message."""
        for line in reversed(stdout.splitlines()):
            match = re.search(r"Results written to (.+?result\.json)\s*$", line)
            if match is None:
                continue
            result_path = Path(match.group(1)).expanduser()
            if not result_path.is_absolute():
                result_path = cwd / result_path
            return result_path.resolve().parent
        return None

    @staticmethod
    def _job_name() -> str:
        timestamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
        return f"autorlbench-{timestamp}-{uuid.uuid4().hex[:8]}"


class HarborResultMapper:
    """Read one Harbor job and convert all of its trial results."""

    def map(
        self,
        prepared: PreparedHarborTask,
        run: HarborRun,
        config: HarborEvaluationConfig,
    ) -> HarborEvaluationReport:
        if not (run.job_dir / "result.json").is_file() and run.command.returncode != 0:
            raise HarborAdapterError(
                "Harbor command failed with exit code "
                f"{run.command.returncode}; no job result was written to {run.job_dir}"
            )
        job_result = self._read_json(run.job_dir / "result.json", "Harbor job result")
        trials = self._trial_results(run.job_dir, job_result)
        units = prepared.units_by_task_name

        results = []
        for trial in trials:
            if not isinstance(trial, dict):
                raise HarborAdapterError("Harbor trial result must be a JSON object")
            unit = self._unit_for_trial(trial, units, prepared.units)
            results.append(self._map_trial(unit, run, config, trial))

        report = HarborEvaluationReport(
            task_ids=[unit.package.task_id for unit in prepared.units],
            evaluation_level=config.evaluation_level,
            trials=results,
            report_dir=run.job_dir,
            command=run.command,
        )
        write_text(
            run.job_dir / "autorlbench-report.json",
            json.dumps(report.to_dict(), indent=2, ensure_ascii=False),
        )
        return report

    @staticmethod
    def _unit_for_trial(
        trial: Dict[str, Any],
        units: Dict[str, HarborTaskUnit],
        all_units: Sequence[HarborTaskUnit],
    ) -> HarborTaskUnit:
        """Map a trial back to its AutoRLBench task by Harbor's task_name."""
        task_name = trial.get("task_name")
        if isinstance(task_name, str) and task_name in units:
            return units[task_name]
        # In a single-task batch a missing task_name can still be attributed unambiguously; older Harbor versions rely on this.
        if task_name is None and len(all_units) == 1:
            return all_units[0]
        raise HarborAdapterError(
            f"Harbor trial task_name {task_name!r} does not match any "
            f"materialized task: {', '.join(sorted(units))}"
        )

    def _trial_results(
        self,
        job_dir: Path,
        job_result: Dict[str, Any],
    ) -> list[Dict[str, Any]]:
        """Current Harbor writes full results in each trial directory; the job result keeps only the summary."""
        trials = []
        try:
            children = sorted(job_dir.iterdir(), key=lambda path: path.name)
        except OSError as exc:
            raise HarborAdapterError(f"failed to inspect Harbor job: {job_dir}") from exc

        for child in children:
            result_path = child / "result.json"
            if not child.is_dir() or not result_path.is_file():
                continue
            trial = self._read_json(result_path, "Harbor trial result")
            if trial.get("trial_name") != child.name:
                raise HarborAdapterError(
                    f"Harbor trial_name does not match its directory: {child}"
                )
            trials.append(trial)
        if trials:
            return trials

        # Support older Harbor versions that still embed trial_results in job/result.json.
        embedded = job_result.get("trial_results")
        if isinstance(embedded, list) and embedded:
            return embedded
        raise HarborAdapterError("Harbor job must contain at least one trial result")

    def _map_trial(
        self,
        unit: HarborTaskUnit,
        run: HarborRun,
        config: HarborEvaluationConfig,
        trial: Dict[str, Any],
    ) -> EvaluationResult:
        package = unit.package
        evaluation = unit.evaluation
        trial_dir = self._trial_dir(run.job_dir, trial.get("trial_name"))

        artifact_error = self._workspace_artifact_error(trial_dir)
        autorlbench_result = self._read_autorlbench_result(trial_dir)
        rewards = self._rewards(trial)
        metrics = self._metrics(autorlbench_result, rewards, trial, config)

        infrastructure_errors = []
        if run.command.returncode != 0 or run.command.timed_out:
            infrastructure_errors.append(
                f"Harbor command failed with exit code {run.command.returncode}"
            )
        if trial.get("exception_info"):
            exception = trial["exception_info"]
            message = (
                exception.get("exception_message")
                if isinstance(exception, dict)
                else exception
            )
            infrastructure_errors.append(f"Harbor trial exception: {message}")
        if artifact_error:
            infrastructure_errors.append(artifact_error)
        if infrastructure_errors:
            metrics["infrastructure_errors"] = infrastructure_errors

        score = self._score(metrics)
        passed = not infrastructure_errors and self._passed(metrics, score)
        workspace_dir = trial_dir / "artifacts" / "workspace"
        workspace_after_hash = tree_digest(workspace_dir) if workspace_dir.is_dir() else ""

        result = EvaluationResult(
            task_id=package.task_id,
            evaluation_level=evaluation.level,
            passed=passed,
            score=score if not infrastructure_errors else 0.0,
            metrics=metrics,
            report_dir=trial_dir,
            # Agent/model/backend and similar info follow Harbor's native command and job config.
            agent_spec=None,
            agent_setup=None,
            agent=None,
            verifier=None,
            workspace_before_hash=unit.workspace_before_hash,
            workspace_after_hash=workspace_after_hash,
            metadata=package.metadata,
        )
        self._write_adapter_outputs(trial_dir, run.command, result)
        return result

    @staticmethod
    def _trial_dir(job_dir: Path, trial_name: Any) -> Path:
        if not isinstance(trial_name, str) or not trial_name:
            raise HarborAdapterError("Harbor trial result is missing trial_name")
        trial_dir = (job_dir / trial_name).resolve()
        try:
            trial_dir.relative_to(job_dir.resolve())
        except ValueError as exc:
            raise HarborAdapterError("Harbor trial_name escapes its job directory") from exc
        if not trial_dir.is_dir():
            raise HarborAdapterError(f"Harbor trial directory not found: {trial_dir}")
        return trial_dir

    def _workspace_artifact_error(self, trial_dir: Path) -> Optional[str]:
        manifest_path = trial_dir / "artifacts" / "manifest.json"
        if not manifest_path.is_file():
            return "Harbor artifact manifest is missing"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return f"Harbor artifact manifest is invalid: {exc}"
        entries = manifest.get("entries") if isinstance(manifest, dict) else manifest
        if not isinstance(entries, list):
            return "Harbor artifact manifest must contain a list"
        workspace_entries = [
            entry
            for entry in entries
            if isinstance(entry, dict) and entry.get("source") == "/workspace"
        ]
        if not workspace_entries:
            return "Harbor did not record the /workspace artifact"
        failed = [
            entry.get("status")
            for entry in workspace_entries
            if entry.get("status") not in {"ok", "empty"}
        ]
        if failed:
            return f"Harbor failed to collect /workspace artifact: {', '.join(map(str, failed))}"
        if not (trial_dir / "artifacts" / "workspace").is_dir():
            return "Harbor /workspace artifact directory is missing"
        return None

    def _read_autorlbench_result(self, trial_dir: Path) -> Dict[str, Any]:
        candidates = (
            trial_dir / "verifier" / "autorlbench-result.json",
            trial_dir / "artifacts" / "logs" / "artifacts" / "result.json",
        )
        for path in candidates:
            if not path.is_file():
                continue
            payload = self._read_json(path, "AutoRLBench verifier result")
            if isinstance(payload, dict):
                return payload
        return {}

    @staticmethod
    def _rewards(trial: Dict[str, Any]) -> Dict[str, float]:
        verifier_result = trial.get("verifier_result")
        rewards = verifier_result.get("rewards") if isinstance(verifier_result, dict) else None
        if not isinstance(rewards, dict):
            return {}
        return {
            str(key): float(value)
            for key, value in rewards.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }

    @staticmethod
    def _metrics(
        autorlbench_result: Dict[str, Any],
        rewards: Dict[str, float],
        trial: Dict[str, Any],
        config: HarborEvaluationConfig,
    ) -> Dict[str, Any]:
        metrics: Dict[str, Any] = {}
        nested = autorlbench_result.get("metrics")
        if isinstance(nested, dict):
            metrics.update(nested)
        for key in ("passed", "score"):
            if key in autorlbench_result:
                metrics[key] = autorlbench_result[key]
        for key, value in autorlbench_result.items():
            if key not in {"metrics", "passed", "score"} and key not in metrics:
                metrics[key] = value
        metrics["harbor_rewards"] = rewards
        metrics["harbor_verifier_mode"] = (
            trial.get("verifier_environment_mode") or "shared"
        )
        return metrics

    @staticmethod
    def _score(metrics: Dict[str, Any]) -> Optional[float]:
        score = metrics.get("score")
        if isinstance(score, (int, float)) and not isinstance(score, bool):
            return float(score)
        rewards = metrics.get("harbor_rewards")
        if isinstance(rewards, dict):
            reward = rewards.get("reward")
            if isinstance(reward, (int, float)) and not isinstance(reward, bool):
                return float(reward)
        if metrics.get("passed") is True:
            return 1.0
        if metrics.get("passed") is False:
            return 0.0
        return None

    @staticmethod
    def _passed(metrics: Dict[str, Any], score: Optional[float]) -> bool:
        if isinstance(metrics.get("passed"), bool):
            return metrics["passed"]
        return score == 1.0

    @staticmethod
    def _read_json(path: Path, label: str) -> Dict[str, Any]:
        if not path.is_file():
            raise HarborAdapterError(f"{label} not found: {path}")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise HarborAdapterError(f"failed to read {label}: {path}") from exc
        if not isinstance(payload, dict):
            raise HarborAdapterError(f"{label} must be a JSON object: {path}")
        return payload

    @staticmethod
    def _write_adapter_outputs(
        trial_dir: Path,
        command: CommandResult,
        result: EvaluationResult,
    ) -> None:
        write_text(
            trial_dir / "autorlbench-harbor-command.json",
            json.dumps(command.to_dict(), indent=2, ensure_ascii=False),
        )
        write_text(
            trial_dir / "report.json",
            json.dumps(result.to_dict(), indent=2, ensure_ascii=False),
        )


class HarborPreparer:
    """Run the task's prepare.sh in the same image before `harbor run`.

    Prepare is the only stage with network: it downloads models and datasets and writes prepared/env. It stays out of Harbor:
    Harbor's agent / verifier are offline, and a job has no slot for "run a setup step first".
    It reuses TaskEvaluator's staging and prepare logic, identical to docker evaluation.
    """

    def __init__(self, evaluator: Optional[TaskEvaluator] = None) -> None:
        self._evaluator = evaluator

    @property
    def evaluator(self) -> TaskEvaluator:
        if self._evaluator is None:
            self._evaluator = TaskEvaluator("docker")
        return self._evaluator

    def prepare(
        self,
        unit: HarborTaskUnit,
        work_dir: Path,
        config: HarborEvaluationConfig,
    ) -> HarborTaskUnit:
        package = unit.package
        image = package.environment.environment_image
        if image is None:
            raise HarborAdapterError(
                f"task {package.task_id!r} declares no environment.image; Harbor needs "
                "a prebuilt image"
            )
        evaluator = self.evaluator
        eval_config = EvaluationConfig(
            work_dir=work_dir,
            evaluation_level=unit.evaluation.level,
            huggingface_mirror=config.huggingface_mirror,
            models_dir=config.models_dir,
            datasets_dir=config.datasets_dir,
            keep_artifacts=True,
        )
        run = evaluator._prepare_run(package, unit.evaluation, eval_config)
        # Images are shared per environment: reuse if present, build only if missing. Harbor uses the same tag.
        artifact = evaluator.runtime.ensure_environment(image)
        print(
            f"[autorlbench] Preparing {package.task_id} in {artifact.reference or image.tag}",
            file=sys.stderr,
            flush=True,
        )
        result = evaluator._run_prepare(package, unit.evaluation, run, eval_config, artifact)
        if result is not None and result.returncode != 0:
            tail = "\n".join(result.stderr.strip().splitlines()[-20:])
            raise HarborAdapterError(
                f"task {package.task_id!r}: prepare failed with exit {result.returncode}"
                + (f"\n{tail}" if tail else "")
            )
        return dataclasses.replace(unit, workspace=run.workspace, prepared=run.prepared)


class HarborShell:
    """prepare, then launch an interactive shell via docker run -it.

    Sets up exactly the same environment as the agent phase of harbor-evaluate, but hands over an
    interactive shell instead of starting Harbor and the agent. Stages with HarborPreparer
    and converts the same mounts and env as HarborRunner._mounts() and HarborTaskMaterializer._write_task_config()
    into docker run arguments.

    The workspace staging result is recorded in state_file and reused on the next launch
    without --force-prepare, so a 16GB+ prepare does not run every time.
    """

    _STATE_FILE = "autorlbench-shell.json"

    def __init__(self, preparer: Optional[HarborPreparer] = None) -> None:
        self.preparer = preparer or HarborPreparer()

    def shell(
        self,
        task_root: str | Path,
        config: HarborEvaluationConfig,
        workspace_dir: Path,
        *,
        shell_cmd: str = "bash",
        force_prepare: bool = False,
        gpus: Optional[str] = None,
        shm_size: Optional[str] = None,
        extra_docker_args: Sequence[str] = (),
    ) -> None:
        """prepare and exec into docker run -it. Does not return, since os.execvp replaces the process."""
        package = load_task_package(Path(task_root))
        if not package.report.ok:
            raise SchemaError(f"task package failed validation: {package.paths.root}")
        try:
            evaluation = package.paths.evaluations[config.evaluation_level]
        except KeyError:
            available = ", ".join(package.paths.evaluations) or "none"
            raise SchemaError(
                f"evaluation level {config.evaluation_level!r} is unavailable; "
                f"available levels: {available}"
            )
        unit = HarborTaskUnit(
            package=package,
            evaluation=evaluation,
            workspace_before_hash="",
        )

        workspace_dir = workspace_dir.resolve()
        state_path = workspace_dir / self._STATE_FILE

        if not force_prepare and state_path.is_file():
            state = json.loads(state_path.read_text(encoding="utf-8"))
            unit = dataclasses.replace(
                unit,
                workspace=Path(state["workspace"]),
                prepared=Path(state["prepared"]) if state.get("prepared") else None,
            )
            print(
                f"[autorlbench] Reusing staged workspace at {unit.workspace}",
                file=sys.stderr,
                flush=True,
            )
        else:
            ensure_dir(workspace_dir)
            unit = self.preparer.prepare(unit, workspace_dir, config)
            state_data: Dict[str, Any] = {
                "task_id": package.task_id,
                "workspace": str(unit.workspace),
                "prepared": str(unit.prepared) if unit.prepared else None,
            }
            write_text(state_path, json.dumps(state_data, indent=2, ensure_ascii=False))

        image = HarborTaskMaterializer._docker_image(package)
        argv = self._build_docker_argv(
            unit,
            package,
            config,
            image,
            shell_cmd,
            gpus=gpus,
            shm_size=shm_size,
            extra_docker_args=extra_docker_args,
        )
        print(f"[autorlbench] Entering container: {shlex.join(argv)}", file=sys.stderr, flush=True)
        os.execvp(argv[0], argv)

    def _build_docker_argv(
        self,
        unit: HarborTaskUnit,
        package: TaskPackage,
        config: HarborEvaluationConfig,
        image: str,
        shell_cmd: str,
        *,
        gpus: Optional[str],
        shm_size: Optional[str],
        extra_docker_args: Sequence[str],
    ) -> list[str]:
        argv = ["docker", "run", "--rm", "-it"]

        # GPU reservation and shm_size follow the values from the docker-compose.yaml overlay.
        # Explicit arguments win; otherwise decide based on whether an overlay exists.
        overlay = package.paths.environment / "docker-compose.yaml"
        has_overlay = overlay.is_file()
        effective_gpus = gpus if gpus is not None else ("all" if has_overlay else None)
        effective_shm = shm_size if shm_size is not None else ("16g" if has_overlay else None)
        if effective_gpus:
            argv.extend(["--gpus", effective_gpus])
        if effective_shm:
            argv.extend(["--shm-size", effective_shm])

        # workspace is writable; everything else is read-only, same policy as harbor-evaluate.
        assert unit.workspace is not None
        argv.extend(["-v", f"{unit.workspace}:/workspace"])
        if unit.prepared is not None and Path(unit.prepared).is_dir():
            argv.extend(["-v", f"{unit.prepared}:{PREPARED_DIR}:ro"])
        if package.paths.environment.is_dir():
            argv.extend(
                ["-v", f"{package.paths.environment.resolve()}:{AUTORLBENCH_ENVIRONMENT_DIR}:ro"]
            )
        argv.extend(["-v", f"{config.models_dir}:{MODELS_DIR}:ro"])
        argv.extend(["-v", f"{config.datasets_dir}:{DATASETS_DIR}:ro"])

        argv.extend(["-w", "/workspace"])

        # Same setup as agent_env in _write_task_config(). AUTORLBENCH_RESULTS points to a
        # writable location under /workspace (/logs/artifacts does not exist).
        env: Dict[str, str] = {
            "AUTORLBENCH_TASK_ID": package.task_id,
            "AUTORLBENCH_EVALUATION_LEVEL": unit.evaluation.level,
            "AUTORLBENCH_WORKSPACE": "/workspace",
            "AUTORLBENCH_RESULTS": "/workspace/autorlbench-results",
            "AUTORLBENCH_TASK_ROOT": "/workspace",
            "AUTORLBENCH_STAGE": "agent",
            "AUTORLBENCH_PREPARED": PREPARED_DIR,
            "AUTORLBENCH_MODELS": MODELS_DIR,
            "AUTORLBENCH_DATASETS": DATASETS_DIR,
            "AUTORLBENCH_ENVIRONMENT": (
                AUTORLBENCH_ENVIRONMENT_DIR if package.paths.environment.is_dir() else ""
            ),
            "BASH_ENV": ACTIVATE_HOOK,
            **self._OFFLINE_ENV,
            # Override with task-declared env (CUDA_VISIBLE_DEVICES, etc.).
            **package.environment.env,
        }
        for key, value in env.items():
            argv.extend(["-e", f"{key}={value}"])

        argv.extend(extra_docker_args)
        argv.append(image)
        argv.append(shell_cmd)
        return argv

    # Same set of offline flags as harbor-evaluate.
    _OFFLINE_ENV = HarborTaskMaterializer._OFFLINE_ENV


class HarborEvaluator:
    """AutoRLBench's Harbor execution entry point, alongside TaskEvaluator."""

    def __init__(
        self,
        materializer: Optional[HarborTaskMaterializer] = None,
        runner: Optional[HarborRunner] = None,
        result_mapper: Optional[HarborResultMapper] = None,
        preparer: Optional[HarborPreparer] = None,
    ) -> None:
        self.preparer = preparer or HarborPreparer()
        self.materializer = materializer or HarborTaskMaterializer()
        self.runner = runner or HarborRunner()
        self.result_mapper = result_mapper or HarborResultMapper()

    def load(self, task_root: str | Path, strict: bool = True) -> TaskPackage:
        return load_task_package(Path(task_root), strict=strict)

    def validate(self, task_root: str | Path, strict: bool = True) -> Dict[str, Any]:
        return self.load(task_root, strict=strict).report.to_dict()

    def evaluate(
        self,
        task_roots: str | Path | Sequence[str | Path],
        config: Optional[HarborEvaluationConfig] = None,
        strict: bool = True,
    ) -> HarborEvaluationReport:
        """Per task: prepare, materialize, one Harbor job; finally merge into one report.

        Each task gets its own job because Harbor mounts are job-level, while the prepared directory
        is per task. A task whose prepare fails is just missing from the report and does not hold back the rest.
        """
        config = config or HarborEvaluationConfig()
        units = [
            self._load_unit(task_root, config.evaluation_level, strict)
            for task_root in self._normalize_task_roots(task_roots)
        ]
        # Same-named tasks would overwrite each other's reports even in separate jobs, so reject the whole batch up front.
        self.materializer._reject_duplicate_task_names(units)
        batch = len(units) > 1
        reports: list[HarborEvaluationReport] = []
        errors: list[str] = []
        for unit in units:
            try:
                reports.append(self._evaluate_unit(unit, config, batch))
            except HarborAdapterError as exc:
                if not batch:
                    raise
                errors.append(f"{unit.package.task_id}: {exc}")
                print(f"[autorlbench] {unit.package.task_id} skipped: {exc}", file=sys.stderr)
        if not reports:
            raise HarborAdapterError("; ".join(errors) or "no task was evaluated")
        return self._merge_reports(units, reports, config)

    def _evaluate_unit(
        self,
        unit: HarborTaskUnit,
        config: HarborEvaluationConfig,
        batch: bool,
    ) -> HarborEvaluationReport:
        root = self._make_run_dir([unit], config.temp_dir)
        try:
            unit = self.preparer.prepare(unit, ensure_dir(root / "prep"), config)
            print(
                f"[autorlbench] Materializing {unit.package.task_id}:{unit.evaluation.level}",
                file=sys.stderr,
                flush=True,
            )
            prepared = self.materializer.materialize([unit], root / "harbor", config)
            run = self.runner.run(
                prepared, config, job_name_suffix=unit.dir_name if batch else None
            )
            print(
                f"[autorlbench] Reading Harbor results from {run.job_dir}",
                file=sys.stderr,
                flush=True,
            )
            return self.result_mapper.map(prepared, run, config)
        finally:
            # Harbor jobs stay in the native --jobs-dir; only prepare staging and temporary tasks are cleaned up here.
            shutil.rmtree(root, ignore_errors=True)

    @staticmethod
    def _merge_reports(
        units: Sequence[HarborTaskUnit],
        reports: Sequence[HarborEvaluationReport],
        config: HarborEvaluationConfig,
    ) -> HarborEvaluationReport:
        if len(units) == 1:
            return reports[0]
        last = reports[-1]
        return HarborEvaluationReport(
            task_ids=[unit.package.task_id for unit in units],
            evaluation_level=config.evaluation_level,
            trials=[trial for report in reports for trial in report.trials],
            report_dir=last.report_dir,
            command=last.command,
        )

    @staticmethod
    def _normalize_task_roots(
        task_roots: str | Path | Sequence[str | Path],
    ) -> list[Path]:
        if isinstance(task_roots, (str, Path)):
            return [Path(task_roots)]
        roots = [Path(task_root) for task_root in task_roots]
        if not roots:
            raise HarborAdapterError("at least one task root is required")
        return roots

    def _load_unit(
        self,
        task_root: str | Path,
        evaluation_level: str,
        strict: bool,
    ) -> HarborTaskUnit:
        package = self.load(task_root, strict=strict)
        if not package.report.ok:
            raise SchemaError(f"task package failed validation: {package.paths.root}")
        evaluation = self._select_evaluation(package, evaluation_level)
        return HarborTaskUnit(
            package=package,
            evaluation=evaluation,
            workspace_before_hash=tree_digest(package.paths.workspace),
        )

    @staticmethod
    def _select_evaluation(package: TaskPackage, level: str) -> EvaluationPaths:
        try:
            return package.paths.evaluations[level]
        except KeyError as exc:
            available = ", ".join(package.evaluation_levels) or "none"
            raise SchemaError(
                f"evaluation level {level!r} is unavailable; available levels: {available}"
            ) from exc

    @staticmethod
    def _make_run_dir(
        units: Sequence[HarborTaskUnit],
        temp_dir: Optional[Path] = None,
    ) -> Path:
        if len(units) == 1:
            unit = units[0]
            safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "-", unit.package.task_id).strip("-.")
            prefix = f"autorlbench-harbor-{safe_id or 'task'}-{unit.evaluation.level}-"
        else:
            prefix = f"autorlbench-harbor-batch-{len(units)}-"
        parent = ensure_dir(temp_dir).resolve() if temp_dir is not None else None
        return Path(
            tempfile.mkdtemp(
                prefix=prefix,
                dir=None if parent is None else str(parent),
            )
        ).resolve()


__all__ = [
    "HarborAdapterError",
    "HarborEvaluationConfig",
    "HarborEvaluationReport",
    "HarborEvaluator",
    "HarborPreparer",
    "HarborResultMapper",
    "HarborRun",
    "HarborRunner",
    "HarborShell",
    "HarborTaskMaterializer",
    "HarborTaskUnit",
    "PreparedHarborTask",
]
