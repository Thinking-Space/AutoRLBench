"""Runtime wiring of environment images: what each of the three stages mounts and which env vars it gets.

This file guards access control, not features. The differences between the three mount tables (who can write,
who sees the cache, who sees the oracle) are the entire implementation of "the agent cannot grade itself and
the verifier cannot read tampered data", so each one deserves a test that would fail.
"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from autorlbench.core.types import (
    DATASETS_DIR,
    MODELS_DIR,
    PREPARED_DIR,
    AUTORLBENCH_ENVIRONMENT_DIR,
    EnvironmentPlan,
)
from autorlbench.runtimes import SandboxRequest
from autorlbench.runtimes.providers.docker import DockerRuntime


ROOT = Path("/host/run")


def make_request(stage: str = "agent", **kwargs) -> SandboxRequest:
    """A request with "everything mounted"; individual tests remove a directory as needed."""
    defaults = dict(
        stage=stage,
        task_id="task_001",
        host_task_root=ROOT / "task",
        host_visible_task=ROOT / "visible",
        host_workspace=ROOT / "workspace",
        host_results=ROOT / "results",
        host_environment=ROOT / "environment",
        host_prepared=ROOT / "prepared",
        host_models=ROOT / "models",
        host_datasets=ROOT / "datasets",
    )
    defaults.update(kwargs)
    return SandboxRequest(**defaults)


class MountTableTests(unittest.TestCase):
    def _mounts(self, request) -> dict:
        return {
            mount.container_path: mount
            for mount in DockerRuntime()._mounts(request)
        }

    # --- workspace: who can write ---------------------------------------------

    def test_prepare_can_write_the_workspace(self) -> None:
        # Weights are downloaded to the relative path the task script references (./Qwen2.5-0.5B-Instruct);
        # with a read-only workspace the first command of prepare.sh would fail.
        self.assertFalse(self._mounts(make_request("prepare"))["/workspace"].read_only)

    def test_agent_can_write_the_workspace(self) -> None:
        self.assertFalse(self._mounts(make_request("agent"))["/workspace"].read_only)

    def test_verifier_gets_a_read_only_workspace(self) -> None:
        # Grading must not alter the scene.
        self.assertTrue(self._mounts(make_request("verifier"))["/workspace"].read_only)

    # --- /task: no spoilers ---------------------------------------------------

    def test_prepare_and_agent_only_see_the_visible_task(self) -> None:
        # Prepare runs a script shipped with the task and has no reason to read the oracle or grading scripts.
        for stage in ("prepare", "agent"):
            mount = self._mounts(make_request(stage))["/task"]
            self.assertEqual(mount.host_path, ROOT / "visible", stage)
            self.assertTrue(mount.read_only, stage)

    def test_verifier_sees_the_whole_evaluation_root(self) -> None:
        mount = self._mounts(make_request("verifier"))["/task"]
        self.assertEqual(mount.host_path, ROOT / "task")
        self.assertTrue(mount.read_only)

    # --- prepared: prepare writes, the later two stages read only -------------

    def test_prepare_owns_the_prepared_directory(self) -> None:
        self.assertFalse(self._mounts(make_request("prepare"))[PREPARED_DIR].read_only)

    def test_agent_and_verifier_get_prepared_read_only(self) -> None:
        # The verifier reads the models/datasets prepare delivered, but must not read a copy the agent modified,
        # or the grading basis would be editable by the examinee.
        for stage in ("agent", "verifier"):
            self.assertTrue(self._mounts(make_request(stage))[PREPARED_DIR].read_only, stage)

    def test_prepared_is_absent_when_the_run_has_none(self) -> None:
        self.assertNotIn(PREPARED_DIR, self._mounts(make_request("agent", host_prepared=None)))

    # --- models / datasets: writable by prepare, read-only later --------------

    def test_prepare_can_write_the_download_cache(self) -> None:
        # Downloads happen in prepare; if it cannot write, there is effectively no cache.
        mounts = self._mounts(make_request("prepare"))
        for container_path in (MODELS_DIR, DATASETS_DIR):
            self.assertIn(container_path, mounts)
            self.assertFalse(mounts[container_path].read_only, container_path)

    def test_later_stages_see_the_download_cache_read_only(self) -> None:
        # The verifier reads them read-only for grading; the agent does not see them at all --
        # an HF cache's directory names are its repo names, and a repo name can spell out the
        # answer. See sees_download_cache().
        self.assertNotIn(MODELS_DIR, self._mounts(make_request("agent")))
        self.assertNotIn(DATASETS_DIR, self._mounts(make_request("agent")))
        mounts = self._mounts(make_request("verifier"))
        for container_path in (MODELS_DIR, DATASETS_DIR):
            self.assertIn(container_path, mounts, container_path)
            self.assertTrue(mounts[container_path].read_only, container_path)

    def test_stages_without_a_download_cache_still_run(self) -> None:
        # No shared cache only means "this download will not be reused next time", not an error.
        for stage in ("prepare", "agent", "verifier"):
            mounts = self._mounts(make_request(stage, host_models=None, host_datasets=None))
            self.assertNotIn(MODELS_DIR, mounts, stage)
            self.assertNotIn(DATASETS_DIR, mounts, stage)

    # --- environment: read-only wherever it is mounted -------------------------

    def test_environment_is_read_only_wherever_mounted(self) -> None:
        for stage in ("prepare", "rerun", "verifier"):
            mount = self._mounts(make_request(stage))[AUTORLBENCH_ENVIRONMENT_DIR]
            self.assertEqual(mount.host_path, ROOT / "environment", stage)
            self.assertTrue(mount.read_only, stage)

    def test_the_agent_never_sees_the_environment_dir(self) -> None:
        # prepare.sh describes how the task was assembled -- which HF repo, which sha256 --
        # and any of those names can point straight at the defect being diagnosed.
        # See sees_environment_dir().
        self.assertNotIn(
            AUTORLBENCH_ENVIRONMENT_DIR, self._mounts(make_request("agent"))
        )

    def test_task_without_environment_dir_mounts_nothing(self) -> None:
        # environment/ is optional; if absent, do not mount it rather than mounting a missing directory.
        request = make_request("agent", host_environment=None)
        self.assertNotIn(AUTORLBENCH_ENVIRONMENT_DIR, self._mounts(request))

    # --- results ------------------------------------------------------------

    def test_prepare_has_no_results_mount(self) -> None:
        # Prepare produces nothing for grading; the harness writes its logs on the host side.
        self.assertNotIn("/results", self._mounts(make_request("prepare")))

    def test_no_mount_is_declared_twice(self) -> None:
        for stage in ("prepare", "agent", "verifier"):
            mounts = DockerRuntime()._mounts(make_request(stage))
            targets = [mount.container_path for mount in mounts]
            self.assertEqual(len(targets), len(set(targets)), f"{stage}: {targets}")


class GpuWiringTests(unittest.TestCase):
    """How the GPU count declared on the task card becomes `docker create` flags.

    This is the entire implementation of "can the task actually train". The host's default runtime is runc,
    and passing device ids via `-e` makes no device visible in the container; only --gpus triggers the nvidia
    hook. Docker exposes N cards as cuda:0..N-1 inside the container, so the task declares a count, not
    device names. The image side (CUDA base image) has always assumed libcuda.so.1 is injected by --gpus;
    this supplies the missing half.
    """

    def _args(self, **kwargs) -> list:
        return DockerRuntime._gpu_args(make_request("agent", **kwargs))

    def test_a_declared_count_becomes_the_gpus_flag(self) -> None:
        # `--gpus` accepts a bare number as a count (`docker run --gpus 8`).
        self.assertEqual(self._args(gpus=8), ["--gpus", "8", "--shm-size=16g"])

    def test_gpu_stages_get_a_usable_shm(self) -> None:
        # Docker's default 64MB /dev/shm makes ray/sglang blow up on the first rollout,
        # with an unhelpful bus error.
        args = self._args(gpus=8)
        self.assertIn("--shm-size=16g", args)

    def test_a_cpu_only_task_is_untouched(self) -> None:
        # Byte-for-byte identical to before the wiring: CPU-only tasks and the grading stage have no reason to use more memory.
        self.assertEqual(self._args(), [])

    def test_declaring_zero_gpus_is_the_same_as_declaring_none(self) -> None:
        self.assertEqual(self._args(gpus=0), [])

    def test_the_host_can_override_the_task_card(self) -> None:
        # A machine with only 2 GPUs should still run a task whose card says 8.
        with mock.patch.dict(os.environ, {"AUTORLBENCH_DOCKER_GPUS": "device=0,1"}):
            args = self._args(gpus=8)
        self.assertEqual(args[:2], ["--gpus", "device=0,1"])

    def test_the_host_can_override_the_shm_size(self) -> None:
        with mock.patch.dict(os.environ, {"AUTORLBENCH_DOCKER_SHM_SIZE": "64g"}):
            args = self._args(gpus=8)
        self.assertIn("--shm-size=64g", args)

    def test_prepare_never_takes_the_cards(self) -> None:
        # Prepare only downloads models and datasets; on some hosts the NVIDIA runtime hook
        # fails at `docker start` when it tries to bind the device nodes.
        self.assertEqual(DockerRuntime._gpu_args(make_request("prepare", gpus=8)), [])

    def test_the_training_stages_take_the_cards_the_task_declares(self) -> None:
        # The rerun runs real training; the agent gets them too, but the protocol forbids it
        # from launching training -- there the GPU is headroom for cheap offline checks.
        for stage in ("agent", "rerun", "verifier"):
            request = make_request(stage, gpus=8)
            self.assertIn("--gpus", DockerRuntime._gpu_args(request), stage)


class StageRequestTests(unittest.TestCase):
    def test_prepare_without_a_landing_zone_is_rejected(self) -> None:
        # A prepare with nowhere to write would "succeed" and leave nothing, surfacing only when the agent cannot find the model,
        # and then it is reported as the task's fault.
        with self.assertRaises(ValueError) as ctx:
            SandboxRequest(
                stage="prepare",
                task_id="t",
                host_task_root=ROOT,
                host_visible_task=ROOT,
                host_workspace=ROOT,
                host_results=ROOT,
            )
        self.assertIn("host_prepared", str(ctx.exception))

    def test_unknown_stage_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            make_request("provision")


class StandardEnvTests(unittest.TestCase):
    def _env(self, stage: str = "agent", **kwargs) -> dict:
        return DockerRuntime().standard_env(make_request(stage, **kwargs))

    def test_every_stage_is_told_which_stage_it_is(self) -> None:
        # The activate hook uses AUTORLBENCH_STAGE to decide per-stage differences.
        for stage in ("prepare", "agent", "verifier"):
            env = self._env(stage)
            self.assertEqual(env["AUTORLBENCH_STAGE"], stage, stage)
            self.assertNotIn("AUTORLBENCH_CONDA_ENV", env, stage)

    def test_prepare_and_agent_never_learn_where_the_oracle_is(self) -> None:
        for stage in ("prepare", "agent"):
            env = self._env(stage)
            self.assertEqual(env["AUTORLBENCH_ORACLE"], "", stage)
            self.assertEqual(env["AUTORLBENCH_VERIFIER"], "", stage)
            self.assertEqual(env["AUTORLBENCH_TASK_ROOT"], "/task", stage)

    def test_verifier_gets_the_grading_paths(self) -> None:
        env = self._env("verifier")
        self.assertEqual(env["AUTORLBENCH_ORACLE"], "/task/oracle")
        self.assertEqual(env["AUTORLBENCH_VERIFIER"], "/task/verifier")
        # visible_task means nothing to the verifier: it sees the whole /task.
        self.assertEqual(env["AUTORLBENCH_VISIBLE_TASK"], "")

    def test_paths_match_what_is_actually_mounted(self) -> None:
        env = self._env("prepare")
        self.assertEqual(env["AUTORLBENCH_ENVIRONMENT"], AUTORLBENCH_ENVIRONMENT_DIR)
        self.assertEqual(env["AUTORLBENCH_PREPARED"], PREPARED_DIR)
        self.assertEqual(env["AUTORLBENCH_MODELS"], MODELS_DIR)
        self.assertEqual(env["AUTORLBENCH_DATASETS"], DATASETS_DIR)

    def test_the_download_cache_reaches_prepare_and_verifier_only(self) -> None:
        # Prepare writes the caches, the verifier reads them; the agent and rerun see nothing,
        # matching sees_download_cache() and the mount table.
        for stage in ("prepare", "verifier"):
            env = self._env(stage)
            self.assertEqual(env["AUTORLBENCH_MODELS"], MODELS_DIR, stage)
            self.assertEqual(env["AUTORLBENCH_DATASETS"], DATASETS_DIR, stage)
        for stage in ("agent", "rerun"):
            env = self._env(stage)
            self.assertEqual(env["AUTORLBENCH_MODELS"], "", stage)
            self.assertEqual(env["AUTORLBENCH_DATASETS"], "", stage)

    def test_unmounted_directories_are_the_empty_string(self) -> None:
        # "You cannot see it", not "the path is there but the directory is empty"; the latter would make prepare.sh
        # download the model somewhere that vanishes after the run.
        env = self._env("verifier", host_models=None, host_datasets=None)
        self.assertEqual(env["AUTORLBENCH_MODELS"], "")
        self.assertEqual(env["AUTORLBENCH_DATASETS"], "")
        # The prepare stage always has a prepared dir (the request validation demands
        # one); the rerun is where a missing one is legal.
        env = self._env("rerun", host_environment=None, host_prepared=None)
        self.assertEqual(env["AUTORLBENCH_ENVIRONMENT"], "")
        self.assertEqual(env["AUTORLBENCH_PREPARED"], "")

    def test_the_agent_does_not_see_the_environment_dir(self) -> None:
        # The variable follows the mount table, not whoever set the field, so the two
        # cannot drift: an agent-stage AUTORLBENCH_ENVIRONMENT is always the empty string.
        # The rerun, harness-initiated and running only the task's declared command,
        # keeps the mount like the verifier does.
        self.assertEqual(self._env("agent")["AUTORLBENCH_ENVIRONMENT"], "")
        self.assertEqual(
            self._env("rerun")["AUTORLBENCH_ENVIRONMENT"], AUTORLBENCH_ENVIRONMENT_DIR
        )


class NetworkPolicyEnvTests(unittest.TestCase):
    """Offline switches are set per stage, not baked into the image ENV."""

    def test_prepare_is_allowed_to_reach_the_network(self) -> None:
        # Image-level ENV treats all three stages alike, so prepare would cut off its own network.
        env = DockerRuntime().standard_env(make_request("prepare"))
        for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
            self.assertNotIn(key, env)

    def test_prepare_still_does_not_report_to_wandb(self) -> None:
        # Prepare is only downloading; there is nothing to report.
        env = DockerRuntime().standard_env(make_request("prepare"))
        self.assertEqual(env["WANDB_MODE"], "offline")

    def test_agent_and_verifier_are_pinned_offline(self) -> None:
        # A silent network download could run the evaluation on weights nobody verified, without anyone noticing.
        for stage in ("agent", "verifier"):
            env = DockerRuntime().standard_env(make_request(stage))
            self.assertEqual(env["HF_HUB_OFFLINE"], "1", stage)
            self.assertEqual(env["TRANSFORMERS_OFFLINE"], "1", stage)
            self.assertEqual(env["HF_DATASETS_OFFLINE"], "1", stage)
            self.assertEqual(env["WANDB_MODE"], "offline", stage)
            # Some frameworks still exit immediately in offline mode when no key is set.
            self.assertTrue(env["WANDB_API_KEY"])

    def test_agent_stage_does_not_see_the_task_id(self) -> None:
        # Task ids often spell out the diagnosis (e.g. "context_parallel_oom"); printenv in
        # the sandbox must not hand over the answer. The rerun is harness-initiated and gets
        # the same neutral placeholder; the other stages keep the real id -- no task-side
        # script consumes it, but it is part of prepare.sh's interface.
        for stage in ("agent", "rerun"):
            self.assertEqual(
                DockerRuntime().standard_env(make_request(stage))["AUTORLBENCH_TASK_ID"],
                "task",
                stage,
            )
        for stage in ("prepare", "verifier"):
            env = DockerRuntime().standard_env(make_request(stage))
            self.assertEqual(env["AUTORLBENCH_TASK_ID"], "task_001", stage)


class PrepareScriptTests(unittest.TestCase):
    """How the evaluator translates prepare.sh into a container path."""

    def _package(self, root: Path, *, prepare: Path | None, image: str | None = "verl-qwen3vl-cu130"):
        from autorlbench.core.types import TaskPaths

        paths = TaskPaths(
            root=root,
            readme_md=None,
            task_toml=root / "task.toml",
            workspace=root / "workspace",
            environment=root / "environment",
            evaluations={},
        )
        from autorlbench.core.types import TaskPackage
        from autorlbench.core.types import ValidationReport

        return TaskPackage(
            paths=paths,
            metadata={},
            report=ValidationReport(),
            environment=EnvironmentPlan(image=image, prepare=prepare),
        )

    def test_script_resolves_under_the_mounted_environment(self) -> None:
        from autorlbench.evaluation.evaluator import TaskEvaluator

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "environment").mkdir()
            script = root / "environment" / "prepare.sh"
            script.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
            package = self._package(root, prepare=script)
            self.assertEqual(
                TaskEvaluator("local")._prepare_script(package),
                f"{AUTORLBENCH_ENVIRONMENT_DIR}/prepare.sh",
            )

    def test_no_prepare_script_means_no_prepare_stage(self) -> None:
        from autorlbench.evaluation.evaluator import TaskEvaluator

        with tempfile.TemporaryDirectory() as temp:
            package = self._package(Path(temp), prepare=None)
            self.assertIsNone(TaskEvaluator("local")._prepare_script(package))

    def test_script_outside_the_environment_directory_is_rejected(self) -> None:
        # Only environment/ is mounted into the container; a path outside it would become
        # "file not found" in the container, far from the real cause.
        from autorlbench.evaluation.evaluator import TaskEvaluator
        from autorlbench.tasks import SchemaError

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "environment").mkdir()
            outside = root / "prepare.sh"
            outside.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
            package = self._package(root, prepare=outside)
            with self.assertRaises(SchemaError) as ctx:
                TaskEvaluator("local")._prepare_script(package)
            self.assertIn("must stay under", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
