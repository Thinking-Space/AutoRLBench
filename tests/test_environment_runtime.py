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
    TOOLING_DIR,
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
        # Visible so debugging can reconcile them; read-only because the grading basis must not be agent-editable.
        for stage in ("agent", "verifier"):
            mounts = self._mounts(make_request(stage))
            for container_path in (MODELS_DIR, DATASETS_DIR):
                self.assertIn(container_path, mounts, (stage, container_path))
                self.assertTrue(mounts[container_path].read_only, (stage, container_path))

    def test_stages_without_a_download_cache_still_run(self) -> None:
        # No shared cache only means "this download will not be reused next time", not an error.
        for stage in ("prepare", "agent", "verifier"):
            mounts = self._mounts(make_request(stage, host_models=None, host_datasets=None))
            self.assertNotIn(MODELS_DIR, mounts, stage)
            self.assertNotIn(DATASETS_DIR, mounts, stage)

    # --- tooling: only prepare mounts the repo docker/ read-only ---------------

    def test_prepare_mounts_the_tooling_dir_read_only(self) -> None:
        # Before its script, prepare uses docker/install-autorlbench-tooling.sh to install the latest fetch/dataset
        # tools, so the repo docker/ must be mounted; read-only, since prepare has no reason to modify the repo.
        mount = self._mounts(make_request("prepare", host_tooling=ROOT / "docker"))[
            TOOLING_DIR
        ]
        self.assertEqual(mount.host_path, ROOT / "docker")
        self.assertTrue(mount.read_only)

    def test_later_stages_never_mount_tooling(self) -> None:
        # Agent / verifier do not run prepare, so the tools are useless there, and they run in separate containers.
        for stage in ("agent", "verifier"):
            self.assertNotIn(
                TOOLING_DIR,
                self._mounts(make_request(stage, host_tooling=ROOT / "docker")),
                stage,
            )

    def test_prepare_without_tooling_still_runs(self) -> None:
        # docker/ may not ship with a wheel install; if absent, do not mount it (the installer step also skips).
        self.assertNotIn(
            TOOLING_DIR, self._mounts(make_request("prepare", host_tooling=None))
        )

    # --- environment: read-only in all three stages ---------------------------

    def test_environment_is_read_only_in_every_stage(self) -> None:
        for stage in ("prepare", "agent", "verifier"):
            mount = self._mounts(make_request(stage))[AUTORLBENCH_ENVIRONMENT_DIR]
            self.assertEqual(mount.host_path, ROOT / "environment", stage)
            self.assertTrue(mount.read_only, stage)

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
    """How GPUs declared on the task card become `docker create` flags.

    This is the entire implementation of "can the task actually train". The host's default runtime is runc,
    and passing NVIDIA_VISIBLE_DEVICES via `-e` makes no device visible in the container; only --gpus
    triggers the nvidia hook. The image side (CUDA base image) has always assumed libcuda.so.1 is
    injected by --gpus; this supplies the missing half.
    """

    def _args(self, **kwargs) -> list:
        return DockerRuntime._gpu_args(make_request("agent", **kwargs))

    def test_a_device_list_becomes_a_device_spec(self) -> None:
        # Task cards hold a bare device list (rlve_hyper/task_001 has "0,1,2,3"),
        # and docker only recognizes the device= prefix.
        args = self._args(task_env={"NVIDIA_VISIBLE_DEVICES": "0,1,2,3"})
        self.assertEqual(args[:2], ["--gpus", "device=0,1,2,3"])

    def test_all_is_passed_through(self) -> None:
        self.assertEqual(self._args(task_env={"NVIDIA_VISIBLE_DEVICES": "all"})[:2],
                         ["--gpus", "all"])

    def test_gpu_stages_get_a_usable_shm(self) -> None:
        # Docker's default 64MB /dev/shm makes ray/sglang blow up on the first rollout,
        # with an unhelpful bus error.
        args = self._args(task_env={"NVIDIA_VISIBLE_DEVICES": "0"})
        self.assertIn("--shm-size=16g", args)

    def test_a_cpu_only_task_is_untouched(self) -> None:
        # Byte-for-byte identical to before the wiring: CPU-only tasks and the grading stage have no reason to use more memory.
        self.assertEqual(self._args(), [])

    def test_explicitly_asking_for_no_gpu_is_honoured(self) -> None:
        # `none` / `void` are NVIDIA_VISIBLE_DEVICES's own "no GPU" values.
        for value in ("none", "void", ""):
            self.assertEqual(self._args(task_env={"NVIDIA_VISIBLE_DEVICES": value}), [], value)

    def test_the_host_can_override_the_task_card(self) -> None:
        # A machine with only 2 GPUs should still run a task whose card says 8.
        with mock.patch.dict(os.environ, {"AUTORLBENCH_DOCKER_GPUS": "device=0,1"}):
            args = self._args(task_env={"NVIDIA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7"})
        self.assertEqual(args[:2], ["--gpus", "device=0,1"])

    def test_the_host_can_override_the_shm_size(self) -> None:
        with mock.patch.dict(os.environ, {"AUTORLBENCH_DOCKER_SHM_SIZE": "64g"}):
            args = self._args(task_env={"NVIDIA_VISIBLE_DEVICES": "0"})
        self.assertIn("--shm-size=64g", args)

    def test_every_stage_can_reach_the_gpu_it_declares(self) -> None:
        # Prepare needs GPUs to convert HF weights to torch_dist, and the agent runs training. The verifier gets them
        # too because task_env treats all three stages alike; if it declares none, it naturally has none.
        for stage in ("prepare", "agent", "verifier"):
            request = make_request(stage, task_env={"NVIDIA_VISIBLE_DEVICES": "0"})
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

    def test_the_download_cache_is_exported_to_every_stage(self) -> None:
        # All three stages can mount them, so all three should get real paths rather than empty strings.
        for stage in ("prepare", "agent", "verifier"):
            env = self._env(stage)
            self.assertEqual(env["AUTORLBENCH_MODELS"], MODELS_DIR, stage)
            self.assertEqual(env["AUTORLBENCH_DATASETS"], DATASETS_DIR, stage)

    def test_unmounted_directories_are_the_empty_string(self) -> None:
        # "You cannot see it", not "the path is there but the directory is empty"; the latter would make prepare.sh
        # download the model somewhere that vanishes after the run.
        env = self._env("agent", host_models=None, host_datasets=None)
        self.assertEqual(env["AUTORLBENCH_MODELS"], "")
        self.assertEqual(env["AUTORLBENCH_DATASETS"], "")
        env = self._env("agent", host_environment=None, host_prepared=None)
        self.assertEqual(env["AUTORLBENCH_ENVIRONMENT"], "")
        self.assertEqual(env["AUTORLBENCH_PREPARED"], "")


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


class PrepareScriptTests(unittest.TestCase):
    """How the evaluator translates prepare.sh into a container path."""

    def _package(self, root: Path, *, prepare: Path | None, image: str | None = "verl"):
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
