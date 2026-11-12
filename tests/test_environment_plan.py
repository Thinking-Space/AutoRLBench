"""environment section: `image` is the pullable reference, `image_stack` describes the stack inside it."""

from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from autorlbench.core.types import EnvironmentImage
from autorlbench.tasks import SchemaError, load_task_package
from autorlbench.tasks.validator import validate_metadata


# A digest-pinned reference from the ENVIRONMENTS.md table, plus its stack label; the
# environment section carries both, and only the reference decides what runs.
FIXTURE_REF = (
    "verlai/verl:vllm017.latest"
    "@sha256:4c43bbf17e90284b1102008399240b25406e8d34fea178d86272231b333b7cb6"
)
FIXTURE_IMAGE = "verl-vllm0.17.0-torch2.10.0-flashattn2.8.3-cu129-py3.12"
# A second reference from the table, to exercise references other than the fixture's.
SLIME_REF = (
    "slimerl/slime:v0.3.2-cu129"
    "@sha256:39be6cbb00f9b6770e664ace0c7b9f5ecff2977a1a205e7926a720f906fbc62c"
)
SLIME_IMAGE = "slime-v0.3.2-sglang0.5.15-torch2.11.0-flashattn2.8.3-cu129-py3.12"

METADATA_HEAD = """title = "Image task"
license = "MIT"
created_date = "2026-09-13"
domain = "rl"

[submitter]
name = "test"

[model]
name = "Qwen2.5-1.5B"

[dataset]
name = "taco"

[training]
framework = "verl"
algo = "grpo"
script = "train.sh"

[outcome]
failure_type = "test"
failure_reason = "test"

"""


def create_task(
    root: Path,
    environment: str,
    *,
    prepare: str | None = None,
) -> Path:
    """Lay out a minimal task; the caller supplies the environment section."""
    task = root / "image-task"
    for directory in ("workspace", "oracle", "environment", "verifier"):
        (task / directory).mkdir(parents=True, exist_ok=True)
    (task / "instruction.md").write_text("Fix it.\n", encoding="utf-8")
    # metadata training.script points here; validation checks the file really is under workspace.
    (task / "workspace" / "train.sh").write_text(
        "#!/usr/bin/env bash\nexit 0\n", encoding="utf-8"
    )
    (task / "verifier" / "test.sh").write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    (task / "task.toml").write_text(METADATA_HEAD + environment, encoding="utf-8")
    if prepare is not None:
        (task / "environment" / "prepare.sh").write_text(prepare, encoding="utf-8")
    return task


class EnvironmentPlanTests(unittest.TestCase):
    def _load(self, environment: str, **kwargs):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        return load_task_package(create_task(Path(temp.name), environment, **kwargs))

    # --- loader -------------------------------------------------------------

    def test_image_task_picks_up_the_conventional_prepare(self) -> None:
        # Nearly every task has a prepare.sh; forcing each task.toml to repeat the path only
        # invites typos.
        package = self._load(
            f'[environment]\nimage = "{FIXTURE_REF}"\nimage_stack = "{FIXTURE_IMAGE}"\n',
            prepare="#!/usr/bin/env bash\n",
        )
        plan = package.environment
        self.assertTrue(package.report.ok, package.report.to_dict())
        self.assertEqual(plan.image, FIXTURE_REF)
        self.assertEqual(plan.image_stack, FIXTURE_IMAGE)
        self.assertEqual(plan.environment_image, EnvironmentImage(FIXTURE_REF))
        self.assertEqual(plan.prepare.name, "prepare.sh")

    def test_explicit_prepare_wins_over_the_convention(self) -> None:
        package = self._load(
            f'[environment]\nimage = "{FIXTURE_REF}"\nprepare = "environment/fetch.sh"\n',
        )
        self.assertEqual(package.environment.prepare.name, "fetch.sh")

    def test_absent_prepare_leaves_the_plan_empty_rather_than_dangling(self) -> None:
        # A plan carrying a missing path would conflate "the task has no prepare.sh" with "declared but wrong path",
        # and the latter must be an error.
        plan = self._load(
            f'[environment]\nimage = "{SLIME_REF}"\n'
            f'image_stack = "{SLIME_IMAGE}"\n'
        ).environment
        self.assertIsNone(plan.prepare)

    def test_image_stack_is_free_form_and_points_at_nothing(self) -> None:
        # A descriptive label, not a file reference: any stack string the author finds
        # informative loads fine; there is nothing to cross-check it against.
        plan = self._load(
            f'[environment]\nimage = "{FIXTURE_REF}"\nimage_stack = "my-private-stack"\n'
        ).environment
        self.assertEqual(plan.image_stack, "my-private-stack")
        self.assertEqual(plan.environment_image, EnvironmentImage(FIXTURE_REF))

    def test_malformed_reference_is_a_schema_error(self) -> None:
        with self.assertRaises(SchemaError) as ctx:
            self._load('[environment]\nimage = "not a reference!!"\n')
        self.assertIn("pullable image reference", str(ctx.exception))

    def test_image_and_stack_are_independent(self) -> None:
        # The pair is not cross-checked: `image` alone determines what runs, so a stack
        # label from a different lineage must not reject the load.
        plan = self._load(
            f'[environment]\nimage = "{SLIME_REF}"\nimage_stack = "{FIXTURE_IMAGE}"\n'
        ).environment
        self.assertEqual(plan.image, SLIME_REF)

    def test_legacy_schema_is_reported_not_silently_ignored(self) -> None:
        package = self._load('[environment]\ndockerfile = "environment/Dockerfile"\n')
        self.assertFalse(package.report.ok)
        paths = {issue.path for issue in package.report.issues}
        self.assertIn("metadata.environment.dockerfile", paths)
        self.assertIn("metadata.environment.image", paths)

    # --- validator branches -------------------------------------------------

    def _environment_issues(self, environment: dict) -> list:
        report = validate_metadata({"environment": environment})
        return [i.path for i in report.issues if i.path.startswith("metadata.environment")]

    def test_validator_takes_any_image_stack_label(self) -> None:
        self.assertEqual(
            self._environment_issues({"image": FIXTURE_REF, "image_stack": "whatever"}),
            [],
        )

    def test_validator_requires_an_image(self) -> None:
        self.assertEqual(self._environment_issues({}), ["metadata.environment.image"])
        self.assertEqual(
            self._environment_issues({"image": ""}), ["metadata.environment.image"]
        )

    def test_validator_rejects_a_reference_without_a_digest(self) -> None:
        # A tag alone floats; the packaged image is the benchmark's ground truth.
        self.assertEqual(
            self._environment_issues({"image": "verlai/verl:vllm017.latest"}),
            ["metadata.environment.image"],
        )

    def test_validator_rejects_a_non_string_image_stack(self) -> None:
        self.assertEqual(
            self._environment_issues({"image": FIXTURE_REF, "image_stack": 3}),
            ["metadata.environment.image_stack"],
        )

    def test_validator_rejects_every_legacy_key(self) -> None:
        # If ignored, task authors would think they still have an effect.
        for key in ("conda", "dockerfile", "requirements"):
            self.assertEqual(
                self._environment_issues({"image": FIXTURE_REF, key: "x"}),
                [f"metadata.environment.{key}"],
                key,
            )

    def test_validator_rejects_a_non_path_prepare(self) -> None:
        self.assertEqual(
            self._environment_issues({"image": FIXTURE_REF, "prepare": 3}),
            ["metadata.environment.prepare"],
        )


class EnvironmentEnvTests(unittest.TestCase):
    """metadata.environment.env: per-task variables (the image carries only those shared by the environment)."""

    def _load(self, environment: str):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        return load_task_package(create_task(Path(temp.name), environment))

    def test_env_is_parsed_into_the_plan(self):
        package = self._load(
            f'[environment]\nimage = "{FIXTURE_REF}"\n'
            '[environment.env]\n'
            'CUDA_VISIBLE_DEVICES = "0,1"\n'
            'NVIDIA_VISIBLE_DEVICES = "0,1"\n'
        )
        self.assertEqual(
            package.environment.env,
            {"CUDA_VISIBLE_DEVICES": "0,1", "NVIDIA_VISIBLE_DEVICES": "0,1"},
        )

    def test_absent_env_is_an_empty_mapping(self):
        self.assertEqual(self._load(f'[environment]\nimage = "{FIXTURE_REF}"\n').environment.env, {})

    def test_tasks_cannot_hijack_the_harness_channel(self):
        # AUTORLBENCH_* is how the harness tells the container "which stage you are".
        with self.assertRaises(SchemaError) as ctx:
            self._load(
                f'[environment]\nimage = "{FIXTURE_REF}"\n[environment.env]\nAUTORLBENCH_STAGE = "verifier"\n'
            )
        self.assertIn("AUTORLBENCH_STAGE", str(ctx.exception))

    def test_unquoted_number_is_rejected_rather_than_coerced(self):
        # `CUDA_VISIBLE_DEVICES = 0` is an int in TOML. Silently calling str() would make
        # "use only GPU 0" indistinguishable from "a typo".
        with self.assertRaises(SchemaError) as ctx:
            self._load(
                f'[environment]\nimage = "{FIXTURE_REF}"\n[environment.env]\nCUDA_VISIBLE_DEVICES = 0\n'
            )
        self.assertIn("CUDA_VISIBLE_DEVICES", str(ctx.exception))

    def test_scalar_env_is_rejected(self):
        with self.assertRaises(SchemaError):
            self._load(f'[environment]\nimage = "{FIXTURE_REF}"\nenv = "nope"\n')


class EnvironmentEnvRuntimeTests(unittest.TestCase):
    def _request(self, **kwargs):
        from autorlbench.runtimes import SandboxRequest

        root = Path("/host/run")
        defaults = dict(
            stage="rerun",
            task_id="t",
            host_task_root=root / "task",
            host_visible_task=root / "visible",
            host_workspace=root / "workspace",
            host_results=root / "results",
        )
        defaults.update(kwargs)
        return SandboxRequest(**defaults)

    def test_task_env_reaches_the_command_but_cannot_shadow_the_harness(self):
        from autorlbench.runtimes.base import CommandSpec
        from autorlbench.runtimes.providers.docker import DockerRuntime

        request = self._request(task_env={"CUDA_VISIBLE_DEVICES": "0,1"})
        merged = DockerRuntime()._merged_spec(request, CommandSpec(shell="true"))

        self.assertEqual(merged.env["CUDA_VISIBLE_DEVICES"], "0,1")
        self.assertEqual(merged.env["AUTORLBENCH_STAGE"], "rerun")

    def test_caller_env_still_wins_over_the_task_declaration(self):
        from autorlbench.runtimes.base import CommandSpec
        from autorlbench.runtimes.providers.docker import DockerRuntime

        request = self._request(
            task_env={"CUDA_VISIBLE_DEVICES": "0,1"},
            env={"CUDA_VISIBLE_DEVICES": "2,3"},
        )
        merged = DockerRuntime()._merged_spec(request, CommandSpec(shell="true"))
        self.assertEqual(merged.env["CUDA_VISIBLE_DEVICES"], "2,3")


class EnvironmentImageTests(unittest.TestCase):
    def test_environment_image_is_just_the_reference(self):
        image = EnvironmentImage(FIXTURE_REF)
        self.assertEqual(image.tag, FIXTURE_REF)
        self.assertEqual(image, EnvironmentImage(image.tag))

    def test_a_malformed_reference_is_rejected(self):
        # The reference is what the backends pull; a malformed one must fail here,
        # not at the first docker pull.
        for bad in ("not a ref!!", "", "UPPERCASE/repo:v1"):
            with self.assertRaises(ValueError):
                EnvironmentImage(bad)


class EnsureEnvironmentTests(unittest.TestCase):
    def _run(self, *, exists: bool, rebuild: bool = False):
        from autorlbench.runtimes.providers.docker import DockerRuntime

        runtime = DockerRuntime()
        image = EnvironmentImage(FIXTURE_REF)
        with mock.patch.object(DockerRuntime, "_image_exists", return_value=exists), \
                mock.patch("autorlbench.runtimes.providers.docker.subprocess.run") as run:
            artifact = runtime.ensure_environment(image, rebuild=rebuild)
        self.assertEqual(artifact.reference, image.tag)
        return run

    def test_existing_image_is_reused_without_pulling(self):
        self.assertFalse(self._run(exists=True).called)

    def test_missing_image_is_pulled_by_reference(self):
        run = self._run(exists=False)
        argv = run.call_args.args[0]
        self.assertEqual(argv[:2], ["docker", "pull"])
        self.assertEqual(argv[2], FIXTURE_REF)
        self.assertTrue(run.call_args.kwargs["check"])

    def test_rebuild_repulls_even_when_the_image_exists(self):
        self.assertTrue(self._run(exists=True, rebuild=True).called)

    def test_pull_failure_propagates(self):
        from autorlbench.runtimes.providers.docker import DockerRuntime

        with mock.patch.object(DockerRuntime, "_image_exists", return_value=False), \
                mock.patch(
                    "autorlbench.runtimes.providers.docker.subprocess.run",
                    side_effect=subprocess.CalledProcessError(1, ["docker", "pull"]),
                ):
            with self.assertRaises(subprocess.CalledProcessError):
                DockerRuntime().ensure_environment(EnvironmentImage(FIXTURE_REF))


if __name__ == "__main__":
    unittest.main()
