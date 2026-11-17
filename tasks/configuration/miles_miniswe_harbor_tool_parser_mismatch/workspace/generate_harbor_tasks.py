#!/usr/bin/env python3
"""Generate Harbor task directories from selected SWE-bench records."""

from __future__ import annotations

import argparse
import json
import re
import shlex
import shutil
from pathlib import Path
from textwrap import dedent


SAFE_INSTANCE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

INSTRUCTION_TEMPLATE = "{problem_statement}\n"

TASK_TEMPLATE = """schema_version = "1.0"

[task]
name = "swe-bench/swebench-verified__{instance_id}"
description = "SWE-bench Verified debugging task."

[metadata]
difficulty = "{difficulty}"
category = "debugging"

[verifier]
network_mode = "public"
timeout_sec = {max_timeout}

[agent]
user = "root"
network_mode = "public"
timeout_sec = {max_timeout}

[environment]
docker_image = "{docker_image}"
build_timeout_sec = 1800.0
cpus = 1
memory_mb = 4096
storage_mb = 10240
gpus = 0
"""

TEST_TEMPLATE = """#!/bin/bash
set -euo pipefail

{test_commands}

cd ..
cat > parser.py <<EOF
# /// script
# requires-python = ">=3.11"
# dependencies = ["swebench==4.0.3", "datasets==2.16.1", "fastcore<1.11"]
# ///

import json
import os
import sys
from swebench.harness.grading import (
    get_logs_eval,
    get_eval_tests_report,
    get_resolution_status,
)
from swebench.harness.constants import (
    ResolvedStatus,
    EvalType,
    FAIL_TO_PASS,
    PASS_TO_PASS,
    KEY_INSTANCE_ID,
    FAIL_ONLY_REPOS,
    START_TEST_OUTPUT,
    END_TEST_OUTPUT,
)
from swebench.harness.test_spec.test_spec import make_test_spec

with open("/tests/config.json", "r") as file:
    datum = json.load(file)

test_spec = make_test_spec(datum)
report_map = {}
instance_id = datum[KEY_INSTANCE_ID]
report_map[instance_id] = {
    "patch_is_None": False,
    "patch_exists": False,
    "patch_successfully_applied": False,
    "resolved": False,
}
report_map[instance_id]["patch_exists"] = True
test_log_path = os.environ["LOG_FILE"]
with open(test_log_path, "r+") as file:
    content = file.read()
    file.seek(0)
    file.write(f"{START_TEST_OUTPUT}\\n{content}\\n{END_TEST_OUTPUT}")
    file.truncate()

eval_status_map, found = get_logs_eval(test_spec, test_log_path)
if found:
    report_map[instance_id]["patch_successfully_applied"] = True
    eval_ref = {
        KEY_INSTANCE_ID: test_spec.instance_id,
        FAIL_TO_PASS: test_spec.FAIL_TO_PASS,
        PASS_TO_PASS: test_spec.PASS_TO_PASS,
    }
    eval_type = (
        EvalType.FAIL_ONLY
        if test_spec.repo in FAIL_ONLY_REPOS
        else EvalType.PASS_AND_FAIL
    )
    report = get_eval_tests_report(
        eval_status_map, eval_ref, eval_type=eval_type
    )
    if get_resolution_status(report) == ResolvedStatus.FULL.value:
        report_map[instance_id]["resolved"] = True
    report_map[instance_id]["tests_status"] = report

json.dump(
    report_map,
    open("/logs/verifier/report.json", "w"),
    indent=4,
)
print("SWEBench results starts here")
print("PASSED" if report_map[instance_id]["resolved"] else "FAILED")
print("SWEBench results ends here")
sys.exit(0 if report_map[instance_id]["resolved"] else 1)
EOF

chmod +x parser.py
set +e
/usr/bin/python3.10 parser.py | tee -a "$LOG_FILE"
exit_code=$?
set -e

mkdir -p /logs/verifier
if [ "${exit_code}" -eq 0 ]; then
  echo 1 > /logs/verifier/reward.txt
else
  echo 0 > /logs/verifier/reward.txt
fi
exit "${exit_code}"
"""

DOCKERFILE_TEMPLATE = """FROM {docker_image}
WORKDIR /testbed
RUN curl -LsSf https://astral.sh/uv/0.7.13/install.sh | sh
RUN mkdir -p /logs
"""

SOLUTION_TEMPLATE = """#!/bin/bash
set -euo pipefail

cat > /testbed/solution_patch.diff << '__SOLUTION__'
{patch}
__SOLUTION__

cd /testbed
patch --fuzz=5 -p1 -i /testbed/solution_patch.diff
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=3600)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def render_literal(template: str, **replacements: str) -> str:
    output = template
    for key, value in replacements.items():
        output = output.replace("{" + key + "}", value)
    return output


def extract_patch_paths(test_patch: str) -> tuple[list[str], list[str]]:
    preimage_paths = [
        path
        for path in re.findall(r"^--- a/(.*)$", test_patch, re.MULTILINE)
        if path != "/dev/null"
    ]
    postimage_paths = [
        path
        for path in re.findall(r"^\+\+\+ b/(.*)$", test_patch, re.MULTILINE)
        if path != "/dev/null"
    ]
    tracked = list(dict.fromkeys(preimage_paths))
    all_paths = list(dict.fromkeys(preimage_paths + postimage_paths))
    return tracked, all_paths


def build_test_commands(record: dict) -> str:
    from swebench.harness.constants import MAP_REPO_VERSION_TO_SPECS
    from swebench.harness.test_spec.python import get_test_directives

    repo = record["repo"]
    version = record["version"]
    specs = MAP_REPO_VERSION_TO_SPECS[repo][version]
    setup_commands = specs.get("eval_commands", [])
    install_command = specs.get("install", "")
    if repo == "scikit-learn/scikit-learn":
        install_command = ""

    test_patch = record["test_patch"]
    tracked_paths, cleanup_paths = extract_patch_paths(test_patch)
    quoted_tracked = " ".join(shlex.quote(path) for path in tracked_paths)
    quoted_cleanup = " ".join(shlex.quote(path) for path in cleanup_paths)
    reset_tracked = (
        f"git checkout {record['base_commit']} {quoted_tracked}"
        if quoted_tracked
        else ":"
    )
    cleanup = (
        dedent(
            f"""
            for path in {quoted_cleanup}; do
              if [ -e "$path" ] && ! git ls-files --error-unmatch -- "$path" >/dev/null 2>&1; then
                rm -rf -- "$path"
              fi
            done
            """
        ).strip()
        if quoted_cleanup
        else ":"
    )
    test_files = get_test_directives({"repo": repo, "test_patch": test_patch})
    test_command = specs["test_cmd"]

    return dedent(
        f"""#!/bin/bash
        set -uo pipefail -x

        cd /testbed
        set +x
        source /opt/miniconda3/bin/activate
        conda activate testbed
        set -x

        {'\n'.join(setup_commands)}
        {install_command}
        {reset_tracked}
        {cleanup}
        echo {shlex.quote(test_patch)} > /tmp/test_patch.diff
        git apply --check /tmp/test_patch.diff
        git apply /tmp/test_patch.diff

        LOG_FILE=$(mktemp)
        export LOG_FILE
        exec 3>&1 4>&2
        exec > >(tee "$LOG_FILE") 2>&1

        set +x
        {test_command} {' '.join(test_files)} || true
        exec 1>&3 2>&4
        {reset_tracked}
        {cleanup}
        """
    )


def docker_image(instance_id: str) -> str:
    image_suffix = instance_id.replace("__", "_1776_").lower()
    return f"docker.io/swebench/sweb.eval.x86_64.{image_suffix}:latest"


def main() -> None:
    args = parse_args()
    records = [
        json.loads(line)
        for line in args.input.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not records:
        raise AssertionError("input contains no SWE-bench records")

    image_manifest = []
    for record in records:
        instance_id = str(record["instance_id"])
        if not SAFE_INSTANCE_ID.fullmatch(instance_id):
            raise ValueError(f"unsafe instance_id {instance_id!r}")
        task_dir = args.output_dir / instance_id
        if task_dir.exists():
            if not args.overwrite:
                raise FileExistsError(task_dir)
            shutil.rmtree(task_dir)

        tests_dir = task_dir / "tests"
        environment_dir = task_dir / "environment"
        solution_dir = task_dir / "solution"
        for directory in (tests_dir, environment_dir, solution_dir):
            directory.mkdir(parents=True)

        (task_dir / "instruction.md").write_text(
            render_literal(
                INSTRUCTION_TEMPLATE,
                problem_statement=record["problem_statement"].strip(),
            ),
            encoding="utf-8",
        )
        task_image = docker_image(instance_id)
        image_manifest.append(
            {
                "instance_id": instance_id,
                "docker_image": task_image,
                "environment_dir": str(task_dir.relative_to(args.output_dir)),
            }
        )
        (task_dir / "task.toml").write_text(
            render_literal(
                TASK_TEMPLATE,
                instance_id=instance_id,
                difficulty=record.get("difficulty", "hard"),
                max_timeout=str(int(args.timeout)),
                docker_image=task_image,
            ),
            encoding="utf-8",
        )
        (tests_dir / "config.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        test_script = render_literal(
            TEST_TEMPLATE,
            test_commands=build_test_commands(record),
        )
        (tests_dir / "test.sh").write_text(test_script, encoding="utf-8")
        (tests_dir / "test.sh").chmod(0o755)
        (environment_dir / "Dockerfile").write_text(
            render_literal(
                DOCKERFILE_TEMPLATE,
                docker_image=docker_image(instance_id),
            ),
            encoding="utf-8",
        )
        solution = render_literal(
            SOLUTION_TEMPLATE,
            patch=(record.get("patch") or "").strip(),
        )
        (solution_dir / "solve.sh").write_text(solution, encoding="utf-8")
        (solution_dir / "solve.sh").chmod(0o755)

    print(f"generated {len(records)} Harbor task directories under {args.output_dir}")
    (args.output_dir / "image_manifest.json").write_text(
        json.dumps(image_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
