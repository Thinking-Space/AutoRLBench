"""Start the archived VeRL training command with portable paths."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--nvfp4", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    output = args.output.resolve()
    nvfp4 = args.nvfp4.resolve()
    recipe = json.loads((root / "recipe.json").read_bytes())
    environment = dict(os.environ, **recipe["environment"])
    worker_env = {
        "PYTHONPATH": environment["PYTHONPATH"],
        "VLLM_ALLOW_INSECURE_SERIALIZATION": "1",
        "QAT_NVFP4_ROLLOUT": "1",
        "QAT_NVFP4_ROLLOUT_MODEL": str(nvfp4),
        "QAT_WA_SCHEME": "nvfp4",
        "QAT_WA_STATE": str(nvfp4 / "modelopt_state.pt"),
    }
    environment.update(worker_env)
    environment.update(SWANLAB_MODE="disabled", SWANLAB_LOG_DIR=str(output / "swanlog"),
                       MOPD_ROLLOUT_KV_VERIFICATION_RECORD=str(output / "rollout_kv_verification.json"))
    with TemporaryDirectory(prefix="verl-rollout-", dir="/tmp") as ray:
        paths = {
            "MODEL": root / "models/student",
            "TRAIN": root / "data/skywork_or1_7b_math_difficulty1-15_16k.parquet",
            "RESULTS": output,
            "RAY": ray,
        }
        overrides = []
        for option in recipe["overrides"]:
            for key, val in paths.items():
                option = option.replace("{" + key + "}", str(val))
            overrides.append(option)
        overrides.extend(
            f"+ray_kwargs.ray_init.runtime_env.env_vars.{key}={json.dumps(val)}"
            for key, val in worker_env.items()
        )
        command = [sys.executable, "-B", "-m", "verl.trainer.main_ppo", *overrides]
        with (output / "train.stdout").open("xb") as stdout, (output / "train.stderr").open("xb") as stderr:
            subprocess.run(command, cwd=root / "qat", env=environment, stdout=stdout, stderr=stderr, check=True)


if __name__ == "__main__":
    main()
