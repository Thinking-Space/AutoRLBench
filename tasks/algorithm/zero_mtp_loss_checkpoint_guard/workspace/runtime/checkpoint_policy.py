"""Execute the exact preserved Bash guard; do not emulate its policy."""
from pathlib import Path
import subprocess

def check_guard(source, *, num_rollouts=1, save_interval=1, debug_train_only=True):
    env = {"PATH": "/usr/bin:/bin", "NUM_ROLLOUT": str(num_rollouts),
           "SAVE_INTERVAL": str(save_interval), "DEBUG_TRAIN_ONLY": "1" if debug_train_only else "0",
           "MODEL_PATH": "unused-random-model", "CKPT_PATH": "unused-path",
           "EXP_NAME": "tiny-smoke", "MEGATRON_MODEL_PATH": "unused-random-model"}
    result = subprocess.run(["bash", "-c", 'source "$1"; training_build_checkpoint_args',
                             "guard-probe", str(Path(source).resolve())], env=env,
                            capture_output=True, text=True, timeout=10)
    return {"accepted": result.returncode == 0, "exit_code": result.returncode,
            "debug_train_only": debug_train_only, "num_rollouts": num_rollouts,
            "save_interval": save_interval}
