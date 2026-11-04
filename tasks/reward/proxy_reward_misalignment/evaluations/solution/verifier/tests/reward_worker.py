"""Run submitted reward requests; expected outcomes are checked by the parent."""
import importlib.util
import json
from pathlib import Path
import sys


def main():
    workspace = Path(sys.argv[1]).resolve()
    sys.path.insert(0, str(workspace))
    path = workspace / "autorlbench_case/type3_length_reward_v2_fn.py"
    spec = importlib.util.spec_from_file_location("submitted_reward", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    requests = json.loads(sys.stdin.read())
    outputs = []
    for request in requests:
        result = module.compute_score(**request)
        if isinstance(result, dict):
            item = {"score": float(result["score"])}
            if "acc" in result:
                item["acc"] = float(result["acc"])
        else:
            item = float(result)
        outputs.append(item)
    print("AUTORLBENCH_REWARD_OUTPUT=" + json.dumps(outputs))


if __name__ == "__main__":
    main()
