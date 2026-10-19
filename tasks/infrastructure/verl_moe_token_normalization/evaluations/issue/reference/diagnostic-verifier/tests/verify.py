"""Native CP2/EP2 z-loss behavior check; not a substitute for full RL evidence."""
import json
import math
import os
import sys
import tempfile
from pathlib import Path


def main():
    results = Path(os.environ["PTDBENCH_RESULTS"])
    results.mkdir(parents=True, exist_ok=True)
    workspace = Path(os.environ["PTDBENCH_WORKSPACE"])
    verifier = Path(os.environ["PTDBENCH_VERIFIER"])
    variant = (workspace / "variant.txt").read_text().strip()
    sys.path.insert(0, str(workspace / "src" / variant))
    sys.path.insert(0, str(verifier / "tests"))
    os.environ["TOKENIZER_REF"] = str(workspace / "models/Qwen1.5-MoE-A2.7B-Chat")
    os.environ["RAY_TMPDIR"] = str(results / "ray")
    os.environ["TMPDIR"] = str(results / "tmp")
    Path(os.environ["TMPDIR"]).mkdir(exist_ok=True)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import native_probe as probe
    import numpy as np
    import torch
    record = {"passed": False, "score": 0.0, "scope": "native gradient normalization, not full RL"}
    try:
        torch.manual_seed(1234)
        np.random.seed(1234)
        with tempfile.TemporaryDirectory(dir=results, prefix="native-") as temporary:
            model = str(Path(temporary) / "tiny_moe")
            vocab = probe._create_tiny_moe_model(model)
            data = probe._build_data(vocab)
            legacy = probe._grad_norm(model, False, data)
            per_token = probe._grad_norm(model, True, data)
        record["metrics"] = {"legacy_grad_norm": legacy, "per_token_grad_norm": per_token}
        record["passed"] = (math.isfinite(legacy) and math.isfinite(per_token)
                            and min(legacy, per_token) > 1e-7
                            and math.isclose(legacy, per_token, rel_tol=0.05, abs_tol=1e-4))
        record["score"] = float(record["passed"])
    except Exception as error:
        record["error"] = f"{type(error).__name__}: {error}"
    (results / "result.json").write_text(json.dumps(record, indent=2) + "\n")
    raise SystemExit(0 if record["passed"] else 1)


if __name__ == "__main__":
    main()
