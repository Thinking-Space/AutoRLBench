from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


VERIFIER = Path(__file__).resolve().parents[1]


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def read_config(path):
    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique_object)
    if not isinstance(value, dict):
        raise ValueError("Generation configuration must be a JSON object")
    return value


def first_stop(sequence, eos_ids):
    return next((index for index, token in enumerate(sequence) if token in eos_ids), None)


def grade(workspace):
    spec = json.loads((VERIFIER / "resources/grader_spec.json").read_text())
    relative = Path(spec["candidate"])
    candidate = workspace / relative
    current = workspace
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError("Configuration path must not contain symlinks")
    config = read_config(candidate)
    reference = read_config(VERIFIER.parent / "oracle/generation_config.json")
    eos_ids = config.get("eos_token_id")
    if not isinstance(eos_ids, list) or not eos_ids:
        raise ValueError("eos_token_id must be a nonempty list of integer identifiers")
    if any(type(token) is not int for token in eos_ids):
        raise ValueError("EOS identifiers must be integers, not strings or booleans")
    expected = set(spec["expected_eos_token_ids"])
    checks = {
        "eos_set_matches": set(eos_ids) == expected and len(eos_ids) == len(expected),
        "other_fields_unchanged": {
            key: value for key, value in config.items() if key != "eos_token_id"
        } == {key: value for key, value in reference.items() if key != "eos_token_id"},
        "assistant_turn_stops": first_stop([42, 151645, 43, 151643], eos_ids) == 1,
        "end_of_text_still_stops": first_stop([42, 151643, 43], eos_ids) == 1,
        "ordinary_tokens_continue": first_stop([42, 43, 44], eos_ids) is None,
    }
    return checks


def main():
    parser = argparse.ArgumentParser(description="Verify the minimal EOS configuration repair.")
    parser.add_argument("--workspace", type=Path, default=Path(os.environ.get("AUTORLBENCH_WORKSPACE", "/workspace")))
    parser.add_argument("--results", type=Path, default=Path(os.environ.get("AUTORLBENCH_RESULTS", "/results")))
    args = parser.parse_args()
    try:
        checks = grade(args.workspace)
        passed = all(checks.values())
        reason = "All configuration checks passed" if passed else "Failed configuration checks"
    except (OSError, ValueError, TypeError) as error:
        checks = {}
        passed = False
        reason = str(error)
    metrics = {
        "evaluation_mode": "configuration_repair_only",
        "checks": checks,
        "reason": reason,
        "training_rerun_verified": False,
        "model_inference_verified": False,
    }
    result = {"passed": passed, "score": float(passed), "metrics": metrics}
    args.results.mkdir(parents=True, exist_ok=True)
    (args.results / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    (args.results / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
