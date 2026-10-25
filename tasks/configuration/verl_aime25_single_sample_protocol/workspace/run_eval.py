#!/usr/bin/env python3

import argparse
import json
import os
from pathlib import Path

import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = SCRIPT_DIR / "config" / "eval_config.yaml"


def load_protocol(config_path: Path) -> dict:
    if not config_path.is_file():
        raise SystemExit(f"Evaluation config does not exist: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise SystemExit("Evaluation config must be a YAML mapping")

    try:
        benchmark = str(config["benchmark"])
        questions = int(config["questions"])

        generation = config["generation"]
        temperature = float(generation["temperature"])
        top_p = float(generation["top_p"])

        evaluation = config["evaluation"]
        samples_per_question = int(
            evaluation["samples_per_question"]
        )
        reducer = str(evaluation["reducer"]).strip().lower()

    except (KeyError, TypeError, ValueError) as exc:
        raise SystemExit(
            f"Invalid evaluation configuration: {exc}"
        ) from exc

    if benchmark.lower() != "aime25":
        raise SystemExit(
            f"This benchmark case requires AIME25; got {benchmark}"
        )

    if questions != 30:
        raise SystemExit(
            f"This benchmark case requires 30 questions; got {questions}"
        )

    if samples_per_question < 1:
        raise SystemExit(
            "samples_per_question must be >= 1"
        )

    if reducer != "mean":
        raise SystemExit(
            f"This benchmark case requires mean aggregation; got {reducer}"
        )

    return {
        "task": "aime25",
        "benchmark": benchmark,
        "questions": questions,
        "temperature": temperature,
        "top_p": top_p,
        "samples_per_question": samples_per_question,
        "reducer": reducer,
        "expected_generations": (
            questions * samples_per_question
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            os.environ.get(
                "EVAL_CONFIG_PATH",
                str(DEFAULT_CONFIG),
            )
        ),
    )

    parser.add_argument(
        "--protocol-only",
        action="store_true",
        help=(
            "Validate the configured evaluation protocol and print "
            "it as JSON without loading a checkpoint or starting vLLM."
        ),
    )

    args = parser.parse_args()

    protocol = load_protocol(args.config)

    if args.protocol_only:
        print(json.dumps(protocol, indent=2, sort_keys=True))
        return

    # Heavy evaluation dependencies are intentionally imported only
    # for a real model run.  Protocol verification remains CPU-only.
    from inspect_ai import eval_set
    from lighteval.main_inspect import get_inspect_ai_task
    from lighteval.models.abstract_model import InspectAIModelConfig
    from lighteval.tasks.tasks.aime import aime25

    checkpoint = Path(
        os.environ.get(
            "CHECKPOINT_PATH",
            "/models/ppo_critic_actor_hf",
        )
    )

    samples = protocol["samples_per_question"]
    reducer = protocol["reducer"]
    temperature = protocol["temperature"]
    top_p = protocol["top_p"]

    output_dir = Path(
        os.environ.get(
            "OUTPUT_DIR",
            (
                "/workspace/autorlbench_case/reproduced_eval/"
                f"n{samples}"
            ),
        )
    )

    gpu_memory_utilization = float(
        os.environ.get(
            "GPU_MEMORY_UTILIZATION",
            "0.25",
        )
    )

    if not checkpoint.exists():
        raise SystemExit(
            f"Checkpoint does not exist: {checkpoint}"
        )

    if not (checkpoint / "config.json").is_file():
        raise SystemExit(
            "Missing checkpoint config.json: "
            f"{checkpoint / 'config.json'}"
        )

    if not (
        checkpoint / "model.safetensors.index.json"
    ).is_file():
        raise SystemExit(
            "Missing checkpoint model.safetensors.index.json: "
            f"{checkpoint / 'model.safetensors.index.json'}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    log_dir = output_dir / "lighteval-logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 68)
    print("AUTORLBENCH CASE 5 — AIME25 EVALUATION")
    print("=" * 68)

    for key, value in protocol.items():
        print(f"{key:24}: {value}")

    print(f"{'checkpoint':24}: {checkpoint}")
    print(
        f"{'gpu_memory_utilization':24}: "
        f"{gpu_memory_utilization}"
    )
    print(f"{'output_dir':24}: {output_dir}")

    task = get_inspect_ai_task(
        aime25,
        epochs=samples,
        epochs_reducer=reducer,
    )

    if len(task.dataset) != protocol["questions"]:
        raise SystemExit(
            "Dataset-size mismatch: expected "
            f"{protocol['questions']}, got {len(task.dataset)}"
        )

    print()
    print("[PASS] exact AIME25 task constructed")
    print("dataset size =", len(task.dataset))

    model = f"vllm/{checkpoint}"

    model_args = InspectAIModelConfig._parse_args(
        (
            "gpu_memory_utilization="
            f"{gpu_memory_utilization}"
        )
    )

    print()
    print("model      =", model)
    print("model_args =", model_args)

    success, logs = eval_set(
        [task],
        model=[model],
        model_args=model_args,
        log_dir=str(log_dir),
        log_dir_allow_dirty=True,
        display="rich",
        max_connections=50,
        timeout=30,
        retry_on_error=1,
        max_retries=5,
        max_tasks=4,
        temperature=temperature,
        top_p=top_p,
    )

    accuracy = None
    statuses = []

    for log in logs:
        statuses.append(str(log.status))

        if log.results is None:
            continue

        for score in log.results.scores:
            metric = score.metrics.get("accuracy")
            if metric is not None:
                accuracy = float(metric.value)
                break

    summary = {
        **protocol,
        "config": str(args.config),
        "checkpoint": str(checkpoint),
        "success": bool(success),
        "statuses": statuses,
        "accuracy": accuracy,
    }

    summary_path = output_dir / "summary.json"

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
            sort_keys=True,
        )
        f.write("\n")

    print()
    print("=" * 68)
    print("EVALUATION SUMMARY")
    print("=" * 68)
    print(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
        )
    )

    if not success:
        raise SystemExit(
            "Inspect evaluation returned success=False"
        )

    if accuracy is None:
        raise SystemExit(
            "Evaluation succeeded but no accuracy metric was found"
        )

    print()
    print(
        f"[PASS] AIME25 n={samples} evaluation completed; "
        f"accuracy={accuracy:.10f}"
    )


if __name__ == "__main__":
    main()
