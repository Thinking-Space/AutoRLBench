#!/usr/bin/env python3
"""Run and adjudicate the FP32 -> KV2 -> KV4 evaluation gate.

The QAT/OPD experiment has an explicit model-selection rule before any long
training is started:

* evaluate the unquantized model (``kv_bits=32``) first;
* evaluate the same model/data with KV2;
* if KV2 is below one third of the FP32 score (or has zero score), evaluate
  KV4 and select KV4; otherwise select KV2.

This helper owns only orchestration and bookkeeping.  The actual generation
and answer extraction remain in the selected evaluator (AIME or LongBench),
so all three runs use the same evaluator within one profile.  By default the command is a
dry-run; pass ``--run`` to launch the independent evaluators.  Existing JSON
reports can be adjudicated without a model or GPU by passing ``--fp-report``
and ``--kv2-report`` (and, when needed, ``--kv4-report``).

The two experiment profiles are intentionally explicit.  DeepSeek is checked
on AIME 2024, while Qwen2.5-7B-1M is checked only on the ``length=medium``
subset of LongBench v2, without filtering difficulty.  Their training datasets are recorded separately in the gate
manifest so a result from one line cannot be silently compared with the
other.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Mapping, Sequence


QAT_ROOT = Path(os.environ.get("QAT_DATA_ROOT", ".")).expanduser()
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PYTHON = Path(
    os.environ.get(
        "MOPD_PREFLIGHT_PYTHON",
        "envs/vllm-0.19.1-test/bin/python",
    )
)
DEFAULT_AIME = QAT_ROOT / "data/raw/aime_2024/data/train-00000-of-00001.parquet"
GATE_SCHEMA_VERSION = 1
KV2_THRESHOLD = 1.0 / 3.0


@dataclass(frozen=True)
class EvalProfile:
    """Paths and defaults that identify one requested experiment line."""

    name: str
    model: Path
    training_data: Path
    eval_data: Path
    evaluator_profile: str
    description: str
    evaluator_script: str = "eval_aime.py"
    subset: str | None = None
    difficulty: str | None = None
    expected_rows: int | None = None
    default_limit: int = 30
    default_max_new_tokens: int = 32768
    benchmark: str = "AIME2024"
    reference_dtype: str = "fp32"
    reference_report_name: str = "fp32"


def profile_defaults(root: Path = QAT_ROOT) -> dict[str, EvalProfile]:
    """Return the two governed model/data profiles without touching the network."""

    return {
        "deepseek_skywork": EvalProfile(
            name="deepseek_skywork",
            model=root / "models/base/deepseek-r1-distill-qwen-7b",
            training_data=root
            / "data/processed/skywork_or1_7b_math_difficulty1-15_16k.parquet",
            eval_data=root / "data/raw/aime_2024/data/train-00000-of-00001.parquet",
            # AIME is the held-out math test set for this line.
            evaluator_profile="aime_math",
            description="DeepSeek-R1-Distill-Qwen-7B + Skywork OR1 math",
            benchmark="AIME2024",
            default_max_new_tokens=32768,
        ),
        "qwen25_longrlvr": EvalProfile(
            name="qwen25_longrlvr",
            model=root / "models/base/qwen2.5-7b-instruct-1m",
            training_data=root
            / "data/processed/longrlvr_qwen25_1m_prompt65536.parquet",
            eval_data=root / "data/raw/longbench-v2/data.json",
            # LongRLVR is training-only.  The held-out benchmark is LongBench
            # v2, currently restricted to all 215 medium-length questions.
            evaluator_profile="longbench_v2_medium",
            description="Qwen2.5-7B-Instruct-1M + LongRLVR",
            evaluator_script="eval_longbench_v2.py",
            subset="medium",
            difficulty="all",
            expected_rows=215,
            default_limit=215,
            default_max_new_tokens=4096,
            benchmark="LongBench-v2/medium",
            # FP32 attention is infeasible for 100k+ token medium contexts;
            # use an unquantized BF16 reference with the same efficient kernel
            # family as the KV runs.
            reference_dtype="bf16",
            reference_report_name="bf16",
        ),
    }


def _finite_score(value: Any, label: str) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric, got {value!r}") from exc
    if not math.isfinite(score) or score < 0.0:
        raise ValueError(f"{label} must be finite and non-negative, got {value!r}")
    return score


def _integral_json_int(value: Any, label: str) -> int:
    """Parse a JSON numeric value without silently truncating fractions."""

    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer, got {value!r}")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value) or not value.is_integer():
            raise ValueError(f"{label} must be an integer, got {value!r}")
        return int(value)
    # Do not accept strings such as ``"4"`` in a strict evaluator report:
    # they indicate a schema/serialization bug and can hide accidental casts.
    raise ValueError(f"{label} must be an integer, got {value!r}")


def _report_summary(report: Mapping[str, Any], label: str) -> dict[str, Any]:
    """Normalize one schema-v2 evaluator report for the relative-bit gate."""

    if not isinstance(report, Mapping):
        raise ValueError(f"{label} report must be a JSON object")
    bits_raw = report.get("kv_bits")
    try:
        bits = _integral_json_int(bits_raw, f"{label}.kv_bits")
    except ValueError as exc:
        raise ValueError(f"{label}.kv_bits is missing or invalid: {bits_raw!r}") from exc
    if bits not in (2, 4, 8, 16, 32):
        raise ValueError(f"{label}.kv_bits must be one of 2,4,8,16,32; got {bits}")

    count_raw = report.get("count")
    try:
        count = _integral_json_int(count_raw, f"{label}.count")
    except ValueError as exc:
        raise ValueError(f"{label}.count is missing or invalid: {count_raw!r}") from exc
    if count < 0:
        raise ValueError(f"{label}.count must be non-negative")

    total_samples_raw = report.get("total_samples")
    try:
        total_samples = _integral_json_int(total_samples_raw, f"{label}.total_samples")
    except ValueError as exc:
        raise ValueError(
            f"{label}.total_samples is missing or invalid: {total_samples_raw!r}"
        ) from exc
    if total_samples < 0:
        raise ValueError(f"{label}.total_samples must be non-negative")

    accuracy = _finite_score(report.get("accuracy"), f"{label}.accuracy")
    if accuracy > 1.0 + 1e-9:
        raise ValueError(f"{label}.accuracy must be <= 1, got {accuracy}")
    truncated = _finite_score(
        report.get("truncated_fraction", 0.0), f"{label}.truncated_fraction"
    )
    if truncated > 1.0 + 1e-9:
        raise ValueError(f"{label}.truncated_fraction must be <= 1, got {truncated}")

    extraction_failures = report.get("extraction_failures")
    if extraction_failures is None:
        raise ValueError(f"{label}.extraction_failures is missing")
    try:
        extraction_failures = int(extraction_failures)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label}.extraction_failures is invalid") from exc
    if extraction_failures < 0 or extraction_failures > total_samples:
        raise ValueError(
            f"{label}.extraction_failures={extraction_failures} outside [0,{total_samples}]"
        )

    eos_samples = report.get("eos_samples")
    if eos_samples is None:
        raise ValueError(f"{label}.eos_samples is missing")
    try:
        eos_samples = int(eos_samples)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label}.eos_samples is invalid") from exc
    if eos_samples < 0 or eos_samples > total_samples:
        raise ValueError(f"{label}.eos_samples={eos_samples} outside [0,{total_samples}]")

    sampling = report.get("sampling")
    if not isinstance(sampling, Mapping):
        raise ValueError(f"{label}.sampling metadata is missing")
    # These fields define comparability.  Other fields (e.g. hf_top_k) are
    # derived aliases and are not part of the cross-bit identity check.
    sampling_contract = {
        key: sampling.get(key)
        for key in ("temperature", "top_k", "top_p", "max_new_tokens", "num_samples")
    }
    missing_sampling = [key for key, value in sampling_contract.items() if value is None]
    if missing_sampling:
        raise ValueError(f"{label}.sampling missing {missing_sampling}")

    dtype_resolved = report.get("dtype_resolved")
    if not isinstance(dtype_resolved, str) or dtype_resolved not in {
        "fp32",
        "bf16",
        "fp16",
    }:
        raise ValueError(
            f"{label}.dtype_resolved must be one of fp32/bf16/fp16, "
            f"got {dtype_resolved!r}"
        )

    return {
        "label": label,
        "path": report.get("_report_path"),
        "model": str(report.get("model", "")),
        "model_path_real": str(report.get("model_path_real", report.get("model", ""))),
        "data": str(report.get("data", "")),
        "profile": str(report.get("profile", "")),
        "kv_bits": bits,
        "accuracy": accuracy,
        "count": count,
        "total_samples": total_samples,
        "truncated_samples": int(report.get("truncated_samples", 0) or 0),
        "truncated_fraction": truncated,
        "extraction_failures": extraction_failures,
        "extraction_failure_fraction": (
            extraction_failures / total_samples if total_samples else 0.0
        ),
        "eos_samples": eos_samples,
        "eos_fraction": eos_samples / total_samples if total_samples else 0.0,
        "mean_response_tokens": report.get("mean_response_tokens"),
        "dtype": report.get("dtype"),
        "dtype_resolved": dtype_resolved,
        "seed": report.get("seed"),
        "instruction": report.get("instruction"),
        "attn_implementation": report.get("attn_implementation"),
        "preserve_projection_bias": report.get("preserve_projection_bias"),
        "evaluator_git_revision": report.get("evaluator_git_revision"),
        "evaluator_source_sha256": report.get("evaluator_source_sha256"),
        "evaluator_schema_version": report.get("evaluator_schema_version", 1),
        "sampling": dict(sampling),
        "sampling_contract": sampling_contract,
    }


def load_report(
    path: Path,
    *,
    label: str,
    expected_bits: int | None = None,
    expected_dtype: str | None = None,
) -> dict[str, Any]:
    """Load and validate one JSON evaluator report."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"cannot read {label} report {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {label} report {path}: {exc}") from exc
    if isinstance(payload, Mapping):
        payload = dict(payload)
        payload["_report_path"] = str(path.resolve())
    normalized = _report_summary(payload, label)
    if expected_bits is not None and normalized["kv_bits"] != expected_bits:
        raise ValueError(
            f"{label} report declares kv_bits={normalized['kv_bits']}, expected {expected_bits}"
        )
    required_dtype = expected_dtype if expected_dtype is not None else ("fp32" if expected_bits == 32 else None)
    if required_dtype is not None and normalized.get("dtype_resolved") != required_dtype:
        raise ValueError(
            f"{label} report does not match required reference dtype {required_dtype!r}: "
            f"dtype_resolved={normalized.get('dtype_resolved')!r}"
        )
    return normalized


def _same_path(left: str, right: str) -> bool:
    if not left or not right:
        return left == right
    return os.path.realpath(left) == os.path.realpath(right)


def _validate_comparable(fp: Mapping[str, Any], quantized: Mapping[str, Any], label: str) -> None:
    """Reject reports that cannot support a relative score decision."""

    for field in ("model_path_real", "data", "profile", "count", "total_samples"):
        left = fp[field]
        right = quantized[field]
        equal = _same_path(str(left), str(right)) if field in {"model_path_real", "data"} else left == right
        if not equal:
            raise ValueError(
                f"{label} is not comparable to FP report: {field} differs ({left!r} vs {right!r})"
            )
    if fp["sampling_contract"] != quantized["sampling_contract"]:
        raise ValueError(
            f"{label} is not comparable to FP report: sampling contract differs"
        )
    # These fields affect generation semantics even though they are not part
    # of the bit-width decision.  Legacy reports may omit them; two equally
    # old reports remain usable, but a mixed pair is rejected rather than
    # silently comparing different instructions/seeds/source code.
    for field in (
        "seed",
        "instruction",
        "attn_implementation",
        "preserve_projection_bias",
        "evaluator_git_revision",
        "evaluator_source_sha256",
    ):
        left = fp.get(field)
        right = quantized.get(field)
        if left is None and right is None:
            continue
        if left != right:
            raise ValueError(
                f"{label} is not comparable to FP report: {field} differs "
                f"({left!r} vs {right!r})"
            )


def _normalize_decision_report(report: Mapping[str, Any], label: str) -> dict[str, Any]:
    """Normalize a report before applying the selection rule.

    ``choose_primary_bits`` is also a public, pure helper and is routinely
    called with the already-normalized objects stored in a previous decision
    manifest.  Those objects contain ``sampling_contract`` and ``path``
    aliases, but may not contain the raw evaluator keys.  Older code skipped
    validation whenever that marker was present, which allowed values such as
    ``kv_bits=4.5`` or ``dtype_resolved='auto'`` to be silently coerced by
    ``int()``.  Reconstruct the small raw-shaped view and run the same strict
    normalizer for every call path.
    """

    if not isinstance(report, Mapping):
        raise ValueError(f"{label} report must be a JSON object")
    candidate = dict(report)
    if not isinstance(candidate.get("sampling"), Mapping):
        contract = candidate.get("sampling_contract")
        if isinstance(contract, Mapping):
            candidate["sampling"] = dict(contract)
    if "_report_path" not in candidate:
        path = candidate.get("path")
        if path not in (None, ""):
            candidate["_report_path"] = path
    normalized = _report_summary(candidate, label)
    # ``_report_summary`` intentionally emits the private path only when it
    # came from a raw report.  Preserve a normalized manifest's path so the
    # returned bundle remains auditable and can be compared against disk.
    if normalized.get("path") in (None, ""):
        path = candidate.get("_report_path", candidate.get("path"))
        if path not in (None, ""):
            normalized["path"] = str(path)
    return normalized


def choose_primary_bits(
    fp_report: Mapping[str, Any],
    kv2_report: Mapping[str, Any],
    kv4_report: Mapping[str, Any] | None = None,
    *,
    threshold: float = KV2_THRESHOLD,
) -> dict[str, Any]:
    """Apply the requested KV2/KV4 selection rule to normalized reports.

    The strict inequality is intentional: a KV2/FP ratio exactly equal to
    one third is *not* below the user's cutoff and therefore keeps KV2 as the
    primary setting.  If the FP score is zero, no relative ratio can prove
    KV2 viable; the conservative result is ``needs_kv4``.
    """

    fp = _normalize_decision_report(fp_report, "fp")
    kv2 = _normalize_decision_report(kv2_report, "kv2")
    if fp.get("kv_bits") != 32:
        raise ValueError(f"FP report must use kv_bits=32, got {fp.get('kv_bits')}")
    if kv2.get("kv_bits") != 2:
        raise ValueError(f"KV2 report must use kv_bits=2, got {kv2.get('kv_bits')}")
    _validate_comparable(fp, kv2, "KV2 report")

    if isinstance(threshold, bool):
        raise ValueError(f"threshold must be a finite number in (0,1), got {threshold!r}")
    try:
        threshold_value = float(threshold)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"threshold must be a finite number in (0,1), got {threshold!r}") from exc
    if not math.isfinite(threshold_value) or not 0.0 < threshold_value < 1.0:
        raise ValueError(f"threshold must be in (0,1), got {threshold!r}")

    fp_accuracy = float(fp["accuracy"])
    kv2_accuracy = float(kv2["accuracy"])
    cutoff = fp_accuracy * threshold_value
    ratio = kv2_accuracy / fp_accuracy if fp_accuracy > 0.0 else None
    near_zero = kv2_accuracy <= 0.0
    below_cutoff = fp_accuracy <= 0.0 or kv2_accuracy < cutoff
    needs_kv4 = bool(near_zero or below_cutoff)

    quality_warnings: list[str] = []
    for item in (fp, kv2):
        if float(item["truncated_fraction"]) > 0.0:
            quality_warnings.append(
                f"{item['label']} has truncation fraction {item['truncated_fraction']:.3f}"
            )
        if int(item["extraction_failures"]) > 0:
            quality_warnings.append(
                f"{item['label']} has {item['extraction_failures']} answer-extraction failures"
            )

    normalized_kv4: dict[str, Any] | None = None
    if kv4_report is not None:
        normalized_kv4 = _normalize_decision_report(kv4_report, "kv4")
        if normalized_kv4.get("kv_bits") != 4:
            raise ValueError(f"KV4 report must use kv_bits=4, got {normalized_kv4.get('kv_bits')}")
        _validate_comparable(fp, normalized_kv4, "KV4 report")
        if float(normalized_kv4["truncated_fraction"]) > 0.0:
            quality_warnings.append(
                f"kv4 has truncation fraction {normalized_kv4['truncated_fraction']:.3f}"
            )
        if int(normalized_kv4["extraction_failures"]) > 0:
            quality_warnings.append(
                f"kv4 has {normalized_kv4['extraction_failures']} answer-extraction failures"
            )

    if needs_kv4:
        if normalized_kv4 is None:
            primary_bits: int | None = None
            status = "needs_kv4"
            reason = (
                "KV2 is zero or below the FP score's one-third cutoff; run KV4 before selecting a primary"
            )
        else:
            primary_bits = 4
            status = "kv4_primary"
            reason = (
                "KV2 is zero or below the FP score's one-third cutoff; KV4 selected as primary"
            )
    else:
        primary_bits = 2
        status = "kv2_primary"
        reason = "KV2 reaches at least one third of the FP score; KV2 selected as primary"

    return {
        "gate_schema_version": GATE_SCHEMA_VERSION,
        "decision_status": status,
        "primary_kv_bits": primary_bits,
        "threshold": threshold_value,
        "fp_accuracy": fp_accuracy,
        "kv2_accuracy": kv2_accuracy,
        "kv2_over_fp_ratio": ratio,
        "kv2_cutoff_accuracy": cutoff,
        "kv2_near_zero": near_zero,
        "kv2_below_cutoff": below_cutoff,
        "kv4_accuracy": None if normalized_kv4 is None else normalized_kv4["accuracy"],
        "reason": reason,
        "quality_warnings": quality_warnings,
        "reports": {"fp": fp, "kv2": kv2, "kv4": normalized_kv4},
    }


def _shell_quote(value: str) -> str:
    if value and all(char.isalnum() or char in "_./:=+@,%+-" for char in value):
        return value
    return "'" + value.replace("'", "'\\''") + "'"


def build_eval_command(
    *,
    python: Path,
    model: Path,
    data: Path | None,
    bits: int,
    evaluator_profile: str,
    output: Path,
    limit: int,
    num_samples: int,
    max_new_tokens: int,
    device: str,
    seed: int | None,
    dtype: str,
    attn_implementation: str | None,
    evaluator_script: str = "eval_aime.py",
    subset: str | None = None,
    difficulty: str | None = None,
    expected_rows: int | None = None,
    thinking: bool = True,
    kivi: bool = False,
) -> list[str]:
    """Construct one evaluator command; kept pure for dry-run tests."""

    command = [
        str(python),
        str(REPO_ROOT / "scripts" / evaluator_script),
        "--model",
        str(model),
        "--kv-bits",
        str(bits),
        "--limit",
        str(limit),
        "--max-new-tokens",
        str(max_new_tokens),
        "--device",
        device,
        "--dtype",
        dtype,
        "--progress",
        "--output",
        str(output),
    ]
    if data is not None:
        command[2:2] = ["--data", str(data)]
    if evaluator_script == "eval_aime.py":
        command.extend(["--profile", evaluator_profile, "--num-samples", str(num_samples)])
    else:
        if subset is not None:
            command.extend(["--subset", subset])
        if difficulty is not None:
            command.extend(["--difficulty", difficulty])
        if expected_rows is not None:
            command.extend(["--expected-rows", str(expected_rows)])
    command.append("--thinking" if thinking else "--no-thinking")
    if kivi:
        command.extend(["--kivi", "--kivi-group-size", "32", "--kivi-residual-length", "32"])
    if seed is not None:
        command.extend(["--seed", str(seed)])
    if attn_implementation:
        command.extend(["--attn-implementation", attn_implementation])
    return command


def _run_one(command: Sequence[str], *, env: Mapping[str, str] | None = None) -> None:
    print("[kv-gate] running: " + " ".join(_shell_quote(str(item)) for item in command), flush=True)
    subprocess.run(command, check=True, env=None if env is None else dict(env))


def _child_environment() -> dict[str, str]:
    """Build the evaluator environment without dropping caller libraries."""

    env = dict(os.environ)
    inherited = [item for item in env.get("PYTHONPATH", "").split(os.pathsep) if item]
    paths: list[str] = [str(REPO_ROOT)]
    paths.extend(item for item in inherited if item != str(REPO_ROOT))
    env["PYTHONPATH"] = os.pathsep.join(paths)
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    return env


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--profile", choices=sorted(profile_defaults()), default="deepseek_skywork")
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--data", type=Path, default=None, help="held-out evaluation data for the selected profile")
    parser.add_argument("--training-data", type=Path, default=None, help="recorded target training parquet")
    parser.add_argument(
        "--evaluator-profile",
        choices=(
            "aime_math",
            "paper",
            "longrlvr",
            "training",
            "longbench_v2_medium",
        ),
        default=None,
        help="decoder preset for the selected evaluator; default follows the selected experiment profile",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--fp-report", type=Path, default=None)
    parser.add_argument("--kv2-report", type=Path, default=None)
    parser.add_argument("--kv4-report", type=Path, default=None)
    parser.add_argument("--summary", type=Path, default=None, help="gate_selection.json path")
    parser.add_argument("--threshold", type=float, default=KV2_THRESHOLD)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", choices=("auto", "bf16", "fp16", "fp32"), default="auto")
    parser.add_argument("--attn-implementation", default=None)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--run", action="store_true", help="run missing evaluations in FP -> KV2 -> KV4 order")
    parser.add_argument(
        "--always-evaluate-kv4",
        action="store_true",
        help="also run KV4 when KV2 passes, for an optional three-way diagnostic",
    )
    parser.add_argument("--overwrite", action="store_true", help="allow replacing existing evaluator reports")
    parser.add_argument("--json", action="store_true", help="print the final machine-readable decision")
    return parser


def _resolve_report(path: Path | None, output_dir: Path, name: str) -> Path:
    return path if path is not None else output_dir / f"{name}.json"


def _assert_output_safe(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise RuntimeError(f"refusing to overwrite existing report {path}; pass --overwrite")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    profiles = profile_defaults()
    selected = profiles[args.profile]
    model = args.model or selected.model
    data = args.data or selected.eval_data
    if args.data is not None and not args.data.is_file():
        raise SystemExit(f"explicit evaluation data path does not exist: {args.data}")
    command_data = data if data.is_file() else None
    training_data = args.training_data or selected.training_data
    evaluator_profile = args.evaluator_profile or selected.evaluator_profile
    limit = args.limit if args.limit is not None else selected.default_limit
    max_new_tokens = (
        args.max_new_tokens if args.max_new_tokens is not None else selected.default_max_new_tokens
    )
    output_root = "aime" if selected.benchmark == "AIME2024" else "longbench_v2"
    output_dir = args.output_dir or QAT_ROOT / f"runs/{output_root}" / f"kv-gate-{args.profile}"
    fp_path = _resolve_report(args.fp_report, output_dir, selected.reference_report_name)
    kv2_path = _resolve_report(args.kv2_report, output_dir, "kv2")
    kv4_path = _resolve_report(args.kv4_report, output_dir, "kv4")
    summary_path = args.summary or output_dir / "gate_selection.json"

    if limit <= 0 or args.num_samples <= 0 or max_new_tokens <= 0:
        raise SystemExit("limit, num-samples, and max-new-tokens must be positive")
    if not math.isfinite(args.threshold) or not 0.0 < args.threshold < 1.0:
        raise SystemExit("threshold must be in (0,1)")

    commands = {
        "fp": build_eval_command(
            python=args.python,
            model=model,
            data=command_data,
            bits=32,
            evaluator_profile=evaluator_profile,
            output=fp_path,
            limit=limit,
            num_samples=args.num_samples,
            max_new_tokens=max_new_tokens,
            device=args.device,
            seed=args.seed,
            # The profile owns the unquantized reference dtype.  Math uses
            # FP32; long-context Qwen uses BF16 so SDPA can use a memory-
            # efficient kernel at 100k+ token inputs.
            dtype=selected.reference_dtype,
            attn_implementation=args.attn_implementation,
            evaluator_script=selected.evaluator_script,
            subset=selected.subset,
            difficulty=selected.difficulty,
            expected_rows=selected.expected_rows,
            thinking=selected.benchmark == "AIME2024",
            kivi=False,
        ),
        "kv2": build_eval_command(
            python=args.python,
            model=model,
            data=command_data,
            bits=2,
            evaluator_profile=evaluator_profile,
            output=kv2_path,
            limit=limit,
            num_samples=args.num_samples,
            max_new_tokens=max_new_tokens,
            device=args.device,
            seed=args.seed,
            dtype=args.dtype,
            attn_implementation=args.attn_implementation,
            evaluator_script=selected.evaluator_script,
            subset=selected.subset,
            difficulty=selected.difficulty,
            expected_rows=selected.expected_rows,
            thinking=selected.benchmark == "AIME2024",
            kivi=True,
        ),
        "kv4": build_eval_command(
            python=args.python,
            model=model,
            data=command_data,
            bits=4,
            evaluator_profile=evaluator_profile,
            output=kv4_path,
            limit=limit,
            num_samples=args.num_samples,
            max_new_tokens=max_new_tokens,
            device=args.device,
            seed=args.seed,
            dtype=args.dtype,
            attn_implementation=args.attn_implementation,
            evaluator_script=selected.evaluator_script,
            subset=selected.subset,
            difficulty=selected.difficulty,
            expected_rows=selected.expected_rows,
            thinking=selected.benchmark == "AIME2024",
            kivi=False,
        ),
    }

    print(f"[kv-gate] profile: {selected.description}")
    print(f"[kv-gate] model: {model}")
    print(f"[kv-gate] target training data: {training_data}")
    print(f"[kv-gate] held-out {selected.benchmark} data: {data}")
    print(f"[kv-gate] evaluator profile: {evaluator_profile}")
    print(
        "[kv-gate] required order: unquantized reference -> KV2 -> "
        "(KV4 only if KV2 < one third of reference)"
    )

    if not args.run:
        for name in ("fp", "kv2", "kv4"):
            print(
                f"[kv-gate] {name} command: "
                + " ".join(_shell_quote(str(item)) for item in commands[name])
            )
    else:
        # Explicitly run FP and KV2 in order, even when an old report happens
        # to exist.  Existing artifacts are only reused when --run is absent;
        # this prevents a stale report from silently satisfying a new gate.
        output_dir.mkdir(parents=True, exist_ok=True)
        for name in ("fp", "kv2"):
            path = fp_path if name == "fp" else kv2_path
            _assert_output_safe(path, args.overwrite)
            _run_one(commands[name], env=_child_environment())

    if not fp_path.is_file() or not kv2_path.is_file():
        # Dry-run is intentionally non-mutating and useful before model files
        # are mounted.  It does not pretend that a scientific decision exists.
        payload = {
            "gate_schema_version": GATE_SCHEMA_VERSION,
            "decision_status": "planned",
            "primary_kv_bits": None,
            "profile": args.profile,
            "model": str(model),
            "training_data": str(training_data),
            "eval_data": str(data),
            "benchmark": selected.benchmark,
            "reference_dtype": selected.reference_dtype,
            "subset": selected.subset,
            "difficulty": selected.difficulty,
            "expected_rows": selected.expected_rows,
            "evaluator_profile": evaluator_profile,
            "required_order": ["unquantized_reference", "kv2", "kv4_conditional"],
            "threshold": args.threshold,
            "reports": {"fp": str(fp_path), "kv2": str(kv2_path), "kv4": str(kv4_path)},
            "reason": "run the unquantized reference and KV2 reports before applying the gate rule",
        }
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    fp_payload = load_report(
        fp_path,
        label="fp",
        expected_bits=32,
        expected_dtype=selected.reference_dtype,
    )
    kv2_payload = load_report(kv2_path, label="kv2", expected_bits=2)
    kv4_payload: dict[str, Any] | None = None
    preliminary = choose_primary_bits(fp_payload, kv2_payload, threshold=args.threshold)
    should_run_kv4 = preliminary["decision_status"] == "needs_kv4" or args.always_evaluate_kv4
    if should_run_kv4 and args.run:
        # A pre-existing KV4 file is dangerous here: the FP/KV2 pair may be
        # from a new seed/model while the conditional KV4 artifact belongs to
        # an older gate.  Refuse to reuse it unless the caller explicitly
        # opted into replacement, rather than silently reporting a stale
        # primary result.
        if kv4_path.is_file() and not args.overwrite:
            raise RuntimeError(
                f"refusing to reuse existing conditional KV4 report {kv4_path}; "
                "pass --overwrite to rerun it"
            )
        _assert_output_safe(kv4_path, args.overwrite)
        _run_one(commands["kv4"], env=_child_environment())
    if kv4_path.is_file():
        kv4_payload = load_report(kv4_path, label="kv4", expected_bits=4)

    decision = choose_primary_bits(
        fp_payload,
        kv2_payload,
        kv4_payload,
        threshold=args.threshold,
    )
    decision.update(
        {
            "profile": args.profile,
            "profile_description": selected.description,
            "model": str(model),
            "training_data": str(training_data.resolve()),
            "eval_data": str(data.resolve()),
            "evaluator_profile": evaluator_profile,
            "required_order": ["fp32", "kv2", "kv4_conditional"],
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "decision_ready": decision["primary_kv_bits"] is not None,
            "summary_path": str(summary_path.resolve()),
        }
    )
    if args.run or args.summary is not None:
        if args.run and summary_path.exists() and not args.overwrite:
            raise RuntimeError(
                f"refusing to overwrite existing gate summary {summary_path}; pass --overwrite"
            )
        _write_json_atomic(summary_path, decision)
        print(f"[kv-gate] wrote decision: {summary_path}")
    print(
        "[kv-gate] decision: "
        f"status={decision['decision_status']} primary_kv_bits={decision['primary_kv_bits']} "
        f"kv2/fp={decision['kv2_over_fp_ratio']} reason={decision['reason']}"
    )
    if decision["quality_warnings"]:
        for warning in decision["quality_warnings"]:
            print(f"[kv-gate] quality warning: {warning}")
    if args.json:
        print(json.dumps(decision, ensure_ascii=False, indent=2))
    # A missing KV4 is an intentional incomplete gate, not a command error;
    # return 2 so Slurm/CI can prevent a primary training launch until it is
    # supplied, while still allowing dry-run planning above to return 0.
    return 0 if decision["decision_ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
