"""Read every experiment we have on disk into typed records.

Two families, because the two are stored completely differently:

* **eval** -- a single JSON per (model, benchmark, KV scheme) written by the
  evaluators in ``scripts/eval_*.py``.  Uniform schema, carries its own
  ``evaluated_at_utc``.  Scattered across ~12 top-level directories under
  ``runs/`` whose names encode nothing a machine can rely on.
* **train** -- a directory per slurm job under ``runs/qat-train/<profile>/``.
  Config lives in ``effective_settings.json``, the KV proof in
  ``rollout_kv_verification.json``, per-step metrics only in the slurm ``.out``
  log under ``r/qat-train/``.

Both are normalised onto a common date key so the archive can be ordered by
when the work actually happened.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import datetime as _dt
import json
import os
from pathlib import Path
import re

def _discover_qat_root() -> Path:
    """Locate the data root that holds ``runs/``.

    The root is under a per-user directory on ``/shared`` whose name differs
    between environments, so it is discovered rather than hardcoded.  Exactly
    one match is required: zero means we are pointed at the wrong filesystem,
    and more than one means the archive would silently pick a root and produce
    a plausible-looking but partial history.  Set ``QAT_DATA_ROOT`` to skip the
    search.
    """

    override = os.environ.get("QAT_DATA_ROOT")
    if override:
        root = Path(override).expanduser()
        if not (root / "runs").is_dir():
            raise HarvestError(f"QAT_DATA_ROOT={root} has no runs/ directory")
        return root
    candidates = sorted({p.parent for p in Path("/shared").glob("*/qat/runs")})
    if len(candidates) != 1:
        raise HarvestError(
            f"expected exactly one /shared/*/qat/runs, found {len(candidates)}: "
            f"{[str(c) for c in candidates]}; set QAT_DATA_ROOT explicitly"
        )
    return candidates[0]


QAT_ROOT = _discover_qat_root()
RUNS_DIR = QAT_ROOT / "runs"
TRAIN_LOG_DIR = QAT_ROOT / "r/qat-train"

# Directories under runs/ that hold evaluator JSONs.  Listed explicitly rather
# than globbed: runs/ also holds ray spill, smoke junk and one-off diagnostics,
# and a glob that swept those in would put noise in the archive under the guise
# of completeness.
EVAL_ROOTS = (
    "qat-eval-1p5b-base-aime24-avg16",
    "qat-eval-1p5b-base-fp-minerva-avg4",
    "qat-eval-1p5b-base-minerva-avg4",
    "qat-eval-1p5b-multibench",
    "qat-eval-1p5b-nmse-step150-minerva-avg4",
    "qat-eval-1p5b-nmse-step60-avg16",
    "qat-eval-1p5b-nvfp4-aime24-avg32",
    "qat-eval-1p5b-opd-step150-avg16",
    "qat-eval-1p5b-opd-step150-minerva-avg4",
    "qat-eval-1p5b-opd-step60-avg16",
    "qat-eval-n16-A",
    "qat-eval-n16-A25",
    "qat-eval-n16-B-aime_2024",
    "qat-eval-n16-B-aime_2025",
    "qat-eval-pack",
    "kv-baselines",
    "kivi-1p5b-aime24",
    "kivi-1p5b-aime24-fixed",
    "kivi-1p5b-aime24-r0",
    "longbench_v2",
    "aime",
    "hf-bf16-ctrl-aime24",
)

# Keys we plot.  A training log line carries ~200 metrics; the archive keeps a
# readable subset and points at the raw log for the rest.
CURVE_KEYS = (
    "actor/block_align_loss",
    "actor/pg_loss",
    "actor/entropy",
    "actor/grad_norm",
    "actor/kl_loss",
    "critic/rewards/mean",
    "critic/score/mean",
    "response_length/mean",
    "rollout_corr/kl",
    "rollout_corr/training_ppl",
    "val-topk/overlap_ratio",
    "perf/throughput",
)

_STEP_LINE = re.compile(r"step:(\d+) - (.*)$")
# A metric is ``key:value``.  verl prints numpy scalars through ``repr``, so half
# the values arrive as ``actor/pg_loss:np.float64(0.0015)`` and the other half as
# bare floats -- match both.  Missing the wrapped form is not a visible failure:
# the parse simply yields no ``actor/*`` curves, and the archive shows a training
# run with no loss plot as if none had been logged.
_METRIC = re.compile(
    r"([A-Za-z0-9_/\-.]+):(?:np\.\w+\()?(-?[0-9.eE+\-]+|nan|inf|-inf)\)?"
)
_JOB_SUFFIX = re.compile(r"-j(\d+)$")
_GPUS_IN_DIRNAME = re.compile(r"-g(\d+)-")


class HarvestError(RuntimeError):
    """A source file exists but does not have the shape we know how to read."""


@dataclass
class EvalRecord:
    """One evaluator JSON."""

    path: Path
    date: str  # YYYY-MM-DD, UTC
    timestamp: str
    benchmark: str
    model: str
    kv_scheme: str
    accuracy: float | None
    count: int | None
    total_samples: int | None
    n_samples: int | None
    vllm_version: str | None
    status: str
    group: str  # the runs/ directory it came from
    label: str  # path within its group, the only human name these carry
    raw: dict = field(repr=False, default_factory=dict)

    @property
    def slug(self) -> str:
        return _slugify(f"eval-{self.group}-{self.label}")


@dataclass
class TrainRecord:
    """One training job directory."""

    path: Path
    date: str
    timestamp: str
    job_id: str
    profile: str
    kv_scheme: str
    model: str
    objective: str
    updates: int | None
    last_step: int | None
    gpus: int | None
    settings: dict = field(repr=False, default_factory=dict)
    kv_verification: dict | None = field(repr=False, default=None)
    curves: dict[str, list[tuple[int, float]]] = field(repr=False, default_factory=dict)
    checkpoint_evals: list[EvalRecord] = field(default_factory=list)
    stdout: Path | None = None
    stderr: Path | None = None

    @property
    def slug(self) -> str:
        return _slugify(f"train-j{self.job_id}-{self.kv_scheme}-{self.objective}")


def _slugify(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "-", text).strip("-").lower()


def _utc_date(stamp: str) -> tuple[str, str]:
    """('2026-09-02', '2026-09-02T15:13:04Z') from an ISO timestamp."""
    parsed = _dt.datetime.fromisoformat(stamp)
    if parsed.tzinfo is None:
        raise HarvestError(
            f"timestamp {stamp!r} has no timezone; refusing to guess UTC and "
            "file the experiment under a possibly wrong day"
        )
    parsed = parsed.astimezone(_dt.timezone.utc)
    return parsed.strftime("%Y-%m-%d"), parsed.strftime("%Y-%m-%dT%H:%M:%SZ")


def _mtime_date(path: Path) -> tuple[str, str]:
    stamp = _dt.datetime.fromtimestamp(path.stat().st_mtime, _dt.timezone.utc)
    return stamp.strftime("%Y-%m-%d"), stamp.strftime("%Y-%m-%dT%H:%M:%SZ")


def _kv_scheme_of(payload: dict) -> str:
    """What KV cache the evaluation actually used.

    Three different evaluator generations wrote three different key names for
    this, and one of them (the KIVI HF path) records it only as a flag.  Return
    an explicit ``"unknown"`` rather than defaulting to bf16 -- calling an
    unlabelled run full-precision is exactly the mislabelling that would make
    the archive lie.
    """

    for key in ("kv_cache_dtype_requested", "kv_cache_dtype", "kv_scheme"):
        value = payload.get(key)
        if value:
            return str(value)

    # The HF-side evaluators (KIVI, naive-quant, the fp controls) predate the
    # vLLM schema and name the quantizer separately from its width.  Read both:
    # ``quantizer`` alone is not enough -- "qat_kv" covers everything from
    # 32-bit controls to 2-bit runs, and treating it as one scheme would put a
    # full-precision baseline and a 2-bit arm in the same bucket.
    bits = payload.get("kv_bits")
    quant = payload.get("quantizer")
    if bits is not None:
        if not isinstance(bits, int):
            raise HarvestError(f"kv_bits={bits!r} is not an integer")
        if bits >= 16:
            # KV is not quantized at all; ``kv_bits`` 16/32 here just mirrors
            # the model dtype, so name the dtype rather than inventing an
            # "fp32 KV scheme" that nobody configured.
            dtype = payload.get("dtype_resolved") or payload.get("dtype")
            if not dtype or dtype == "auto":
                raise HarvestError(
                    f"kv_bits={bits} (unquantized) but no resolved dtype recorded; "
                    "cannot name the baseline"
                )
            return str(dtype)
        if payload.get("kivi_group_size") is not None:
            return f"kivi-{bits}bit-g{payload['kivi_group_size']}"
        if payload.get("naive_kv"):
            return f"naive-{bits}bit"
        if quant:
            return f"{quant}-{bits}bit"
        # The earliest evaluator generation (runs/aime/, mid-August) recorded a
        # width but never a quantizer name.  The width is real and worth
        # keeping; the algorithm is genuinely unrecorded, so say so in the
        # label rather than assuming it was the one we use today.
        return f"{bits}bit-unspecified-quantizer"

    if payload.get("kivi_enabled"):
        return "kivi"
    quantization = payload.get("kv_cache_quantization")
    if quantization in (None, "none", "auto"):
        return "bf16"
    return "unknown"


def _benchmark_of(payload: dict, path: Path) -> str:
    """Which benchmark this evaluation scored.

    Newer evaluators write ``benchmark``; the HF-side ones write only
    ``profile``.  Neither present means we do not know what the accuracy
    number refers to, and an accuracy without a benchmark is not a result --
    so raise instead of filing it under "unknown".
    """

    for key in ("benchmark", "profile"):
        value = payload.get(key)
        if value:
            return str(value)
    raise HarvestError(f"{path} records an accuracy but names no benchmark or profile")


def _read_eval_json(path: Path, group: str, label: str) -> EvalRecord | None:
    """Parse one evaluator JSON, or return None if it is not one.

    Returns None only for files that are structurally *not* eval results (a
    diagnostic dump, a shard manifest).  A file that looks like an eval result
    but is malformed raises -- silently skipping those is how an archive ends
    up quietly missing the runs that mattered.
    """

    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise HarvestError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        return None
    if "evaluator_schema_version" not in payload:
        return None  # a diagnostic dump, not an evaluation
    if payload.get("status") == "failed" or "error" in payload:
        # Failed evaluations are real history and are kept, but they carry no
        # timestamp, so date them by the file itself.
        date, timestamp = _mtime_date(path)
        return EvalRecord(
            path=path,
            date=date,
            timestamp=timestamp,
            # A run that died in engine startup never learned which benchmark
            # it was going to score, so this one field is genuinely unknown --
            # unlike a successful eval, where an unnamed benchmark means the
            # accuracy is uninterpretable and must raise.
            benchmark=str(payload.get("benchmark") or payload.get("profile") or "unknown"),
            model=str(payload.get("model") or "unknown"),
            kv_scheme=_kv_scheme_of(payload),
            accuracy=None,
            count=None,
            total_samples=None,
            n_samples=None,
            vllm_version=payload.get("vllm_version"),
            status="failed",
            group=group,
            label=label,
            raw=payload,
        )
    if "evaluated_at_utc" not in payload:
        raise HarvestError(
            f"{path} has evaluator_schema_version but no evaluated_at_utc; "
            "cannot place it on the timeline"
        )
    date, timestamp = _utc_date(payload["evaluated_at_utc"])
    sampling = payload.get("sampling") or {}
    return EvalRecord(
        path=path,
        date=date,
        timestamp=timestamp,
        benchmark=_benchmark_of(payload, path),
        model=str(payload.get("model") or "unknown"),
        kv_scheme=_kv_scheme_of(payload),
        accuracy=payload.get("accuracy"),
        count=payload.get("count"),
        total_samples=payload.get("total_samples"),
        n_samples=sampling.get("n"),
        vllm_version=payload.get("vllm_version"),
        status=str(payload.get("status") or "passed"),
        group=group,
        label=label,
        raw=payload,
    )


def harvest_evals(runs_dir: Path = RUNS_DIR) -> list[EvalRecord]:
    records: list[EvalRecord] = []
    for group in EVAL_ROOTS:
        root = runs_dir / group
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*.json")):
            # Label by the path *within* the group, not the file stem: the
            # groups nest (``shards/kv2-6.json``, ``bf16-nothinking-6732/bf16
            # .json``) and stems repeat across those subdirectories.  Keying on
            # the stem alone collapsed 33 distinct evaluations onto 16 archive
            # directories, each silently overwriting the last.
            label = str(path.relative_to(root).with_suffix(""))
            record = _read_eval_json(path, group, label)
            if record is not None:
                records.append(record)
    return records


def parse_curves(log: Path, keys=CURVE_KEYS) -> dict[str, list[tuple[int, float]]]:
    """Pull per-step scalars out of a slurm training log.

    verl prints one line per optimizer step, ``step:N - key:value - key:value``.
    Ray prefixes it with a worker tag, so the step marker is matched anywhere in
    the line rather than anchored.
    """

    wanted = set(keys)
    curves: dict[str, list[tuple[int, float]]] = {}
    for line in log.read_text(errors="replace").splitlines():
        match = _STEP_LINE.search(line)
        if not match:
            continue
        step = int(match.group(1))
        for name, value in _METRIC.findall(match.group(2)):
            if name not in wanted:
                continue
            try:
                number = float(value)
            except ValueError:
                continue
            curves.setdefault(name, []).append((step, number))
    for name, points in curves.items():
        # A resumed or restarted job can print a step twice; keep the last.
        deduped: dict[int, float] = {}
        for step, value in points:
            deduped[step] = value
        curves[name] = sorted(deduped.items())
    return curves


def _settings_value(settings: dict, *names, default=None):
    for name in names:
        if name in settings:
            return settings[name]
    return default


def _gpus_of(settings: dict, run_dir: Path) -> int | None:
    """How many GPUs the job trained on.

    ``effective_settings.json`` does not record this, but the launcher bakes it
    into the run directory name as ``-g<N>-``.  Read it from there rather than
    leaving the field empty: GPU count changes the effective batch size, so a
    README that omits it invites comparing two arms that were not comparable.
    """

    recorded = _settings_value(settings, "n_gpus_per_node", "gpus")
    if recorded is not None:
        return int(recorded)
    match = _GPUS_IN_DIRNAME.search(run_dir.name)
    return int(match.group(1)) if match else None


def _train_kv_scheme(settings: dict) -> str:
    """The KV scheme of a training job, including the unquantized controls.

    ``kv_scheme`` is ``null`` for the fp control arms; that is a deliberate
    "no quantizer", not missing data, so name it rather than printing "None".
    """

    scheme = _settings_value(settings, "kv_scheme")
    if scheme:
        return str(scheme)
    if settings.get("rollout_kv_quantized") is False:
        return "bf16 (fp control)"
    raise HarvestError(
        "training run has no kv_scheme and is not marked unquantized; refusing "
        "to label its KV configuration"
    )


def harvest_trains(
    runs_dir: Path = RUNS_DIR, log_dir: Path = TRAIN_LOG_DIR
) -> list[TrainRecord]:
    records: list[TrainRecord] = []
    train_root = runs_dir / "qat-train"
    if not train_root.is_dir():
        return records
    for profile_dir in sorted(p for p in train_root.iterdir() if p.is_dir()):
        for run_dir in sorted(p for p in profile_dir.iterdir() if p.is_dir()):
            settings_path = run_dir / "effective_settings.json"
            if not settings_path.is_file():
                continue  # an aborted job that never got past preflight
            match = _JOB_SUFFIX.search(run_dir.name)
            if match is None:
                raise HarvestError(
                    f"{run_dir} does not end in -j<jobid>; the archive keys "
                    "training runs by slurm job and cannot place this one"
                )
            job_id = match.group(1)
            settings = json.loads(settings_path.read_text())
            date, timestamp = _mtime_date(settings_path)

            verification_path = run_dir / "rollout_kv_verification.json"
            verification = (
                json.loads(verification_path.read_text())
                if verification_path.is_file()
                else None
            )

            stdout = log_dir / f"train-{job_id}.out"
            stderr = log_dir / f"train-{job_id}.err"
            curves = parse_curves(stdout) if stdout.is_file() else {}
            last_step = max(
                (points[-1][0] for points in curves.values() if points), default=None
            )

            checkpoint_evals = [
                rec
                for rec in (
                    _read_eval_json(p, f"qat-train/{profile_dir.name}", f"j{job_id}-{p.stem}")
                    for p in sorted((run_dir / "eval").glob("*.json"))
                )
                if rec is not None
            ]

            records.append(
                TrainRecord(
                    path=run_dir,
                    date=date,
                    timestamp=timestamp,
                    job_id=job_id,
                    profile=profile_dir.name,
                    kv_scheme=_train_kv_scheme(settings),
                    model=str(_settings_value(settings, "model", default="unknown")),
                    objective=str(
                        _settings_value(
                            settings, "probe_objective", "objective", default="unknown"
                        )
                    ),
                    updates=_settings_value(settings, "max_updates", "updates"),
                    last_step=last_step,
                    gpus=_gpus_of(settings, run_dir),
                    settings=settings,
                    kv_verification=verification,
                    curves=curves,
                    checkpoint_evals=checkpoint_evals,
                    stdout=stdout if stdout.is_file() else None,
                    stderr=stderr if stderr.is_file() else None,
                )
            )
    return records


def harvest(runs_dir: Path = RUNS_DIR, log_dir: Path = TRAIN_LOG_DIR):
    """Every experiment on disk, as (eval records, train records)."""

    return harvest_evals(runs_dir), harvest_trains(runs_dir, log_dir)
