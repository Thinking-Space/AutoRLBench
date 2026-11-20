"""One table and one set of parsers for every environment variable we own.

Why this exists
---------------
Five separate incidents, all the same shape: an environment variable was not
set, a default silently decided which experiment ran, and nothing printed.

* unset ``VLLM_OSCAR_{K,V}_ROTATION_PATH`` -> ``oscar_fake_quant`` returned
  ``None`` and the arm quantized in the identity basis.
* ``EVAL_ENFORCE_EAGER`` defaulted to 0 in the eval sbatch -> every OSCAR arm
  scored ~0 instead of 0.44-0.50, for months.
* ``QAT_OSCAR_STAGING_PROBE="0"`` -> ``"0"`` is a truthy string, so the
  documented off switch left the probe running in j104404 and j104820.
* ``qat_e2e.sh --windowed`` defaulted to 0 -> the sink64+recent256 window was
  silently dropped from the *loss*.
* unset ``TEACHER_PATH`` -> ``preflight_mopd.py:912`` fell back to
  ``profile.teacher``, so every 4B and 8B on-policy arm since j104715 distilled
  from DeepSeek-R1-Distill-Qwen-7B instead of from itself. That one invalidated
  a published conclusion.

The rule
--------
==============  ==================  =========================
tier            unset               set but unparseable
==============  ==================  =========================
``REQUIRED``    **raise**           **raise**
``DEFAULTED``   default, echoed     **raise**
==============  ==================  =========================

Never fall back after a *parse* failure. A bad value is a typo the operator
believes took effect; substituting a default is worse than crashing. That is
the bug in the helper this module replaces, which was copied into four files::

    def _env_int(name, default):
        try:    return int(os.environ.get(name, default))
        except (TypeError, ValueError): return default

``VLLM_OSCAR_SINK_TOKENS=64x`` silently became the default there.

Reading a name that is not in ``REGISTRY`` also raises. That is the part that
makes the policy survive the next module: you cannot add a quiet default
without adding a row here first.

Echo
----
Each variable prints one ``[env] NAME = value (env|default)`` line the first
time it is resolved in a process, and no more. Deliberately not a manifest
file and not cross-process plumbing -- it goes wherever the process's stdout
goes, which for the Ray workers is the same job log.
"""

from __future__ import annotations

import glob
import os
import sys
from typing import Any, Callable, Dict, Optional, Tuple

__all__ = [
    "EnvError",
    "REQUIRED",
    "DEFAULTED",
    "REGISTRY",
    "env_str",
    "env_int",
    "env_float",
    "env_bool",
    "env_path",
    "env_csv",
    "env_optint",
    "env_optfloat",
    "require_all",
    "resolved",
]

REQUIRED = "required"
DEFAULTED = "defaulted"

# Underscore-delimited components that mean "never print this value".
#
# Matched as whole components, not as substrings: "TOKEN" in
# VLLM_OSCAR_SINK_TOKENS and VLLM_OSCAR_RECENT_TOKENS is a window size, and a
# substring test redacted both of them as if they were credentials -- which
# hides exactly the two numbers the window incidents were about.  The plural
# TOKENS is therefore not in this set; SWANLAB_API_KEY and DEEPSEEK_API_KEY are.
_SECRET_PARTS = frozenset(
    {"KEY", "TOKEN", "SECRET", "PASSWORD", "PASSWD", "CREDENTIAL", "CREDENTIALS"}
)


def _is_secret(name: str) -> bool:
    return bool(_SECRET_PARTS & set(name.upper().split("_")))


class EnvError(RuntimeError):
    """Raised for any broken environment-variable contract."""


# --------------------------------------------------------------------------
# Parsers.  Each raises ValueError on bad input; the caller turns that into an
# EnvError carrying the variable name and its help text.
# --------------------------------------------------------------------------

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}


def _p_str(raw: str) -> str:
    return raw


def _p_int(raw: str) -> int:
    return int(raw.strip())


def _p_float(raw: str) -> float:
    return float(raw.strip())


def _p_bool(raw: str) -> bool:
    """Only an explicit spelling.  Truthiness is what let ``"0"`` mean on."""
    v = raw.strip().lower()
    if v in _TRUE:
        return True
    if v in _FALSE:
        return False
    raise ValueError(
        f"expected one of {sorted(_TRUE | _FALSE)}, got {raw!r} "
        "(a non-empty string is NOT automatically true here)"
    )


def _p_path(raw: str) -> str:
    """A path that must exist and be readable.

    An unreadable rotation file currently degrades to the identity rotation,
    which is a different quantizer wearing OSCAR's name.
    """
    p = raw.strip()
    if not p:
        raise ValueError("empty path")
    if not os.path.exists(p):
        raise ValueError(f"path does not exist: {p}")
    if not os.access(p, os.R_OK):
        raise ValueError(f"path is not readable: {p}")
    return p


def _p_csv(raw: str) -> list[str]:
    return [s.strip() for s in raw.split(",") if s.strip()]


def _p_optint(raw: str) -> Optional[int]:
    """An int, or None for "not specified".

    Several knobs mean "let the caller decide" when absent -- the staging
    arena width falls back to the engine's ``max_num_seqs``, the OSCAR group
    size falls back to the scheme's own. ``None`` says that explicitly instead
    of encoding it as a magic 0 or -1.
    """
    return None if not raw.strip() else int(raw.strip())


def _p_optfloat(raw: str) -> Optional[float]:
    return None if not raw.strip() else float(raw.strip())


_PARSERS: Dict[str, Callable[[str], Any]] = {
    "str": _p_str,
    "int": _p_int,
    "float": _p_float,
    "bool": _p_bool,
    "path": _p_path,
    "csv": _p_csv,
    "optint": _p_optint,
    "optfloat": _p_optfloat,
}


# --------------------------------------------------------------------------
# The table.  NAME -> (tier, kind, help, default)
#
# REQUIRED  -- changes what a number MEANS.  The operator must type it.
# DEFAULTED -- infrastructure, bitwise-free performance, or diagnostics.
#
# `default` is ignored for REQUIRED rows and must be a *string* for DEFAULTED
# rows: it goes through the same parser as an environment value, so a default
# that cannot be parsed is caught by the unit tests rather than in a run.
# --------------------------------------------------------------------------

_R = REQUIRED
_D = DEFAULTED

REGISTRY: Dict[str, Tuple[str, str, str, Optional[str]]] = {}


def _default_qat_root() -> str:
    return os.environ.get("AUTORLBENCH_WORKSPACE", "/workspace")


def _reg(name: str, tier: str, kind: str, help_: str, default: Optional[str] = None) -> None:
    if name in REGISTRY:
        raise EnvError(f"duplicate registry entry for {name}")
    if tier == DEFAULTED and default is None:
        raise EnvError(f"{name}: DEFAULTED rows need a default")
    if tier == REQUIRED and default is not None:
        raise EnvError(f"{name}: REQUIRED rows must not carry a default")
    if kind not in _PARSERS:
        raise EnvError(f"{name}: unknown kind {kind!r}")
    REGISTRY[name] = (tier, kind, help_, default)


# ---- quantizer identity and geometry -------------------------------------
#
# VLLM_OSCAR_QUANT_ALL_LAYERS is deliberately ABSENT. Commit 0cfd9c8 abolished
# it: the boundary-layer set is a property of the scheme, and the knob existed
# only to dodge a failure mode. turboquant_fake_quant reads the name directly
# in order to REFUSE when it is set. Adding a row here would resurrect it.
_reg("QAT_KV_SCHEME", _R, "str", "KV quantizer selected for this arm (oscar_int2, k2v3, ...)")
_reg("VLLM_OSCAR_K_ROTATION_PATH", _R, "path",
     "OSCAR K rotation .pt, fitted per checkpoint; unset is silently the identity")
_reg("VLLM_OSCAR_V_ROTATION_PATH", _R, "path",
     "OSCAR V rotation .pt, fitted per checkpoint; unset is silently the identity")
_reg("VLLM_OSCAR_SINK_TOKENS", _R, "int", "BF16 sink window; part of the quantizer definition")
_reg("VLLM_OSCAR_RECENT_TOKENS", _R, "int",
     "BF16 recent window; a wider window closes the gap instead of training, removing the object of QAT")
_reg("VLLM_OSCAR_K_CLIP_RATIO", _R, "float", "OSCAR K clip ratio")
_reg("VLLM_OSCAR_V_CLIP_RATIO", _R, "float", "OSCAR V clip ratio")
_reg("VLLM_OSCAR_EXTRA_SKIP_LAYERS", _R, "str",
     "extra native-dtype layers on top of OSCAR's boundary set; '' means none")
_reg("QAT_OSCAR_WINDOWED_ATTN", _R, "bool",
     "make the TRAINING fake quantizer honour serving's window; changes what the loss optimizes")
_reg("QAT_OSCAR_IDENTITY_ROTATION", _R, "bool", "identity-rotation ablation; must be named, never inherited")
_reg("VLLM_OSCAR_BLOCK_SIZE", _R, "int",
     "KV page size; it sets the quantization block, so it is part of the scheme, not a tuning knob")

# ---- alternative quantizers (off the main line, but still arm-defining) ----
_reg("QAT_WA_SCHEME", _D, "str", "weight+activation fake-quant scheme; '' means none", "")
_reg("QAT_WA_STATE", _D, "str", "path to the W/A calibration state; '' means none", "")
_reg("QAT_HYBRID_FAKEQUANT", _D, "bool", "hybrid K/V quantizer; OFF the main line", "0")
_reg("QAT_HYBRID_BITS", _D, "int", "hybrid quantizer width", "2")
_reg("QAT_HYBRID_K_SCHEME", _D, "str", "hybrid K axis: channel | oscar", "channel")
_reg("QAT_HYBRID_V_SCHEME", _D, "str", "hybrid V axis: oscar | token", "oscar")
_reg("QAT_HYBRID_SINK_TOKENS", _D, "int", "hybrid BF16 sink window", "64")
_reg("QAT_HYBRID_RECENT_TOKENS", _D, "int", "hybrid BF16 recent window", "256")
_reg("QAT_KIVI_FAKEQUANT", _D, "bool", "KIVI baseline quantizer", "0")
_reg("QAT_KIVI_BITS", _D, "int", "KIVI width", "2")
_reg("QAT_KIVI_VALUE_GROUP", _D, "int", "KIVI value group size", "32")
_reg("KIVI_ROOT", _D, "str", "checkout of the KIVI reference implementation; '' means auto-detect", "")
_reg("QAT_NVFP4_ROLLOUT", _D, "bool", "serve the rollout from an NVFP4 weight-quantized copy", "0")
_reg("QAT_NVFP4_ROLLOUT_MODEL", _D, "str", "path to that NVFP4 copy", "")
_reg("QAT_KV_BITS", _D, "optint", "legacy Qwen2 KV width; None means the scheme decides", "")
_reg("QAT_KV_GROUP_SIZE", _D, "int", "legacy Qwen2 KV group size; -1 is per-token", "-1")
_reg("VLLM_OSCAR_GROUP_SIZE", _D, "optint",
     "override the scheme's OSCAR group size; None keeps the scheme's own", "")
_reg("QAT_OSCAR_V_STRIDE_FIX", _D, "bool",
     "stage V through V's strides; OFF reproduces the port bug that voided every "
     "pre-2026-09-16 Qwen3 OSCAR number, so it is only ever off deliberately", "1")
_reg("QAT_TEACHER_REPLAY", _D, "str", "teacher-replay rollout spec; '' means off", "")

# ---- objective ------------------------------------------------------------
_reg("QAT_PROBE_OBJECTIVE", _R, "str", "both | kl | mse | nopg -- which loss terms are active")
_reg("QAT_BLOCK_ALIGN_STATIC_REF", _R, "bool", "align against a frozen reference forward")
_reg("QAT_SKIP_TEACHER_ENTROPY", _R, "bool", "skip the teacher-entropy metric family")
_reg("QAT_GRAD_PROBE", _R, "bool",
     "LR=0 diagnostic: consumes the backward and leaves one component's gradient installed; corrupts training if on")
_reg("QAT_FULLKL_COEF", _R, "float", "full-vocabulary KL coefficient")
_reg("QAT_FULLKL_DIRECTION", _R, "str", "reverse | forward | teacher_top1_ratio | head_hinge")
_reg("QAT_FULLKL_W_MAX", _R, "float", "upper bound for adaptive full-KL forward weight")
# DEFAULTED, not REQUIRED, and the default is the old behaviour: two arms were
# already running when this landed and they resolve the config at their first
# actor step, so a REQUIRED knob would have killed them mid-run.  0 = the
# forward sum runs over the whole vocabulary; k > 0 restricts it to the
# teacher's top-k, which is the only thing that stops the mode-covering pull on
# the tail.  Launchers state it explicitly regardless.
# rho and lambda define the head_hinge objective.  DEFAULTED here only so that
# arms running other directions -- which resolve the whole config regardless --
# do not die on a knob they never use; FullVocabKLConfig REFUSES a head_hinge
# arm whose rho is not > 1, which is where the requirement is really enforced.
_reg("QAT_FULLKL_RHO", _D, "float",
     "head_hinge slack: no penalty until the student is rho times under the teacher", "0")
_reg("QAT_FULLKL_LAMBDA", _D, "float", "weight on the head_hinge term", "1.0")
# rho annealing.  Both DEFAULTED to 0 = off, so every existing arm resolves
# them and is bitwise unchanged.  The pair is validated in FullVocabKLConfig:
# a non-zero ANNEAL_STEPS on a non-head_hinge direction, or a RHO_END <= 1,
# raises there rather than silently annealing into a plain forward KL.
_reg("QAT_FULLKL_RHO_END", _D, "float",
     "head_hinge rho at the end of the anneal; 0 = no annealing", "0")
_reg("QAT_FULLKL_RHO_ANNEAL_STEPS", _D, "int",
     "steps over which rho goes RHO -> RHO_END linearly, then held; 0 = off", "0")
_reg("QAT_FULLKL_TOPK", _D, "int",
     "restrict the forward KL to the teacher's top-k; 0 = full vocabulary", "0")
_reg("QAT_FULLKL_STUDENT_TEMPERATURE", _R, "float", "student temperature in the full-vocab KL")
_reg("QAT_FULLKL_TEACHER_TEMPERATURE", _R, "float", "teacher temperature in the full-vocab KL")
_reg("QAT_FULLKL_CHUNK", _D, "int",
     "time-chunk width for the full-vocab KL; value-neutral, memory only", "256")
_reg("QAT_OPD_COEF", _D, "optfloat",
     "override the OPD coefficient; None keeps the value the hydra config already names", "")
_reg("QAT_OPD_TEACHER_IS", _D, "float",
     "teacher-replay importance cap; 0 disables the correction, which is how "
     "pg_clipfrac and ppo_kl were identically 0.0 for eight runs", "0")
_reg("QAT_OPD_POSITION_RAMP", _D, "str", "position-dependent OPD weight spec; '' means uniform", "")
_reg("QAT_OPD_POSITION_RAMP_NORMALIZE", _D, "bool", "renormalize the position ramp to mean 1", "0")
_reg("QAT_OPD_OVERLAP_RATCHET", _D, "str", "overlap-ratchet spec; '' means off", "")
_reg("QAT_OPD_OVERLAP_RATCHET_NORMALIZE", _D, "bool", "renormalize the ratchet weight", "0")
_reg("QAT_OPD_OVERLAP_THRESHOLD", _D, "float", "overlap threshold the ratchet compares against", "0.6")

# ---- eval protocol --------------------------------------------------------
_reg("EVAL_ENFORCE_EAGER", _R, "bool",
     "OSCAR must eval eager: compiled scored 0.0167 against eager's 0.4417")
_reg("QAT_OSCAR_FULL_CUDAGRAPH", _R, "bool",
     "full decode graphs; bitwise-free for OSCAR but has produced 0.0117 vs 0.559 when mismatched")
_reg("VLLM_OSCAR_BF16_TAIL_MODE", _R, "int",
     "tail mode 2 is NOT bitwise-equivalent, so it is a between-arms decision, not a perf knob")
_reg("QAT_FORCE_ATTENTION_BACKEND", _R, "str", "pin the attention backend; '' means vLLM chooses")
_reg("EVAL_NO_INSTRUCTION", _D, "bool",
     "drop the instruction block from the prompt; changes what is measured, so "
     "an arm that uses it must say so in its name", "0")

# ---- gates whose unsafe direction is the default --------------------------
_reg("QAT_KV_REQUIRE_QUANTIZED_ROLLOUT", _R, "bool", "refuse to train if the rollout is not quantized")
_reg("MOPD_ALLOW_ROLLOUT_POLICY_MISMATCH", _R, "bool", "allow actor/rollout policy mismatch")
_reg("MOPD_ROLLOUT_KV_VERIFY_ENFORCE", _D, "bool",
     "hard-fail when the rollout KV verification finds native dtype; default on", "1")
_reg("MOPD_ROLLOUT_KV_VERIFICATION_RECORD", _D, "str",
     "where the rollout KV verification writes its record; '' means do not write", "")
_reg("MOPD_ROLLOUT_LOGPROBS", _D, "str", "rollout logprob mode override; '' means the profile decides", "")

# ---- infrastructure, performance, diagnostics -----------------------------
#
# Two rows below reconcile a default that two call sites disagreed about.
# VLLM_OSCAR_GQA_DECODE and VLLM_OSCAR_PER_REQUEST_STAGING both read as OFF at
# the patch site and were exported as ON by preflight, so the effective value
# depended on which process you asked. The registry picks the training value;
# the eval sbatch exports the other one explicitly, so no path changes.
_reg("QAT_OSCAR_PLUGIN", _D, "bool", "install the qat_kv vLLM plugin at all", "1")
_reg("QAT_OSCAR_FAST_QUANTILE", _D, "bool", "bitwise-identical quantile; 11.5% of a CPU-bound decode step", "1")
_reg("QAT_OSCAR_STEP_CACHE", _D, "bool", "cache layer-invariant decode geometry per step", "1")
_reg("QAT_OSCAR_STEP_CACHE_CHECK", _D, "bool", "re-derive cached geometry and compare; slow, for debugging", "0")
_reg("QAT_OSCAR_STAGING_PROBE", _D, "int", "staging-coverage sampling interval; 0 is off", "0")
_reg("QAT_OSCAR_REPLAY_PROBE", _D, "int", "post-step arena probe sampling interval; 0 is off", "0")
_reg("VLLM_OSCAR_PER_REQUEST_STAGING", _D, "bool", "per-request BF16 staging arena", "1")
_reg("VLLM_OSCAR_STAGE_MAX_SEQS", _D, "optint",
     "staging arena width; None falls back to the engine's max_num_seqs, which "
     "inside the EngineCore worker can be a DEFAULT VllmConfig (jobs 52990-52993)", "")
_reg("VLLM_OSCAR_STAGE_DEBUG", _D, "bool", "per-step staging dump", "0")
_reg("VLLM_OSCAR_SINK_DIAG", _D, "bool", "sink-coverage diagnostic", "0")
_reg("VLLM_OSCAR_GQA_DECODE", _D, "bool", "fused GQA decode kernel", "1")
_reg("VLLM_OSCAR_GQA_VERIFY", _D, "bool", "self-check the GQA kernel against the reference, once", "1")
_reg("VLLM_OSCAR_GQA_VERIFY_MAX_TRIES", _D, "int",
     "give up self-checking after this many dispatches; 8192 is ~290 decode "
     "steps at 28 layers, chosen because 256 gave up after ~9 and every run "
     "logged UNVERIFIED for a kernel that was fine (job 62812)", "8192")
_reg("VLLM_OSCAR_DECODE_KERNEL", _D, "str", "GQA kernel variant; '' picks the default", "")
_reg("VLLM_OSCAR_GQA_BLOCK_KV", _D, "optint", "GQA kernel BLOCK_KV; None = per-variant default", "")
_reg("VLLM_OSCAR_GQA_NUM_WARPS", _D, "int", "GQA kernel num_warps", "1")
_reg("VLLM_OSCAR_GQA_NUM_STAGES", _D, "int", "GQA kernel num_stages", "1")
_reg("VLLM_OSCAR_GQA_SPLITS", _D, "int", "GQA kernel max_num_kv_splits", "128")
_reg("QAT_TQ_DECODE_TUNE", _D, "bool",
     "enable qat_kv/tq_decode_tune.py's k2v3 decode kernel retune; "
     "numerics must be reverified if the BLOCK_KV/warps/stages defaults change", "0")
_reg("QAT_TQ_BLOCK_KV", _D, "int",
     "TurboQuant k2v3 decode stage-1 kernel BLOCK_KV; 4 is the shared "
     "install's unturned default (qat_kv/tq_decode_tune.py)", "4")
_reg("QAT_TQ_NUM_WARPS", _D, "int", "TurboQuant k2v3 decode stage-1 kernel num_warps", "1")
_reg("QAT_TQ_NUM_STAGES", _D, "int", "TurboQuant k2v3 decode stage-1 kernel num_stages", "1")
_reg("QAT_TQ_DECODE_V2", _D, "bool",
     "swap k2v3 decode stage-1 for qat_kv/tq_decode_v2.py's ported upstream "
     "v2 kernel (grouped-Q grid, tensor-core tl.dot, vectorized pair LUT, "
     "exp2); unlike QAT_TQ_DECODE_TUNE this changes the accumulation dtype "
     "(fp16 tensor core vs fp32 scalar), so numerics must be verified by a "
     "greedy token diff, never assumed", "0")
_reg("QAT_TQ_DECODE_LUT", _D, "str",
     "k2v3 2-bit key-dequant source transform (qat_kv/tq_decode_lut.py): "
     "'select' = centroid gather -> 4-way select tree (MEASURED WORSE, "
     "127.0 vs 102.6 ms), 'onebyte' = drop the second byte load of the "
     "MSE-index stitch, which a 2- or 4-bit field never needs, 'dedup' = "
     "load each packed byte once and fan it out in registers instead of "
     "fetching it four times, 'both' = select+onebyte, 1 = both, '' = off. "
     "Every variant is bitwise identical to stock; "
     "verify_kernel.py --patch lut asserts that rather than assuming it", "")
_reg("QAT_FSDP_SHARDING_STRATEGY", _D, "str", "override verl's FSDP sharding strategy; '' keeps verl's", "")
_reg("QAT_FSDP_NO_SYNC", _D, "bool", "skip the no_sync context on gradient-accumulation steps", "0")
_reg("QAT_ROLLOUT_MAX_SEQS", _D, "optint", "rollout max_num_seqs override; None keeps the profile's", "")
_reg("QAT_ROLLOUT_MAX_BATCHED_TOKENS", _D, "optint", "rollout max_num_batched_tokens override", "")
_reg("QAT_OPD_WORKER_CACHE", _D, "bool", "cache teacher outputs in the worker across micro-batches", "0")
_reg("QAT_KV_FORCE_WORKER_PATCH", _D, "str", "force the worker-side KV patch; '' means auto", "")
_reg("QAT_KV_VLLM_EXTERNAL_LAUNCHER", _D, "str", "marker set by the external-launcher path", "")
_reg("QAT_DATA_METRICS_EVERY", _D, "int", "interval for the EXPENSIVE metric family; length metrics are always on", "1")
_reg("QAT_DIAG_EVERY", _D, "int", "diagnostic dump interval", "10")
_reg("QAT_DATA_ROOT", _D, "str", "root of the shared QAT data tree", _default_qat_root())
_reg("MOPD_VENV_BIN", _D, "str", "bin/ of the venv preflight should launch into; '' means auto-detect", "")
_reg("MOPD_PREFLIGHT_PYTHON", _D, "str", "python preflight re-execs into; '' means auto-detect", "")
_reg("MOPD_RAY_PATH", _D, "str", "PATH handed to the Ray workers; '' means inherit", "")
_reg("MOPD_SWANLAB_AUTOKEY", _D, "str", "file holding the SwanLab key (line 6)", "~/.config/qat/swanlab.key")
_reg("MOPD_HYDRA_PROBE_TIMEOUT_S", _D, "int", "hydra config probe timeout", "120")


# --------------------------------------------------------------------------
# Resolution
# --------------------------------------------------------------------------

_seen: Dict[str, Any] = {}


def _echo(name: str, value: Any, source: str) -> None:
    shown = f"<redacted len={len(str(value))}>" if _is_secret(name) else repr(value)
    print(f"[env] {name} = {shown} ({source})", file=sys.stderr, flush=True)


def _resolve(name: str, kind: str) -> Any:
    try:
        tier, reg_kind, help_, default = REGISTRY[name]
    except KeyError:
        raise EnvError(
            f"{name} is not in qat_kv/envreg.py's REGISTRY. Add a row saying "
            "which tier it belongs to and what it controls -- an unregistered "
            "read is how a silent default gets reintroduced."
        ) from None
    if reg_kind != kind:
        raise EnvError(
            f"{name} is registered as {reg_kind!r} but was read as {kind!r}; "
            "one of the two is wrong"
        )

    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        if tier == REQUIRED:
            raise EnvError(
                f"{name} is REQUIRED and is not set. {help_}. "
                "It is experiment-defining, so there is deliberately no default: "
                "a silent one has cost this project five incidents."
            )
        raw, source = default, "default"
    else:
        source = "env"

    try:
        value = _PARSERS[kind](raw)
    except ValueError as exc:
        # Both tiers. A set-but-bad value is a typo the operator believes took
        # effect; falling back to the default would hide it.
        raise EnvError(
            f"{name}={raw!r} could not be parsed as {kind}: {exc}. {help_}. "
            "Refusing to fall back to a default -- fix the value."
        ) from None

    if name not in _seen:
        _seen[name] = value
        _echo(name, value, source)
    return value


def env_str(name: str) -> str:
    return _resolve(name, "str")


def env_int(name: str) -> int:
    return _resolve(name, "int")


def env_float(name: str) -> float:
    return _resolve(name, "float")


def env_bool(name: str) -> bool:
    return _resolve(name, "bool")


def env_path(name: str) -> str:
    return _resolve(name, "path")


def env_csv(name: str) -> list[str]:
    return _resolve(name, "csv")


def env_optint(name: str) -> Optional[int]:
    return _resolve(name, "optint")


def env_optfloat(name: str) -> Optional[float]:
    return _resolve(name, "optfloat")


def require_all(names: Optional[list[str]] = None) -> None:
    """Resolve every REQUIRED variable now, so a launch fails in minute 0.

    ``preflight_mopd.py`` calls this before it builds the hydra line. Without
    it the first read happens inside a Ray worker, hours in, if at all.
    """
    targets = names if names is not None else [
        n for n, (tier, *_rest) in REGISTRY.items() if tier == REQUIRED
    ]
    missing = []
    for n in targets:
        tier, kind, _help, _default = REGISTRY[n]
        try:
            _resolve(n, kind)
        except EnvError as exc:
            missing.append(str(exc))
    if missing:
        raise EnvError(
            f"{len(missing)} required environment variable(s) unusable:\n  - "
            + "\n  - ".join(missing)
        )


def resolved() -> Dict[str, Any]:
    """Everything resolved so far in this process, for a banner or a test."""
    return dict(_seen)
