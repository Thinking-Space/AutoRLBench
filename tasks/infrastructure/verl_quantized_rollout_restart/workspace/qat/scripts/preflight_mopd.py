#!/usr/bin/env python3
"""Resolve and preflight the reproducible Open-MOPD/veRL launch.

This module is deliberately a *launcher helper*, not a second RL trainer.  The
actual update, reward construction, and PPO loss are imported from the pinned
Open-MOPD veRL tree in ``/historical/shared/qat/tmp/reference/Open-MOPD``.  Keeping
the launch policy here makes the two agreed profiles (DeepSeek/Skywork and
Qwen-1M/LongRLVR) inspectable without editing Hydra YAML in the reference
checkout.

The command is dry-run by default.  A real launch requires ``--run`` and a
strict preflight.  In particular, JSONL prepared by ``prepare_opd_data.py`` is
not silently handed to veRL: veRL's RLHFDataset reads Parquet.  Use
``--materialize-jsonl`` to create the explicitly named Parquet companion.

Examples
--------
::

    # Print every environment variable and Hydra override, but do not train.
    bash scripts/run_mopd_verl.sh --profile deepseek_skywork

    # First 20-update gate (still requires an explicit --run).
    bash scripts/run_mopd_verl.sh --profile deepseek_skywork --run

    # Main 150-update stage (strict: pass a fresh frozen-model gate and use
    # the validated quantized-rollout prototype).
    bash scripts/run_mopd_verl.sh --profile qwen25_longrlvr --stage main \
        --kv-bits 4 --require-eval-gate --require-quantized-rollout \
        --eval-gate-summary /historical/shared/qat/runs/longbench_v2/kv-gate-qwen25_longrlvr/gate_selection.json \
        --run

    # Deliberate actor-only diagnostic (not an end-to-end quantized rollout).
    bash scripts/run_mopd_verl.sh --profile qwen25_longrlvr --stage main \
        --allow-missing-eval-gate --allow-rollout-policy-mismatch --run

No secret is read into the generated command.  SwanLab authentication is
inherited from ``SWANLAB_API_KEY`` when set by the caller, or loaded only in
memory from line 6 of ``/historical/home/.autokey`` for the child trainer;
its status is always reported as redacted.

On a fresh ``--run`` the launcher also writes
``<output>/effective_settings.json`` immediately before spawning the official
trainer.  The manifest contains the redacted command and dtype/KV policy, but
never the SwanLab token; dry-runs and resumed runs do not overwrite it.
"""

import argparse
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Mapping, Sequence


QAT_ROOT = Path(os.environ.get("QAT_DATA_ROOT", "/historical/shared/qat")).expanduser()
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VERL_DIR = QAT_ROOT / "tmp/reference/Open-MOPD/training/verl"
def _default_python() -> Path:
    """Resolve the interpreter that will actually run the training driver.

    ``MOPD_VENV_BIN`` is what the launchers set to pick an environment, and it
    is the only knob a caller is expected to touch.  Deriving the interpreter
    from it keeps the two in step.  They used to be independent: the sbatch put
    ``MOPD_VENV_BIN`` on ``PATH`` while this constant stayed pinned to the
    0.21.0 environment, so job 11825 launched ``int4_per_token_head`` against a
    vLLM that has no such ``CacheDType`` and pydantic rejected it in 3 minutes.
    Ray actors fork the driver's interpreter, so the ``runtime_env`` PATH that
    did carry the new environment never got a chance to matter.

    ``MOPD_PREFLIGHT_PYTHON`` still wins if set, for the case where the
    preflight interpreter is deliberately not the training one.  Nothing falls
    back silently: an explicit ``MOPD_VENV_BIN`` with no ``python`` in it is an
    error, not a reason to quietly reach for the old environment.
    """

    explicit = os.environ.get("MOPD_PREFLIGHT_PYTHON")
    if explicit:
        return Path(explicit)
    venv_bin = os.environ.get("MOPD_VENV_BIN")
    if venv_bin:
        candidate = Path(venv_bin) / "python"
        if not candidate.is_file():
            raise SystemExit(
                f"MOPD_VENV_BIN={venv_bin} has no 'python'; refusing to fall back to "
                "another environment.  Set MOPD_PREFLIGHT_PYTHON to override."
            )
        return candidate
    return Path("/historical/shared/bitdistiller/envs/vllm-0.19.1-test/bin/python")


DEFAULT_PYTHON = _default_python()
# Keep experiment-only wheels outside the repository and outside the old QAT
# environment.  This directory currently contains SwanLab (and its small
# transitive dependencies) for the pinned Open-MOPD interpreter.  It is put
# first in ``PYTHONPATH`` below so a stale compatibility wheel cannot shadow
# the version used for this run.
MOPD_EXTRA_LIBS = Path("/historical/shared/qat/envs/mopd-pylibs")
# NVIDIA modelopt is not installed into the venv; it lives here and is only
# needed by arms that set --wa-scheme.  Job 22939 died in the Ray worker with
# ModuleNotFoundError because the driver's preflight found the state file but
# the workers had no import path for modelopt itself.
MODELOPT_LIBS = Path("/historical/shared/qat/envs/modelopt-pylibs")
# The native TurboQuant experiment uses a small dependency overlay (Ray,
# msgpack, Hydra, etc.) kept separate from the archived vLLM environment.  It
# must precede the older compatibility wheels so their partial Ray package
# cannot shadow the complete runtime.
TURBOQUANT_TRAIN_OVERLAY = Path(
    os.environ.get(
        "MOPD_TURBOQUANT_TRAIN_OVERLAY",
        "/historical/shared/qat/tmp/turboquant-train-overlay",
    )
).expanduser()
DEFAULT_PYLIBS = Path("/historical/shared/entropy_floor_rlvr/envs/pylibs")
SWANLAB_AUTOKEY = Path(
    os.environ.get("MOPD_SWANLAB_AUTOKEY", "/historical/home/.autokey")
).expanduser()
_SWANLAB_TOKEN_RE = re.compile(r"^[A-Za-z0-9]{21}$")

# These labels are deliberately launcher metadata.  The actual reward and
# PPO implementation remains the pinned Open-MOPD code path.  Keep the old
# descriptive labels as compatibility aliases so existing dashboards and
# scripts can still identify earlier runs without accidentally treating them
# as a second algorithm.
MOPD_OBJECTIVE_NAME = "single_teacher_topk_opd"
MOPD_OBJECTIVE_LEGACY_NAME = "topk_student_weighted_teacher_delta"
MOPD_OBJECTIVE_LEGACY_ALIASES = (
    MOPD_OBJECTIVE_LEGACY_NAME,
    "student_topk_mopd_reward",
    "mopd_topk_ppo",
)
MOPD_REWARD_TOP_K_SOURCE = "student_untruncated"
MOPD_REWARD_TOP_K_SOURCE_LEGACY = "student_before_nucleus_filter"
# The requested experiment evaluates every 20 optimizer updates.  Keep the
# checkpoint schedule aligned so each requested evaluation has a checkpoint.
MOPD_EVAL_CHECKPOINTS_DEFAULT = (20, 40, 60, 80, 100, 120, 140, 160, 180, 200)
# Keep this launcher-side value in lockstep with ``scripts.eval_kv_gate``.
# Importing the evaluator module here would make ``python scripts/preflight_mopd.py``
# depend on package-path details, so the strict check intentionally uses the
# small, stable manifest version directly.
EVAL_GATE_SCHEMA_VERSION = 1
# Actor FSDP contract: request FP32 actor initialization, BF16 forward
# parameters, and FP32 reductions/buffers. The pinned worker's exact
# residency still needs runtime verification; these values are not an
# assertion about teacher arithmetic or a replacement optimizer. FP32
# parameter storage is roughly 2x BF16 before sharding, and FP32 optimizer
# master/moment state can add further peak memory; the actual multiplier
# depends on FSDP sharding/offload.
ACTOR_FSDP_PARAM_DTYPE = "bf16"
ACTOR_FSDP_REDUCE_DTYPE = "fp32"
ACTOR_FSDP_BUFFER_DTYPE = "fp32"

# Native vLLM cache dtypes.  These are passed directly to vLLM's
# ``kv_cache_dtype`` argument; they are distinct from the older QAT/KIVI
# hooks in this repository.
NATIVE_TURBOQUANT_DTYPES = (
    "turboquant_k8v4",
    "turboquant_4bit_nc",
    "turboquant_k3v4_nc",
    "turboquant_3bit_nc",
    "turboquant_k2v3_nc",
    "turboquant_2bit_nc",
    # vLLM's own INT4 per-(token, head) cache, available from 0.25.0.  It is
    # *not* a naive INT4 baseline: it applies a randomized Hadamard rotation
    # before quantizing, which is one of TurboQuant's own devices.  Kept in
    # this tuple because the mechanism is identical -- a native vLLM
    # ``kv_cache_dtype`` with a matching training-side fake quantizer.
    "int4_per_token_head",
)

# Written by the rollout worker (qat_kv.native_tq_verify) once the vLLM engine
# has structurally proven it is running the requested preset.  Preflight only
# publishes the path: the verdict is not knowable before the engine exists.
ROLLOUT_KV_VERIFICATION_FILENAME = "rollout_kv_verification.json"


def _native_tq_dtype(args) -> "str | None":
    """The requested native cache dtype, with ``"none"`` meaning unquantized.

    The CLI option defaults to a quantized preset, so ``"none"`` is the only
    way to ask for a full-precision control run.
    """
    value = getattr(args, "native_turboquant_kv_cache_dtype", None)
    return None if value in (None, "", "none") else value

# Linux's ``sockaddr_un.sun_path`` stores at most 108 bytes including its
# terminating NUL, so Ray's validator accepts at most 107 bytes.  Ray builds
# its local socket names below ``<temp_dir>/session_<date>_<pid>/sockets``;
# keep a conservative, fixed-width session placeholder here so a long output
# or Ray temp path is rejected before a cluster is spawned.
_RAY_AF_UNIX_MAX_BYTES = 107
_RAY_SESSION_PLACEHOLDER = "session_9999-99-99_99-99-99_999999_9999999999"
_RAY_SOCKET_LEAVES = ("raylet", "plasma_store")


def _ray_socket_projection(temp_dir: Path) -> tuple[int, str]:
    """Return worst-case Ray socket byte length and the longest path.

    ``ray._private.node.Node`` creates both ``raylet`` and ``plasma_store``
    sockets below a timestamped session directory.  Count encoded bytes (not
    Python characters) because ``sun_path`` is a byte-limited field.
    """

    base = temp_dir.expanduser().resolve(strict=False)
    projected = [
        base / _RAY_SESSION_PLACEHOLDER / "sockets" / leaf
        for leaf in _RAY_SOCKET_LEAVES
    ]
    longest = max(projected, key=lambda path: len(os.fsencode(str(path))))
    return len(os.fsencode(str(longest))), str(longest)


@dataclass(frozen=True)
class Profile:
    """Filesystem and context defaults for one experiment profile."""

    name: str
    model: Path
    teacher: Path
    prepared_jsonl: Path
    prepared_parquet: Path
    prepared_manifest: Path
    raw_parquet: Path
    expected_rows: int
    prompt_length: int
    response_length: int
    model_context_length: int
    dataset_revision: str
    paper: str
    description: str
    # Evaluation protocol.  These used to be duplicated in every bash consumer,
    # which is how a 1.5B run inherited the 7B's context window, concurrency and
    # tensor-parallel width: values chosen because a 7B at 131k context cannot
    # afford more.  Tensor parallelism alone moved AIME by 1.67 pp on identical
    # problems and seed, so it is part of the protocol, not a tuning knob.
    eval_benchmark: str = "aime_math"
    eval_max_model_len: int = 131072
    eval_max_num_seqs: int = 2
    eval_tensor_parallel_size: int = 2
    eval_num_samples: int = 4
    eval_data: str = ""


def _profiles(root: Path = QAT_ROOT) -> dict[str, Profile]:
    """Return deterministic paths; no network lookup or download occurs."""

    models = root / "models/base"
    processed = root / "data/processed"
    raw = root / "data/raw"
    deepseek = models / "deepseek-r1-distill-qwen-7b"
    qwen = models / "qwen2.5-7b-instruct-1m"
    brorl = root / "models/Nemotron-Research-Reasoning-Qwen-1.5B-brorl"
    return {
        "deepseek_skywork": Profile(
            name="deepseek_skywork",
            model=deepseek,
            # MOPD needs a frozen FP teacher.  The default is the same
            # checkpoint as the student; only the student's actor path gets
            # the optional KV hook.
            teacher=deepseek,
            prepared_jsonl=processed / "skywork_or1_7b_math_difficulty1-15_16k.jsonl",
            prepared_parquet=processed / "skywork_or1_7b_math_difficulty1-15_16k.parquet",
            # The parquet companion carries the exact output hash/schema used
            # by veRL.  The older JSONL manifest remains useful as source
            # provenance but is not the authoritative training artifact.
            prepared_manifest=root / "data/manifests/skywork_or1_7b_math_16k.parquet.manifest.json",
            raw_parquet=raw / "skywork-or1-rl-data/data/math-00000-of-00001.parquet",
            expected_rows=16000,
            # Skywork math prompts are short (max 1,670 tokens in the
            # prepared 16K training split).  Keep a finite training pad width;
            # the no-prompt-limit rule applies only to the 32K evaluator.
            prompt_length=2048,
            response_length=16384,
            model_context_length=131072,
            dataset_revision="1cdedc52e0e2db85fdf252f9be682e63a5a38c33",
            paper="arXiv:2505.22312",
            description="DeepSeek-R1-Distill-Qwen-7B on Skywork OR1 math (difficulty 1..15)",
            eval_benchmark="aime_math",
            eval_max_model_len=131072,
            eval_max_num_seqs=2,
            eval_tensor_parallel_size=2,
            eval_num_samples=4,
            eval_data=str(raw / "aime_2024/data/train-00000-of-00001.parquet"),
        ),
        "qwen25_longrlvr": Profile(
            name="qwen25_longrlvr",
            model=qwen,
            teacher=qwen,
            prepared_jsonl=processed / "longrlvr_qwen25_1m_prompt65536.jsonl",
            prepared_parquet=processed / "longrlvr_qwen25_1m_prompt65536.parquet",
            # The governed manifest describes the exact Parquet artifact that
            # veRL reads.  Keep the converter's transactional suffix instead
            # of the older JSONL/source manifest.
            prepared_manifest=root / "data/manifests/longrlvr_qwen25_1m_prompt65536.parquet.manifest.json",
            raw_parquet=raw / "longrlvr-data/qwen2.5-7b/train_data_64k.parquet",
            expected_rows=16000,
            prompt_length=65536,
            response_length=4096,
            model_context_length=1010000,
            dataset_revision="282447325bc6034eeb39107598fe025ecc2474cd",
            paper="arXiv:2603.02146",
            description="Qwen2.5-7B-Instruct-1M on LongRLVR (training input cap only)",
            eval_benchmark="longbench_v2_medium",
            eval_max_model_len=1010000,
            eval_max_num_seqs=2,
            eval_tensor_parallel_size=2,
            eval_num_samples=4,
            eval_data="",
        ),
        # The 1.5B student. Its evaluation protocol is not the 7B's: it is the
        # one the published 2-bit baseline table was measured with (AIME24
        # avg@32 at TP=1, 33,792-token window, 64 concurrent sequences), so a
        # checkpoint can be compared against those numbers directly.  Reusing
        # the 7B settings silently changed all four of those.
        "qwen_brorl_1p5b": Profile(
            name="qwen_brorl_1p5b",
            model=brorl,
            teacher=brorl,
            # Same math corpus as deepseek_skywork; the file name records the
            # 7B that scored the difficulty labels, not the student it trains.
            prepared_jsonl=processed / "skywork_or1_7b_math_difficulty1-15_16k.jsonl",
            prepared_parquet=processed / "skywork_or1_7b_math_difficulty1-15_16k.parquet",
            prepared_manifest=root / "data/manifests/skywork_or1_7b_math_16k.parquet.manifest.json",
            raw_parquet=raw / "skywork-or1-rl-data/data/math-00000-of-00001.parquet",
            expected_rows=16000,
            prompt_length=512,
            response_length=16384,
            model_context_length=131072,
            dataset_revision="1cdedc52e0e2db85fdf252f9be682e63a5a38c33",
            paper="arXiv:2505.22312",
            description="Nemotron-Research-Reasoning-Qwen-1.5B-brorl on Skywork OR1 math",
            eval_benchmark="aime_math",
            # 32,768 response budget plus prompt headroom; matches the tier table.
            eval_max_model_len=33792,
            eval_max_num_seqs=64,
            eval_tensor_parallel_size=1,
            eval_num_samples=32,
            eval_data=str(raw / "aime_2024/data/train-00000-of-00001.parquet"),
        ),
    }


def _default_ray_root(output: Path, profile: Profile, kv_bits: int, stage: str, updates: int) -> Path:
    """Choose a short, deterministic shared Ray root for a run.

    Ray appends a timestamped session/socket suffix to ``_temp_dir``.  Using
    the human-readable artifact path verbatim (for example
    ``runs/mopd/deepseek_skywork/kv4/gate20``) can exceed Linux's 107-byte
    ``AF_UNIX`` limit before Ray starts.  Keep artifacts in their descriptive
    output directory, but put transient Ray state in a compact sibling keyed
    by an 8-character digest so concurrent profiles cannot collide.
    """

    identity = "|".join(
        (
            str(output.expanduser().resolve(strict=False)),
            profile.name,
            f"kv{kv_bits}",
            stage,
            str(updates),
        )
    )
    digest = hashlib.sha1(identity.encode("utf-8")).hexdigest()[:8]
    # Keep the directory name intentionally tiny: the conservative socket
    # projection below uses a longer-than-usual Ray session placeholder, so a
    # descriptive profile name here would make even the default path fail.
    return QAT_ROOT / "ray" / digest


@dataclass
class Resolved:
    profile: Profile
    model: Path
    teacher: Path
    train: Path
    val: Path
    output: Path
    python: Path
    verl_dir: Path
    py_libs: Path | None
    stage: str
    updates: int
    nodes: int
    gpus: int
    ray_num_cpus: int
    ray_temp_dir: Path
    ray_spill_dir: Path
    legacy_kv_bits: int
    lr: float
    # Training-side fake-quant scheme.  None/'symmetric' keeps the legacy
    # pre-RoPE projection hook; a 'turboquant_*' preset selects the
    # post-RoPE TurboQuant quantizer that matches the vLLM inference path.
    kv_scheme: str | None
    # NVFP4 weight/activation fake-quant (qat_kv.wa_scheme).  Orthogonal to
    # kv_scheme: an arm can quantize the KV cache, the linear layers, or both.
    wa_scheme: str | None
    wa_state: str | None
    # Weight of the quantized-vs-full-precision block alignment term.
    # 0 disables it; it needs *a* quantizer -- kv_scheme or wa_scheme.
    block_align_coef: float
    block_align_layer_every: int
    block_align_keep_fraction: float
    block_align_metric: str
    block_align_last_only: bool
    # Weight of the quantized-K/V vs full-precision-K/V alignment term.
    kv_align_coef: float
    kv_align_layer_every: int
    kv_align_keep_fraction: float
    kv_align_metric: str
    kv_align_align_keys: bool
    kv_align_align_values: bool
    preserve_projection_bias: bool
    sampling_top_k: int
    sampling_top_p: float
    reward_top_k: int
    top_k_strategy: str
    reward_weight_mode: str
    temperature: float
    train_batch_size: int
    micro_batch_size_per_gpu: int
    reward_micro_batch_size_per_gpu: int
    log_prob_micro_batch_size_per_gpu: int
    ppo_mini_batch_size: int
    resume: bool
    resume_path: str | None
    rollout_engine: str
    actor_attention: str
    reward_attention: str
    trust_remote_code: bool
    disable_torch_compile: bool
    swanlab_mode: str
    project_name: str
    experiment_name: str
    extra_overrides: list[str]
    # Launcher policy/diagnostic knobs.  These do not replace any Open-MOPD
    # trainer implementation; they make the effective experiment contract
    # explicit in the generated Hydra command and summary.
    # Defaults keep this value object backwards-compatible for callers that
    # construct ``Resolved`` directly (older notebooks/tests did so), while
    # ``_resolve`` still fills every field from the CLI parser.
    actor_model_dtype: str = "fp32"
    quantize_vllm_rollout: bool = False
    require_quantized_rollout: bool = False
    # A real main run rejects an actor/rollout KV-policy mismatch by default.
    # This explicit escape hatch is for diagnostic actor-only experiments and
    # is never implied by the gate stage or by a dry-run.
    allow_rollout_policy_mismatch: bool = False
    strict_fp32_arithmetic: bool = False
    one_epoch: bool = False
    save_freq: int = 20
    eval_checkpoints: tuple[int, ...] = MOPD_EVAL_CHECKPOINTS_DEFAULT
    # Optional frozen-model FP32/KV2/KV4 selection manifest.  Short gate
    # runs remain backwards-compatible when this is unset; main runs can opt
    # into a hard requirement with --require-eval-gate.
    eval_gate_summary: Path | None = None
    require_eval_gate: bool = False
    # Main runs require the frozen-model selection by default.  This explicit
    # escape hatch is limited to diagnostic runs with no summary at all;
    # supplied-but-invalid summaries remain fail-closed.
    allow_missing_eval_gate: bool = False
    # Exact HF bias-free recompute is an explicit, opt-in path.  The default
    # remains the cheaper post-forward approximate bias preservation.
    exact_bias_free_recompute: bool = False
    kivi_hf_rollout: bool = False
    native_turboquant_kv_cache_dtype: str | None = None
    rollout_gpu_memory_utilization: float = 0.45
    # Capture vLLM decode CUDA graphs.  See --rollout-cuda-graph for why
    # this has to be FULL_DECODE_ONLY rather than the usual PIECEWISE.
    rollout_cuda_graph: bool = True

    @property
    def world_size(self) -> int:
        return self.nodes * self.gpus

    @property
    def gradient_accumulation_steps(self) -> int:
        per_update = self.world_size * max(1, self.micro_batch_size_per_gpu)
        if self.train_batch_size % per_update:
            return -1
        return self.train_batch_size // per_update

    @property
    def actor_kv_hook_expected(self) -> bool:
        """Whether a sub-32-bit actor should install the KV fake/dequant hook."""

        return self.legacy_kv_bits < 32

    @property
    def native_turboquant_rollout(self) -> bool:
        return self.native_turboquant_kv_cache_dtype is not None

    @property
    def hf_rollout_registry_available(self) -> bool:
        """Whether this pinned checkout actually registers ``hf/sync``.

        Merely selecting ``--rollout-engine hf`` must not be reported as a
        quantized rollout: older Open-MOPD checkouts import ``HFRollout`` but
        omit it from ``_ROLLOUT_REGISTRY``.  Keep this source-level check
        dependency-free so it is safe in a dry-run.
        """

        if self.rollout_engine != "hf":
            return False
        registry = self.verl_dir / "verl/workers/rollout/base.py"
        try:
            text = registry.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            return False
        # The pinned fork omits this entry; qat_kv.verl_plugin registers the
        # existing VeRL HFRollout class at worker import time for KIVI runs.
        return bool(
            re.search(r"\(\s*['\"]hf['\"]\s*,\s*['\"]sync['\"]\s*\)", text)
            or (self.kivi_hf_rollout and (REPO_ROOT / "qat_kv/verl_plugin.py").is_file())
        )

    @property
    def vllm_kv_patch_available(self) -> bool:
        """Whether this workspace ships the explicit vLLM Qwen2 patch."""

        return (REPO_ROOT / "qat_kv/vllm_qwen2_patch.py").is_file()

    @property
    def rollout_kv_status(self) -> str:
        """Return an honest pre-build status for the selected rollout path."""

        if self.rollout_engine == "vllm":
            if self.native_turboquant_rollout:
                return f"native_{self.native_turboquant_kv_cache_dtype}_unverified"
            if not self.quantize_vllm_rollout:
                return "unquantized_external_engine"
            if self.legacy_kv_bits >= 32:
                return "vllm_patch_incompatible_with_kv32"
            if not self.vllm_kv_patch_available:
                return "vllm_patch_missing"
            # The actual class patch/model hit is validated inside the Ray
            # worker by qat_kv.verl_plugin; preflight can only record intent.
            return "vllm_patch_requested_unverified"
        if self.rollout_engine == "hf":
            if self.rollout_kv_quantized:
                return "official_kivi_hf" if self.kivi_hf_rollout else "actor_hook_shared"
            return "unverified_hf_registry"
        return "unquantized_external_engine"

    @property
    def rollout_kv_quantized(self) -> bool:
        """Whether the selected rollout engine can share the HF KV hook.

        vLLM/other external engines do not execute the Hugging Face
        ``k_proj``/``v_proj`` hooks by default.  The only exception is the
        explicit, model-specific ``--quantize-vllm-rollout`` prototype; even
        then this value means *expected from the patch*, while the Ray-worker
        plugin publishes post-build verification metadata.
        """

        if self.rollout_engine == "vllm":
            if self.native_turboquant_rollout:
                # Native vLLM owns both sampling and KV-cache execution.  The
                # runtime constructor check below is still required before
                # this expectation becomes verified evidence.
                return True
            # This is an expected capability, not a runtime assertion.  The
            # plugin publishes a post-build hit/validation record and can
            # still fail closed if vLLM spawned a child process unexpectedly.
            return bool(
                self.quantize_vllm_rollout
                and self.actor_kv_hook_expected
                and self.vllm_kv_patch_available
            )
        return bool(
            self.rollout_engine == "hf"
            and self.actor_kv_hook_expected
            and self.hf_rollout_registry_available
            and (self.kivi_hf_rollout or self.legacy_kv_bits >= 32)
        )

    @property
    def rollout_policy_mismatch(self) -> bool:
        if self.native_turboquant_rollout:
            return False
        return self.actor_kv_hook_expected and not self.rollout_kv_quantized

    @property
    def eval_gate_required_for_run(self) -> bool:
        """Whether this planned stage requires a complete selection summary."""

        return bool(
            self.require_eval_gate
            # The historical FP32/KV2/KV4 selector does not describe the
            # native vLLM TurboQuant experiment: the requested policy is a
            # fixed cache dtype (3bit_nc), and both TurboQuant variants are
            # benchmarked independently.  Do not force that old selector on
            # a native-TurboQuant main run; the evaluator reports are checked
            # before launch by the experiment driver.
            or (
                self.stage == "main"
                and not self.allow_missing_eval_gate
                and not self.native_turboquant_rollout
            )
        )

    @property
    def rollout_quantized_required_for_run(self) -> bool:
        """Whether runtime rollout hit verification is required for this plan.

        A strict main run must not silently fall back from the actor's
        sub-32-bit KV path to an unquantized/unverified rollout.  The explicit
        mismatch escape hatch is the only way to keep that actor-only mode;
        an explicit ``--require-quantized-rollout`` remains authoritative.
        """

        return bool(
            self.require_quantized_rollout
            or self.native_turboquant_rollout
            or (
                self.stage == "main"
                and not self.allow_rollout_policy_mismatch
                and self.actor_kv_hook_expected
            )
        )


def _rollout_policy_mismatch_is_fatal(r: Resolved, *, strict: bool) -> bool:
    """Whether the selected actor/rollout KV mismatch blocks this launch.

    The short gate stage and dry-runs may intentionally inspect an actor-only
    setup, so they retain an advisory mismatch.  A real ``main`` stage is
    fail-closed unless the caller explicitly opts into that diagnostic mode.
    ``--require-quantized-rollout`` remains stronger than the escape hatch.
    """

    if not strict or not r.rollout_policy_mismatch:
        return False
    if r.require_quantized_rollout:
        return True
    return r.stage == "main" and not r.allow_rollout_policy_mismatch


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < -1:
        raise argparse.ArgumentTypeError("must be -1 or a non-negative integer")
    return parsed


def _float_01(value: str) -> float:
    parsed = float(value)
    if not 0.0 < parsed <= 1.0:
        raise argparse.ArgumentTypeError("must be in (0, 1]")
    return parsed


def _checkpoint_list(value: str) -> tuple[int, ...]:
    """Parse a deterministic comma-separated external-evaluation schedule."""

    if value is None:
        return MOPD_EVAL_CHECKPOINTS_DEFAULT
    raw = [item.strip() for item in str(value).split(",")]
    if not raw or any(not item for item in raw):
        raise argparse.ArgumentTypeError("must be a comma-separated list of positive integers")
    try:
        values = tuple(sorted({int(item) for item in raw}))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a comma-separated list of positive integers") from exc
    if not values or any(item < 1 for item in values):
        raise argparse.ArgumentTypeError("checkpoint steps must be positive")
    return values


def _manifest_for(path: Path) -> Path | None:
    """Find the companion manifest used by ``prepare_opd_data.py``."""

    candidates = [
        # ``convert_opd_jsonl_to_parquet.py`` writes this transactional
        # companion by default.
        Path(str(path) + ".manifest.json"),
        # The governed-data workflow may keep companions in the shared
        # ``data/manifests`` directory rather than beside a large artifact.
        QAT_ROOT / "data/manifests" / f"{path.name}.manifest.json",
        path.with_suffix(".json"),
        QAT_ROOT / "data/manifests" / f"{path.stem}.json",
        QAT_ROOT / "data/manifests" / f"{path.stem.replace('.jsonl', '')}.json",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _manifest_candidates(path: Path, explicit_manifest: Path | None = None) -> list[Path]:
    """Return unique row-governance manifests in deterministic precedence.

    An explicit profile/governed manifest is authoritative when supplied;
    otherwise the transactional Parquet companion is strongest because it
    describes the exact file veRL will consume.  Legacy discovery is last for
    custom or JSONL paths.  Candidates are retained even when absent so
    callers can explain which fallback was attempted.
    """

    candidates: list[Path] = []

    def add(candidate: Path | None) -> None:
        if candidate is None:
            return
        candidate = candidate.expanduser()
        if candidate not in candidates:
            candidates.append(candidate)

    # The profile's governed manifest is authoritative when present (for the
    # canonical files); malformed/absent files fall through to the companion.
    add(explicit_manifest)
    # ``convert_opd_jsonl_to_parquet.py`` writes this exact companion.  It is
    # the first fallback so a pyarrow-less preflight still has file-local row
    # governance for custom paths.
    add(Path(str(path) + ".manifest.json"))
    add(_manifest_for(path))
    return candidates


def _manifest_rows(path: Path, explicit_manifest: Path | None = None) -> int | None:
    """Read governed row count from a companion or profile manifest.

    Invalid or partially-written candidates are skipped, allowing a valid
    companion/governed manifest later in the precedence list to be used.
    """

    for manifest in _manifest_candidates(path, explicit_manifest):
        if not manifest.is_file():
            continue
        try:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        for key_path in (("output", "rows"), ("rows",), ("stats", "selected_rows")):
            current: Any = payload
            try:
                for key in key_path:
                    current = current[key]
                return int(current)
            except (KeyError, TypeError, ValueError):
                continue
    return None


def _count_jsonl(path: Path, limit: int | None = None) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                count += 1
                if limit is not None and count >= limit:
                    break
    return count


def _parquet_rows(path: Path) -> int | None:
    try:
        import pyarrow.parquet as parquet  # type: ignore

        return int(parquet.ParquetFile(path).metadata.num_rows)
    except Exception:
        return None


def _safe_text(value: str) -> str:
    """Redact values that look like credentials when printing overrides."""

    # Do not redact ordinary model settings such as ``max_response_tokens``
    # or ``algorithm.adv_estimator=token_reward_direct``.  Only a key/name
    # that is conventionally credential-bearing is hidden.
    key = value.split("=", 1)[0] if "=" in value else value
    if re.search(
        r"(?:^|[._-])(api[_-]?key|secret|password|passwd|access[_-]?key|auth[_-]?token)(?:$|[._-])",
        key,
        re.I,
    ):
        return f"{key}=<redacted>" if "=" in value else "<redacted>"
    return value


def _load_swanlab_key(env: dict[str, str] | None = None) -> tuple[dict[str, str], str]:
    """Resolve the cloud credential without ever returning it to the caller's logs.

    The cluster convention is ``swan-lab <21-character-token>`` on line 6 of
    ``~/.autokey``.  We intentionally parse only that line and only place the
    token in the private environment handed to the child trainer.  It is not
    included in the Hydra argv, JSON summary, manifest, or shell command.

    Returns ``(environment, status)`` where status is a redacted diagnostic
    such as ``"inherited"`` or ``"autokey-line6"``.  In offline/disabled mode
    no credential is needed and the input environment is copied unchanged.
    """

    result = dict(os.environ if env is None else env)
    mode = str(result.get("SWANLAB_MODE", "cloud")).strip().lower()
    if mode != "cloud":
        return result, "not-required"

    inherited = result.get("SWANLAB_API_KEY", "").strip()
    if inherited:
        return result, "inherited"

    selected: str | None = None
    try:
        # Read only through line 6.  Apart from being cheaper, this avoids
        # copying unrelated credential material from a shared ``.autokey``
        # file into Python memory or an exception object.
        with SWANLAB_AUTOKEY.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if line_number == 6:
                    selected = line
                    break
    except (OSError, UnicodeError) as exc:
        raise RuntimeError(
            f"SwanLab cloud mode requires SWANLAB_API_KEY or readable {SWANLAB_AUTOKEY}"
        ) from exc
    if selected is None:
        raise RuntimeError(f"{SWANLAB_AUTOKEY} has no line 6; refusing cloud launch")
    fields = selected.strip().split()
    if len(fields) != 2 or fields[0] != "swan-lab" or not _SWANLAB_TOKEN_RE.fullmatch(fields[1]):
        raise RuntimeError(
            f"{SWANLAB_AUTOKEY}: line 6 must be 'swan-lab <21 alphanumeric chars>'; refusing cloud launch"
        )
    result["SWANLAB_API_KEY"] = fields[1]
    return result, "autokey-line6"


def _shell_quote(value: str) -> str:
    """A POSIX shell quote without invoking shell expansion."""

    # repr() is not shell-compatible for all characters; this small helper is
    # enough for paths/Hydra values and keeps the output easy to paste.
    if re.fullmatch(r"[A-Za-z0-9_./:=+@,%+-]+", value):
        return value
    return "'" + value.replace("'", "'\\''") + "'"


def _materialize_jsonl(src: Path, dst: Path, *, overwrite: bool = False) -> Path:
    """Delegate JSONL materialization to the canonical converter.

    VeRL expects nested Arrow values (not a JSON-encoded ``extra_info``
    string).  The repository's dedicated converter owns that schema and its
    transactional writer; keeping a second writer here previously allowed the
    two contracts to drift.  Loading it by file path keeps this preflight
    helper usable both as a script and as an imported module.
    """

    converter_path = REPO_ROOT / "scripts/convert_opd_jsonl_to_parquet.py"
    if not converter_path.is_file():
        raise RuntimeError(f"canonical converter is missing: {converter_path}")
    spec = importlib.util.spec_from_file_location("_qat_opd_parquet_converter", converter_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load canonical converter: {converter_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    try:
        module.convert(
            src,
            dst,
            overwrite=overwrite,
            # Keep conversion memory bounded for LongRLVR's large evidence
            # strings; the converter validates the complete output before it
            # returns.
            batch_size=64,
            verify=True,
        )
    except Exception as exc:
        raise RuntimeError(f"canonical JSONL materialization failed: {exc}") from exc
    print(f"[mopd] materialized canonical Parquet: {dst}")
    return dst


def _resolve(args: argparse.Namespace) -> Resolved:
    profiles = _profiles()
    profile = profiles[args.profile]
    # Keep profile defaults reproducible while allowing a controlled budget
    # sweep from the launcher.  The resulting lengths flow through data,
    # rollout, reward, and PPO token-limit overrides together.
    prompt_length = getattr(args, "prompt_length", None)
    response_length = getattr(args, "response_length", None)
    if prompt_length is not None or response_length is not None:
        profile = replace(
            profile,
            prompt_length=prompt_length if prompt_length is not None else profile.prompt_length,
            response_length=response_length if response_length is not None else profile.response_length,
        )
    stage_updates = 20 if args.stage == "gate" else 150
    updates = args.updates if args.updates is not None else stage_updates
    model = Path(args.model).expanduser() if args.model else profile.model
    teacher = Path(args.teacher).expanduser() if args.teacher else profile.teacher

    # Overriding --model does not re-point the rest of the profile.  Every 1.5B
    # launcher in this repository passes --profile deepseek_skywork (the 7B
    # profile) and overrides the model path, which leaves BASE_MODEL, N,
    # TENSOR_PARALLEL_SIZE, MAX_MODEL_LEN and MAX_NUM_SEQS at the 7B values.
    # The checkpoint-eval watcher reads exactly those, so it merged 1.5B
    # tensors into a 7B config.json (hidden_size 3584 vs 1536) and every
    # evaluation died in vLLM's WorkerProc init.  Compare architectures here,
    # where the override happens, instead of letting it surface as a shape
    # error hours later.
    if args.model and profile.model != model:
        def _hidden_size(path: Path) -> int | None:
            try:
                with open(Path(path) / "config.json") as fh:
                    return json.load(fh).get("hidden_size")
            except Exception:  # noqa: BLE001
                return None

        want, got = _hidden_size(profile.model), _hidden_size(model)
        if want is not None and got is not None and want != got:
            # Printed, not raised.  profile.name is part of the output
            # directory, so switching profiles would orphan every existing run
            # and break resume; the arms in flight have to keep the mismatched
            # profile until they finish.  The eval path no longer reads the
            # profile blindly -- experiments/2026-09-11/scripts/eval_kvqat_manual.sbatch
            # asks for qwen_brorl_1p5b explicitly and verifies the merged
            # hidden_size -- so this is here to stop the next person rediscovering it.
            print(
                f"[WARN] profile/model architecture mismatch: --model {model} "
                f"has hidden_size={got}, but profile {profile.name!r} describes "
                f"hidden_size={want} ({profile.model}).\n"
                f"[WARN]   The profile also supplies the *evaluation* protocol "
                f"(base model for checkpoint merging, N, tensor-parallel size, "
                f"context length). --model changes none of it.\n"
                f"[WARN]   Anything reading profile_env.py for this run -- the "
                f"checkpoint-eval watcher above all -- will use the "
                f"{want}-hidden-size settings and produce unloadable merges.\n"
                f"[WARN]   qwen_brorl_1p5b is the matching profile for the 1.5B student."
            )
    train = Path(args.train).expanduser() if args.train else profile.prepared_parquet
    val = Path(args.val).expanduser() if args.val else train
    kv_bits = args.kv_bits
    output = (
        Path(args.output).expanduser()
        if args.output
        else QAT_ROOT
        / "runs/mopd"
        / profile.name
        / f"kv{kv_bits}"
        / f"{args.stage}{updates}"
    )
    # Keep transient Ray sockets out of the descriptive (and potentially
    # long) artifact path.  Explicit CLI paths still take precedence.
    ray_root = _default_ray_root(output, profile, kv_bits, args.stage, updates)
    ray_temp_dir = Path(args.ray_temp_dir).expanduser() if args.ray_temp_dir else ray_root / "tmp"
    ray_spill_dir = Path(args.ray_spill_dir).expanduser() if args.ray_spill_dir else ray_root / "spill"
    python = Path(args.python).expanduser() if args.python else DEFAULT_PYTHON
    verl_dir = Path(args.verl_dir).expanduser() if args.verl_dir else DEFAULT_VERL_DIR
    py_libs = Path(args.py_libs).expanduser() if args.py_libs else DEFAULT_PYLIBS
    if not py_libs.is_dir():
        py_libs = None
    project = args.project or f"qat-mopd-{profile.name}"
    experiment = args.experiment or f"{profile.name}-kv{kv_bits}-{args.stage}{updates}"
    # Ray's resource-pool builder requests one CPU per colocated GPU worker,
    # while ``main_ppo`` separately schedules a one-CPU TaskRunner actor on
    # the head node.  A null/all-CPU Ray setting previously spawned hundreds
    # of workers; derive a small but sufficient default from the requested
    # topology (9 for one 8-GPU node, 17 for two such nodes).  An explicit
    # ``--ray-num-cpus`` remains an escape hatch, and preflight rejects a
    # value that cannot satisfy the placement group.
    ray_num_cpus = (
        args.ray_num_cpus
        if args.ray_num_cpus is not None
        else max(8, args.nodes * args.gpus + 1)
    )
    return Resolved(
        profile=profile,
        model=model,
        teacher=teacher,
        train=train,
        val=val,
        output=output,
        python=python,
        verl_dir=verl_dir,
        py_libs=py_libs,
        stage=args.stage,
        updates=updates,
        nodes=args.nodes,
        gpus=args.gpus,
        ray_num_cpus=ray_num_cpus,
        ray_temp_dir=ray_temp_dir,
        ray_spill_dir=ray_spill_dir,
        legacy_kv_bits=kv_bits,
        kv_scheme=args.kv_scheme,
        wa_scheme=args.wa_scheme,
        wa_state=args.wa_state,
        block_align_coef=args.block_align_coef,
        block_align_layer_every=args.block_align_layer_every,
        block_align_keep_fraction=args.block_align_keep_fraction,
        block_align_metric=args.block_align_metric,
        block_align_last_only=args.block_align_last_only,
        kv_align_coef=args.kv_align_coef,
        kv_align_layer_every=args.kv_align_layer_every,
        kv_align_keep_fraction=args.kv_align_keep_fraction,
        kv_align_metric=args.kv_align_metric,
        kv_align_align_keys=args.kv_align_align_keys,
        kv_align_align_values=args.kv_align_align_values,
        preserve_projection_bias=bool(
            args.preserve_projection_bias and not _native_tq_dtype(args)
        ),
        # vLLM uses -1 for an unrestricted sampling vocabulary.  The CLI
        # keeps 0 as the HF spelling, but native vLLM TurboQuant must never
        # receive 0 (it is rejected by the vLLM sampler contract).
        # The constraint below is a *vLLM spelling* rule, so it must key off
        # the rollout engine, not off the quantization preset.  Gating it on
        # _native_tq_dtype left the unquantized-control arm at 0, which the
        # validity check further down then rejects for engine=vllm -- making
        # an FP control run unlaunchable.
        sampling_top_k=(
            -1
            if args.rollout_engine == "vllm" and args.sampling_top_k == 0
            else args.sampling_top_k
        ),
        lr=args.lr,
        sampling_top_p=args.sampling_top_p,
        reward_top_k=args.reward_top_k,
        top_k_strategy=args.top_k_strategy,
        reward_weight_mode=args.reward_weight_mode,
        temperature=args.temperature,
        train_batch_size=args.train_batch_size,
        micro_batch_size_per_gpu=args.micro_batch_size_per_gpu,
        reward_micro_batch_size_per_gpu=args.reward_micro_batch_size_per_gpu,
        log_prob_micro_batch_size_per_gpu=args.log_prob_micro_batch_size_per_gpu,
        ppo_mini_batch_size=args.ppo_mini_batch_size,
        rollout_gpu_memory_utilization=args.rollout_gpu_memory_utilization,
        rollout_cuda_graph=args.rollout_cuda_graph,
        resume=args.resume,
        resume_path=args.resume_path,
        rollout_engine=args.rollout_engine,
        actor_model_dtype=args.actor_model_dtype,
        actor_attention=args.actor_attention,
        reward_attention=args.reward_attention,
        trust_remote_code=not args.no_trust_remote_code,
        disable_torch_compile=args.disable_torch_compile,
        swanlab_mode=args.swanlab_mode,
        project_name=project,
        experiment_name=experiment,
        extra_overrides=list(args.override or []),
        quantize_vllm_rollout=args.quantize_vllm_rollout,
        require_quantized_rollout=args.require_quantized_rollout,
        allow_rollout_policy_mismatch=args.allow_rollout_policy_mismatch,
        strict_fp32_arithmetic=args.strict_fp32_arithmetic,
        one_epoch=args.one_epoch,
        save_freq=args.save_freq,
        eval_checkpoints=tuple(args.eval_checkpoints),
        eval_gate_summary=(
            Path(args.eval_gate_summary).expanduser()
            if args.eval_gate_summary
            else None
        ),
        require_eval_gate=bool(args.require_eval_gate),
        allow_missing_eval_gate=bool(args.allow_missing_eval_gate),
        exact_bias_free_recompute=bool(
            args.exact_bias_free_recompute and not _native_tq_dtype(args)
        ),
        # Native vLLM TurboQuant owns the rollout cache; do not let the
        # legacy KIVI HF flag silently install a second cache path.
        kivi_hf_rollout=bool(args.kivi_hf_rollout and not _native_tq_dtype(args)),
        native_turboquant_kv_cache_dtype=_native_tq_dtype(args),
    )


def _status(ok: bool, label: str, detail: str, *, strict: bool, warnings: list[str]) -> bool:
    state = "OK" if ok else ("WARN" if not strict else "FAIL")
    print(f"[{state:4}] {label}: {detail}")
    if not ok:
        warnings.append(f"{label}: {detail}")
    return ok


def _strict_integral(value: Any) -> int | None:
    """Parse a JSON number only when it is finite and integral.

    Gate manifests are machine-generated and should carry integer bit widths;
    accepting ``int(4.5)`` would silently turn a malformed/stale summary into
    a KV4 launch.  Integral floats are accepted for backwards-compatible JSON
    producers, while booleans and strings remain invalid schema values.
    """

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    try:
        return parsed if float(value) == float(parsed) else None
    except (TypeError, ValueError, OverflowError):
        return None


def _strict_fraction(value: Any) -> float | None:
    """Return a finite JSON numeric fraction in the closed unit interval."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(parsed) or not 0.0 <= parsed <= 1.0:
        return None
    return parsed


def _check_path(path: Path, label: str, *, strict: bool, warnings: list[str], directory: bool = False) -> bool:
    exists = path.is_dir() if directory else path.is_file()
    if not exists:
        return _status(False, label, f"missing: {path}", strict=strict, warnings=warnings)
    if not directory and path.stat().st_size == 0:
        return _status(False, label, f"empty file: {path}", strict=strict, warnings=warnings)
    return _status(True, label, str(path), strict=strict, warnings=warnings)


def _check_eval_gate(r: Resolved, *, strict: bool, warnings: list[str]) -> None:
    """Validate an optional frozen-model FP32/KV2/KV4 selection manifest.

    The evaluator owns the numerical decision.  The launcher verifies that a
    supplied manifest is complete, belongs to this profile/model, and selects
    the exact KV bit-width requested for training.  A real ``main`` run
    requires a complete summary by default; ``--allow-missing-eval-gate`` is
    an explicit diagnostic escape hatch for the absence of a summary only.
    """

    summary = r.eval_gate_summary
    main_real_run = bool(strict and r.stage == "main" and not r.native_turboquant_rollout)
    # The historical selector chooses between HF KV2/KV4.  Native vLLM
    # TurboQuant runs have a fixed requested cache dtype (3bit_nc or
    # k3v4_nc), so that selector is not applicable and must not block the
    # native-TurboQuant main run.  An explicitly requested legacy summary
    # remains validated below for callers that deliberately ask for it.
    legacy_gate_applicable = not r.native_turboquant_rollout
    summary_strict = bool(r.require_eval_gate or (main_real_run and legacy_gate_applicable))
    summary_required = bool(
        r.require_eval_gate
        or (main_real_run and legacy_gate_applicable and not r.allow_missing_eval_gate)
    )
    if summary is None:
        if summary_required:
            _status(
                False,
                "evaluation gate",
                (
                    "a complete FP32/KV2(/KV4) selection summary is required for "
                    f"stage={r.stage} (pass --eval-gate-summary or use "
                    "--allow-missing-eval-gate only for an explicit diagnostic run)"
                ),
                strict=True,
                warnings=warnings,
            )
        elif main_real_run and r.allow_missing_eval_gate:
            detail = (
                "selection summary is missing but --allow-missing-eval-gate was "
                "explicitly set; this is an actor-only diagnostic run"
            )
            print(f"[WARN] evaluation gate missing but explicitly allowed: {detail}")
            warnings.append(f"evaluation gate missing but explicitly allowed: {detail}")
        else:
            print(
                "[INFO] evaluation gate: not required "
                + (
                    "(native vLLM TurboQuant dtype is fixed by the requested experiment)"
                    if r.native_turboquant_rollout
                    else "(pass --eval-gate-summary/--require-eval-gate before a main run)"
                )
            )
        return
    summary = summary.expanduser()
    if not summary.is_file():
        # ``--allow-missing-eval-gate`` is deliberately an escape hatch for
        # omitting the summary, not for mistyping a path (or pointing at a
        # report that has not been mounted yet).  Once a caller supplies a
        # path, a real main run remains fail-closed even when the escape hatch
        # is present; otherwise a typo could silently launch an ungated run.
        if summary_required or main_real_run:
            _status(
                False,
                "evaluation gate",
                f"missing summary: {summary}",
                strict=True,
                warnings=warnings,
            )
        else:
            detail = f"missing summary: {summary}"
            message = (
                "evaluation gate missing but explicitly allowed: "
                + detail
                if main_real_run and r.allow_missing_eval_gate
                else "evaluation gate optional: " + detail
            )
            print(f"[WARN] {message}")
            warnings.append(message)
        return
    try:
        payload = json.loads(summary.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        _status(
            False,
            "evaluation gate",
            f"invalid JSON summary {summary}: {type(exc).__name__}",
            strict=True,
            warnings=warnings,
        )
        return
    if not isinstance(payload, dict):
        _status(False, "evaluation gate", "summary must be a JSON object", strict=True, warnings=warnings)
        return

    failures: list[str] = []
    if payload.get("decision_ready") is not True:
        failures.append(f"decision_ready={payload.get('decision_ready')!r}")
    if payload.get("decision_status") not in {"kv2_primary", "kv4_primary"}:
        failures.append(f"decision_status={payload.get('decision_status')!r}")
    selected_bits = _strict_integral(payload.get("primary_kv_bits"))
    if selected_bits is None:
        failures.append(f"primary_kv_bits={payload.get('primary_kv_bits')!r}")
    if selected_bits != r.legacy_kv_bits:
        failures.append(f"primary_kv_bits={selected_bits!r} != requested legacy_kv_bits={r.legacy_kv_bits}")
    if payload.get("profile") != r.profile.name:
        failures.append(f"profile={payload.get('profile')!r} != {r.profile.name!r}")
    summary_model = payload.get("model")
    if not isinstance(summary_model, str) or not summary_model:
        failures.append("model is missing")
    elif os.path.realpath(summary_model) != os.path.realpath(str(r.model)):
        failures.append(f"model={summary_model!r} != {str(r.model)!r}")
    summary_eval_data = payload.get("eval_data")
    if summary_eval_data is not None and not isinstance(summary_eval_data, str):
        failures.append("eval_data is not a string")

    # A top-level ``decision_ready=true`` is not sufficient evidence for a
    # strict main run.  In particular, an older gate writer could combine a
    # newly generated KV4 report with legacy FP/KV2 reports while still
    # emitting that flag.  Require the normalized report bundle and its
    # immutable provenance whenever the caller explicitly asks for a hard
    # gate.  Advisory summaries retain the historical lightweight check so a
    # dry-run can inspect an in-progress/legacy artifact without pretending it
    # is safe for training.
    if summary_strict:
        gate_schema_version = payload.get("gate_schema_version")
        if (
            isinstance(gate_schema_version, bool)
            or not isinstance(gate_schema_version, int)
            or gate_schema_version != EVAL_GATE_SCHEMA_VERSION
        ):
            failures.append(
                f"gate_schema_version={gate_schema_version!r} != "
                f"{EVAL_GATE_SCHEMA_VERSION}"
            )
        expected_evaluator_profile = (
            "longrlvr" if r.profile.name == "qwen25_longrlvr" else "training"
        )
        if payload.get("training_data") is None:
            failures.append("training_data is missing")
        elif os.path.realpath(str(payload.get("training_data"))) != os.path.realpath(str(r.train)):
            failures.append("training_data does not match the resolved profile")
        if payload.get("evaluator_profile") != expected_evaluator_profile:
            failures.append(
                f"evaluator_profile={payload.get('evaluator_profile')!r} != "
                f"{expected_evaluator_profile!r}"
            )
        if payload.get("required_order") != ["fp32", "kv2", "kv4_conditional"]:
            failures.append("required_order is not FP32 -> KV2 -> conditional KV4")
        try:
            summary_threshold = float(payload.get("threshold"))
        except (TypeError, ValueError):
            summary_threshold = float("nan")
        if not math.isfinite(summary_threshold) or not 0.0 < summary_threshold < 1.0:
            failures.append(f"threshold={payload.get('threshold')!r} is not in (0,1)")
        elif not math.isclose(summary_threshold, 1.0 / 3.0, rel_tol=0.0, abs_tol=1e-12):
            failures.append(
                f"threshold={summary_threshold!r} differs from the registered one-third rule"
            )
        reports = payload.get("reports")
        if not isinstance(reports, dict):
            failures.append("reports provenance bundle is missing")
        else:
            required_report_names = ["fp", "kv2"]
            if selected_bits == 4:
                required_report_names.append("kv4")
            report_objects: dict[str, dict[str, Any]] = {}
            for report_name in required_report_names:
                report = reports.get(report_name)
                if not isinstance(report, dict):
                    failures.append(f"reports.{report_name} is missing or not an object")
                    continue
                report_objects[report_name] = report
                for field in (
                    "path",
                    "model_path_real",
                    "data",
                    "profile",
                    "count",
                    "total_samples",
                    "sampling_contract",
                    "seed",
                    "instruction",
                    "attn_implementation",
                    "preserve_projection_bias",
                    "evaluator_git_revision",
                    "evaluator_source_sha256",
                    "evaluator_schema_version",
                    "kv_bits",
                    "dtype_resolved",
                    "accuracy",
                ):
                    if field not in report or report.get(field) in (None, ""):
                        failures.append(f"reports.{report_name}.{field} is missing")
                report_schema_version = _strict_integral(
                    report.get("evaluator_schema_version")
                )
                if report_schema_version is None or report_schema_version < 2:
                    failures.append(
                        f"reports.{report_name}.evaluator_schema_version must be >=2"
                    )
                if report_name == "fp":
                    if report.get("kv_bits") != 32:
                        failures.append(f"reports.fp.kv_bits={report.get('kv_bits')!r}, expected 32")
                    if report.get("dtype_resolved") != "fp32":
                        failures.append(
                            "reports.fp.dtype_resolved must be 'fp32' for a strict reference"
                        )
                elif report.get("kv_bits") != (2 if report_name == "kv2" else 4):
                    expected_report_bits = 2 if report_name == "kv2" else 4
                    failures.append(
                        f"reports.{report_name}.kv_bits={report.get('kv_bits')!r}, "
                        f"expected {expected_report_bits}"
                    )
                dtype_resolved = report.get("dtype_resolved")
                if not isinstance(dtype_resolved, str) or dtype_resolved not in {
                    "fp32",
                    "bf16",
                    "fp16",
                }:
                    failures.append(
                        f"reports.{report_name}.dtype_resolved={dtype_resolved!r} "
                        "must be one of fp32/bf16/fp16"
                    )
                accuracy = _strict_fraction(report.get("accuracy"))
                if accuracy is None:
                    failures.append(
                        f"reports.{report_name}.accuracy={report.get('accuracy')!r} "
                        "must be a finite number in [0,1]"
                    )
                source_sha = report.get("evaluator_source_sha256")
                if not isinstance(source_sha, str) or re.fullmatch(r"[0-9a-fA-F]{64}", source_sha) is None:
                    failures.append(
                        f"reports.{report_name}.evaluator_source_sha256 is not a 64-hex digest"
                    )
                report_path = report.get("path")
                if isinstance(report_path, str) and report_path:
                    report_file = Path(report_path).expanduser()
                    if not report_file.is_file():
                        failures.append(
                            f"reports.{report_name}.path does not exist: {report_file}"
                        )
                    else:
                        # Re-read the raw report and compare the fields that
                        # determine the gate.  This prevents a copied or
                        # edited normalized bundle from being trusted without
                        # its underlying evaluator artifact.
                        try:
                            raw_report = json.loads(report_file.read_text(encoding="utf-8"))
                        except (OSError, UnicodeError, ValueError) as exc:
                            failures.append(
                                f"reports.{report_name}.path is not valid JSON ({type(exc).__name__})"
                            )
                        else:
                            if not isinstance(raw_report, dict):
                                failures.append(f"reports.{report_name}.path JSON is not an object")
                            else:
                                for field in (
                                    "kv_bits",
                                    "dtype_resolved",
                                    "model_path_real",
                                    "data",
                                    "profile",
                                    "count",
                                    "total_samples",
                                    "accuracy",
                                    "sampling",
                                    "evaluator_schema_version",
                                    "evaluator_source_sha256",
                                    "evaluator_git_revision",
                                    "seed",
                                    "instruction",
                                    "attn_implementation",
                                    "preserve_projection_bias",
                                ):
                                    raw_value = raw_report.get(field)
                                    expected_value = report.get(field)
                                    if field == "sampling":
                                        raw_value = {
                                            key: raw_value.get(key)
                                            for key in (
                                                "temperature",
                                                "top_k",
                                                "top_p",
                                                "max_new_tokens",
                                                "num_samples",
                                            )
                                        } if isinstance(raw_value, Mapping) else None
                                        expected_value = report.get("sampling_contract")
                                    if field in {"model_path_real", "data"}:
                                        equal = os.path.realpath(str(raw_value)) == os.path.realpath(str(expected_value))
                                    else:
                                        equal = raw_value == expected_value
                                    if not equal:
                                        failures.append(
                                            f"reports.{report_name} raw/normalized {field} differs"
                                        )

            if report_objects:
                fp_report = report_objects.get("fp")
                if fp_report is not None:
                    # Every report must describe the same generation contract
                    # and evaluator source.  A missing field was already
                    # reported above; this loop catches mixed-schema bundles
                    # such as the historical DeepSeek gate.
                    for report_name, report in report_objects.items():
                        for field in (
                            "model_path_real",
                            "data",
                            "profile",
                            "count",
                            "total_samples",
                            "sampling_contract",
                            "seed",
                            "instruction",
                            "attn_implementation",
                            "preserve_projection_bias",
                            "evaluator_git_revision",
                            "evaluator_source_sha256",
                        ):
                            left = fp_report.get(field)
                            right = report.get(field)
                            if field in {"model_path_real", "data"}:
                                equal = os.path.realpath(str(left)) == os.path.realpath(str(right))
                            else:
                                equal = left == right
                            if not equal:
                                failures.append(
                                    f"reports.{report_name} is not comparable to reports.fp: "
                                    f"{field} differs"
                                )
                    expected_model_real = os.path.realpath(str(r.model))
                    for report_name, report in report_objects.items():
                        if os.path.realpath(str(report.get("model_path_real"))) != expected_model_real:
                            failures.append(
                                f"reports.{report_name}.model_path_real does not match profile model"
                            )
                        if report.get("data") and os.path.realpath(str(report.get("data"))) != os.path.realpath(str(payload.get("eval_data", report.get("data")))):
                            # The evaluator data is governed by the summary;
                            # this branch is intentionally only a consistency
                            # check against the top-level path when present.
                            if payload.get("eval_data") is not None:
                                failures.append(
                                    f"reports.{report_name}.data differs from summary eval_data"
                                )
                # Recompute the selection from the nested point estimates so
                # a hand-edited top-level status cannot satisfy the gate.
                if "fp" in report_objects and "kv2" in report_objects:
                    try:
                        fp_accuracy = float(report_objects["fp"].get("accuracy"))
                        kv2_accuracy = float(report_objects["kv2"].get("accuracy"))
                        threshold = float(payload.get("threshold", 1.0 / 3.0))
                        if not math.isfinite(threshold) or not 0.0 < threshold < 1.0:
                            raise ValueError("invalid threshold")
                        expected_needs_kv4 = fp_accuracy <= 0.0 or kv2_accuracy <= 0.0 or kv2_accuracy < fp_accuracy * threshold
                        expected_status = "needs_kv4" if expected_needs_kv4 and "kv4" not in report_objects else "kv4_primary" if expected_needs_kv4 else "kv2_primary"
                        if payload.get("decision_status") != expected_status:
                            failures.append(
                                f"decision_status={payload.get('decision_status')!r} disagrees with nested reports ({expected_status!r})"
                            )
                        if expected_needs_kv4 and "kv4" not in report_objects:
                            failures.append("nested reports omit required KV4 result")
                        for top_key, report_key in (
                            ("fp_accuracy", "fp"),
                            ("kv2_accuracy", "kv2"),
                            ("kv4_accuracy", "kv4"),
                        ):
                            if report_key not in report_objects:
                                continue
                            try:
                                top_value = float(payload.get(top_key))
                                nested_value = float(report_objects[report_key].get("accuracy"))
                            except (TypeError, ValueError):
                                failures.append(f"{top_key} is not numeric")
                            else:
                                if not math.isclose(top_value, nested_value, rel_tol=0.0, abs_tol=1e-12):
                                    failures.append(f"{top_key} disagrees with reports.{report_key}.accuracy")
                    except (TypeError, ValueError):
                        failures.append("nested report accuracies/threshold are not numeric")
    if failures:
        _status(
            False,
            "evaluation gate",
            f"summary={summary}; " + "; ".join(failures),
            strict=True,
            warnings=warnings,
        )
        return
    print(
        f"[OK  ] evaluation gate: {summary} (profile={r.profile.name}, "
        f"primary_kv_bits={selected_bits}, decision_ready=true)"
    )


def _needs_legacy_kv_plugin(r: "Resolved") -> bool:
    """Is ``qat_kv.verl_plugin`` actually doing anything?

    Its job is the pre-TurboQuant actor-side KV fake-quant hook.  With the
    native TurboQuant rollout it is not loaded at all; with no legacy
    quantization requested either (``legacy_kv_bits == 32`` and no
    ``--kv-scheme``) it is a pure no-op.  It is not free, though: it patches
    ``ActorRolloutRefWorker``, and that makes verl's colocated ``WorkerDict``
    unpicklable ("cannot pickle 'ConfigModuleInstance' object"), so a full
    precision control run cannot even reach the first step.  The teacher-side
    flash-attention workaround it also carries is vestigial here -- this tree's
    RewardModelWorker reads ``attn_implementation`` from the config.
    """
    if r.native_turboquant_rollout:
        return False
    return r.legacy_kv_bits < 32 or bool(r.kv_scheme)


def _dependency_report(r: Resolved, *, strict: bool, warnings: list[str], skip: bool) -> None:
    if skip:
        print("[SKIP] dependency checks requested by caller")
        return
    _check_path(r.python, "python", strict=strict, warnings=warnings)
    _check_path(r.verl_dir / "verl/trainer/main_ppo.py", "Open-MOPD veRL source", strict=strict, warnings=warnings)
    plugin = REPO_ROOT / "qat_kv/verl_plugin.py"
    if not plugin.is_file():
        _status(
            False,
            "KV external_lib",
            f"{plugin} is not present yet; actor hook import will fail at run time",
            strict=strict,
            warnings=warnings,
        )
    # The pinned fork's legacy ``RewardModelWorker`` contains a literal
    # ``flash_attention_2`` argument even when flash-attn is not installed.
    # Do a cheap source check so the dry-run records why the external shim is
    # present; the shim (owned by ``qat_kv/verl_plugin.py``) applies the
    # requested backend only around the teacher build and never quantizes that
    # model.
    reward_source = r.verl_dir / "verl/workers/fsdp_workers.py"
    try:
        reward_text = reward_source.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        reward_text = ""
    # Restrict the search to RewardModelWorker; the actor/critic builders also
    # contain independent flash-attention defaults and should not trigger a
    # teacher warning.
    reward_start = reward_text.find("class RewardModelWorker")
    reward_region = reward_text[reward_start:] if reward_start >= 0 else ""
    hardcoded_flash = bool(
        reward_region
        and re.search(
            r"attn_implementation\s*=\s*['\"]flash_attention_2['\"]",
            reward_region,
        )
    )
    configurable_reward_attention = bool(
        reward_region
        and re.search(r"config\.model\.get\(\s*['\"]attn_implementation['\"]", reward_region)
    )
    if hardcoded_flash:
        if plugin.is_file():
            _status(
                True,
                "teacher attention backend",
                f"legacy flash_attention_2 detected; QAT_KV_REWARD_ATTN={r.reward_attention} shim is configured",
                strict=strict,
                warnings=warnings,
            )
        else:
            _status(
                False,
                "teacher attention backend",
                "legacy flash_attention_2 detected but attention fallback plugin is missing",
                strict=strict,
                warnings=warnings,
            )
    elif reward_source.is_file() and configurable_reward_attention:
        _status(
            True,
            "teacher attention backend",
            f"RewardModelWorker reads configurable attn_implementation (launcher requests {r.reward_attention})",
            strict=strict,
            warnings=warnings,
        )
    # Use importlib.metadata in the launcher process where possible.  This
    # avoids importing vLLM/Ray (which can initialize CUDA) just to report a
    # version.  For a non-default interpreter, query that interpreter below.
    required_base = ("torch", "transformers", "vllm", "ray", "pyarrow", "hydra-core", "omegaconf")
    # Offline/disabled smoke checks intentionally use console-only logging;
    # requiring the optional SwanLab wheel in those modes would make a local
    # config probe fail for an unrelated dependency.
    required = required_base + (("swanlab",) if r.swanlab_mode == "cloud" else ())
    probe = r.python if r.python.is_file() else Path(sys.executable)
    # Keep the probe a tiny metadata-only Python process.  In particular, do
    # not import torch/vLLM/Ray here: those imports can initialize CUDA or a
    # Ray runtime and make a read-only dry-run surprisingly stateful.
    code = (
        "import importlib.metadata as m, importlib.util\n"
        "names=%r\n"
        "mods={'hydra-core':'hydra','omegaconf':'omegaconf','swanlab':'swanlab'}\n"
        "parts=[]\n"
        "for n in names:\n"
        "  try: parts.append(n+'='+m.version(n))\n"
        "  except m.PackageNotFoundError:\n"
        "    if n in mods and importlib.util.find_spec(mods[n]) is not None: parts.append(n+'=importable')\n"
        "    else: parts.append(n+'=MISSING')\n"
        "print(';'.join(parts))\n"
        % (required,)
    )
    try:
        completed = subprocess.run(
            [str(probe), "-c", code],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
            env={"PATH": os.environ.get("PATH", ""), "PYTHONPATH": _pythonpath(r)},
        )
        output = completed.stdout.strip() or completed.stderr.strip()
        print(f"[INFO] package probe ({probe}): {output}")
        versions = dict(item.split("=", 1) for item in output.split(";") if "=" in item)
    except Exception as exc:
        versions = {}
        _status(False, "package probe", str(exc), strict=strict, warnings=warnings)
    for package in required:
        version = versions.get(package, "MISSING")
        # hydra-core is exposed as ``hydra`` at import time, but metadata is
        # the stable check.  A missing SwanLab is also fatal when logger is
        # enabled; it is kept separate so cloud credentials are never probed.
        _status(version != "MISSING", package, version, strict=strict, warnings=warnings)


def _pythonpath(r: Resolved) -> str:
    # Keep the repository and pinned VeRL checkout first, then the dedicated
    # SwanLab wheel directory.  Ray workers inherit this exact value, so the
    # driver and every actor/reward process resolve identical dependencies.
    # De-duplicate paths because callers often pass the same compatibility
    # directory explicitly via ``--py-libs``.
    candidates: list[str] = [str(REPO_ROOT), str(r.verl_dir)]
    if TURBOQUANT_TRAIN_OVERLAY.is_dir():
        candidates.append(str(TURBOQUANT_TRAIN_OVERLAY))
    if MOPD_EXTRA_LIBS.is_dir():
        candidates.append(str(MOPD_EXTRA_LIBS))
    # Only for arms that actually quantize weights/activations: every Ray actor
    # imports qat_kv.wa_scheme, which imports modelopt.
    if r.wa_scheme and MODELOPT_LIBS.is_dir():
        candidates.append(str(MODELOPT_LIBS))
    if r.py_libs is not None:
        candidates.append(str(r.py_libs))
    if os.environ.get("PYTHONPATH"):
        candidates.extend(item for item in os.environ["PYTHONPATH"].split(os.pathsep) if item)
    pieces: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        if candidate not in seen:
            pieces.append(candidate)
            seen.add(candidate)
    return os.pathsep.join(pieces)


def preflight(r: Resolved, *, strict: bool, skip_deps: bool, materialize: bool, overwrite_data: bool) -> tuple[list[str], int | None]:
    """Print checks and return warnings plus an estimated batches/epoch."""

    warnings: list[str] = []
    print("[mopd] preflight")
    print(f"[mopd] profile: {r.profile.description}")
    print(f"[mopd] source revision: {r.profile.dataset_revision}")
    print(f"[mopd] stage: {r.stage} ({r.updates} optimizer updates)")
    _check_path(r.model, "student model", strict=strict, warnings=warnings, directory=True)
    _check_path(r.teacher, "frozen teacher", strict=strict, warnings=warnings, directory=True)

    # A prepared Parquet companion is preferred.  If only JSONL exists, an
    # explicit materialization flag may create it; otherwise dry-run reports a
    # warning and leaves the complete command visible.
    if not r.train.is_file() and r.train.suffix.lower() == ".parquet":
        source_jsonl = r.profile.prepared_jsonl
        if source_jsonl.is_file() and materialize:
            try:
                _materialize_jsonl(source_jsonl, r.train, overwrite=overwrite_data)
            except Exception as exc:
                _status(False, "JSONL materialization", str(exc), strict=strict, warnings=warnings)
        elif source_jsonl.is_file():
            _status(
                False,
                "training Parquet",
                f"missing: {r.train}; prepared JSONL exists at {source_jsonl}; rerun with --materialize-jsonl",
                strict=strict,
                warnings=warnings,
            )
        elif r.profile.raw_parquet.is_file():
            _status(
                False,
                "governed training Parquet",
                f"missing: {r.train}; raw source exists but is intentionally not used ({r.profile.raw_parquet})",
                strict=strict,
                warnings=warnings,
            )
        else:
            _check_path(r.train, "training Parquet", strict=strict, warnings=warnings)
    else:
        _check_path(r.train, "training data", strict=strict, warnings=warnings)
    _check_path(r.val, "validation data", strict=strict, warnings=warnings)

    # A main launch may be tied to the frozen-model FP/KV selection gate.
    # Keep this separate from row counting so a missing summary is visible
    # even while a Parquet artifact is being materialized.
    _check_eval_gate(r, strict=strict, warnings=warnings)

    rows: int | None = None
    if r.train.is_file():
        is_parquet = r.train.suffix.lower() == ".parquet"
        rows = _parquet_rows(r.train) if is_parquet else _count_jsonl(r.train)
        manifest_rows = _manifest_rows(
            r.train,
            # The governed preparation manifest is authoritative for the
            # profile's canonical file; custom ``--train`` paths use their
            # local companion (if present).
            r.profile.prepared_manifest if r.train == r.profile.prepared_parquet else None,
        )
        if rows is None and is_parquet and manifest_rows is not None:
            # ``_parquet_rows`` intentionally returns None when pyarrow (or
            # Parquet metadata access) is unavailable.  A converter companion
            # or governed manifest still gives us a validated row budget, so
            # use it for epoch planning rather than silently assuming the
            # profile's nominal 16K rows.
            rows = manifest_rows
            print(
                f"[INFO] training rows (companion/governed manifest fallback; "
                f"Parquet metadata unavailable): {rows:,}"
            )
        elif rows is None and is_parquet:
            print(
                "[WARN] training rows: unknown (Parquet metadata unavailable; "
                "no companion/governed manifest)"
            )
        else:
            print(f"[INFO] training rows: {rows:,}" if rows is not None else "[WARN] training rows: unknown")
        if rows is not None and rows <= 0:
            _status(False, "training rows", "zero rows", strict=strict, warnings=warnings)
        if (
            manifest_rows is not None
            and rows is not None
            and manifest_rows != rows
        ):
            _status(
                False,
                "manifest row count",
                f"manifest={manifest_rows}, file={rows}",
                strict=strict,
                warnings=warnings,
            )
    elif r.train == r.profile.prepared_parquet and r.profile.prepared_jsonl.is_file():
        # The prepared manifest is cheap to read and lets a dry-run compute
        # the correct number of dataloader epochs even before the optional
        # JSONL->Parquet conversion has been performed.
        rows = _manifest_rows(r.profile.prepared_jsonl, r.profile.prepared_manifest)
        if rows is not None:
            print(f"[INFO] governed JSONL rows (Parquet pending): {rows:,}")
        else:
            rows = r.profile.expected_rows
            print(f"[INFO] expected governed rows (manifest pending): {rows:,}")
    elif r.train == r.profile.prepared_parquet:
        # Both the corrected JSONL and its manifest may still be being
        # generated on a shared filesystem.  The profile's fixed row budget
        # is a planning estimate only; strict --run will still fail until the
        # actual Parquet file exists.
        rows = r.profile.expected_rows
        print(f"[INFO] expected governed rows (data preparation pending): {rows:,}")

    if r.output.exists():
        try:
            entries = list(r.output.iterdir()) if r.output.is_dir() else [r.output]
        except OSError:
            entries = [r.output]
        nonempty = bool(entries)
        if nonempty and not r.resume:
            _status(
                False,
                "output directory",
                f"non-empty and --resume not set: {r.output}",
                strict=strict,
                warnings=warnings,
            )
        else:
            _status(True, "output directory", f"{'resume' if r.resume else 'empty'}: {r.output}", strict=strict, warnings=warnings)
    else:
        _status(True, "output directory", f"will create: {r.output}", strict=strict, warnings=warnings)

    if r.nodes not in (1, 2):
        _status(False, "nodes", f"expected 1 or 2, got {r.nodes}", strict=True, warnings=warnings)
    if r.gpus < 1:
        _status(False, "GPUs per node", f"must be positive, got {r.gpus}", strict=True, warnings=warnings)
    # ``main_ppo`` itself is a one-CPU Ray actor.  The standard resource pool
    # then creates one CPU+GPU bundle per colocated worker.  A cap of exactly
    # eight CPUs with the eight-GPU profile consequently leaves no CPU for
    # the TaskRunner and causes a permanent STRICT_PACK pending demand.  Keep
    # the old null/all-CPU storm bounded while checking the real floor here.
    # ResourcePoolManager creates one CPU+GPU bundle for every GPU process on
    # every requested node, and main_ppo's TaskRunner consumes one additional
    # head-node CPU.  Use the full topology here; otherwise a two-node launch
    # would pass a one-node floor and hang in STRICT_PACK placement.
    ray_cpu_floor = max(8, r.nodes * r.gpus + 1)
    _status(
        r.ray_num_cpus >= ray_cpu_floor,
        "Ray CPU capacity",
        f"{r.ray_num_cpus} (minimum {ray_cpu_floor}: one TaskRunner CPU + {r.nodes * r.gpus} GPU-worker CPUs; launcher cap avoids null/all-CPU storm)",
        strict=True,
        warnings=warnings,
    )
    ray_config = r.verl_dir / "verl/trainer/config/ppo_trainer.yaml"
    try:
        ray_config_text = ray_config.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        ray_config_text = ""
    # Keep this check dependency-free (the launcher must also work before
    # Hydra is importable) and tolerate the comments/blank lines in the
    # upstream YAML.
    ray_default_num_cpus_null = False
    ray_section = ray_init_section = False
    for line in ray_config_text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        indent = len(line) - len(line.lstrip())
        if indent == 0 and stripped == "ray_kwargs:":
            ray_section, ray_init_section = True, False
            continue
        if ray_section and indent == 2 and stripped == "ray_init:":
            ray_init_section = True
            continue
        if ray_section and indent == 0 and stripped != "ray_kwargs:":
            ray_section, ray_init_section = False, False
        if ray_init_section and indent == 4 and re.match(r"num_cpus\s*:\s*null(?:\s|$)", stripped):
            ray_default_num_cpus_null = True
            break
    if ray_default_num_cpus_null:
        # A null value makes Ray claim every visible CPU.  On this cluster
        # that can pre-start hundreds of one-CPU workers and leave the task
        # runner waiting forever; the explicit override below is intentional.
        warning = (
            "Ray config default num_cpus=null; launcher forces "
            f"ray_kwargs.ray_init.num_cpus={r.ray_num_cpus}"
        )
        print(f"[WARN] {warning}")
        warnings.append(warning)
    else:
        print(f"[OK  ] Ray CPU capacity override: {r.ray_num_cpus}")

    def _shared_path(path: Path) -> bool:
        resolved_path = path.expanduser().resolve(strict=False)
        return resolved_path.is_absolute() and (
            str(resolved_path) == "/shared" or str(resolved_path).startswith("/shared/")
        )

    _status(
        _shared_path(r.ray_temp_dir),
        "Ray _temp_dir",
        f"{r.ray_temp_dir} (must be under /shared)",
        strict=True,
        warnings=warnings,
    )
    _status(
        _shared_path(r.ray_spill_dir),
        "Ray object spilling directory",
        f"{r.ray_spill_dir} (pre-created metadata path; Ray 2.58 uses its default under _temp_dir)",
        strict=True,
        warnings=warnings,
    )
    socket_bytes, socket_path = _ray_socket_projection(r.ray_temp_dir)
    _status(
        socket_bytes <= _RAY_AF_UNIX_MAX_BYTES,
        "Ray Unix socket path",
        f"worst-case {socket_bytes} bytes / {_RAY_AF_UNIX_MAX_BYTES}: {socket_path}",
        strict=strict,
        warnings=warnings,
    )
    _status(
        r.ray_temp_dir.resolve(strict=False) != r.ray_spill_dir.resolve(strict=False),
        "Ray directory separation",
        "_temp_dir and object spilling directory are distinct",
        strict=True,
        warnings=warnings,
    )
    batch_unit = max(1, r.world_size) * max(1, r.micro_batch_size_per_gpu)
    if r.train_batch_size % batch_unit:
        _status(
            False,
            "global batch divisibility",
            f"batch={r.train_batch_size} is not divisible by world={r.world_size} "
            f"x micro_batch={r.micro_batch_size_per_gpu}",
            strict=True,
            warnings=warnings,
        )
    else:
        print(
            f"[OK  ] global batch: {r.train_batch_size} = world {r.world_size} "
            f"x micro-batch {r.micro_batch_size_per_gpu} "
            f"x gradient accumulation {r.gradient_accumulation_steps}"
        )

    total_context = r.profile.prompt_length + r.profile.response_length
    _status(
        total_context <= r.profile.model_context_length,
        "context budget",
        f"prompt={r.profile.prompt_length} + response={r.profile.response_length} = {total_context} "
        f"(model context {r.profile.model_context_length}; over-limit rows are skipped)",
        strict=True,
        warnings=warnings,
    )
    rollout_chunked_prefill = total_context > 32768
    # Chunked prefill permits a request longer than one scheduler window.  A
    # 32K window limits per-iteration activation memory while still accepting
    # the full 64K Qwen training prompt.
    rollout_max_num_batched_tokens = min(32768, total_context)
    print(
        f"[OK  ] vLLM scheduler: max_num_batched_tokens={rollout_max_num_batched_tokens}; "
        f"chunked_prefill={'true' if rollout_chunked_prefill else 'false'}; "
        "long requests will not be rejected by the scheduler"
    )
    print(
        f"[OK  ] actor gradient checkpointing: {'enabled' if (total_context >= 32768 or r.profile.response_length >= 16384) else 'disabled'} "
        "(long-context memory safeguard)"
    )
    print(
        f"[OK  ] HF attention backends: actor={r.actor_attention}; "
        f"frozen teacher={r.reward_attention}"
    )
    actor_fp32_contract = r.actor_model_dtype == "fp32"
    optimizer_expectation = (
        "FP32 master optimizer parameters expected (source worker builds model before FSDP mixed cast)"
        if actor_fp32_contract
        else "optimizer-state precision is not guaranteed when actor model is initialized in " + r.actor_model_dtype
    )
    _status(
        actor_fp32_contract or not r.strict_fp32_arithmetic,
        "actor dtype contract",
        (
            f"model_dtype={r.actor_model_dtype}; FSDP mixed_precision="
            f"param={ACTOR_FSDP_PARAM_DTYPE},reduce={ACTOR_FSDP_REDUCE_DTYPE},"
            f"buffer={ACTOR_FSDP_BUFFER_DTYPE}; {optimizer_expectation}; "
            "FP32 actor parameters are roughly 2x BF16 before sharding and FP32 "
            "optimizer master/moments can add memory; profile peak usage"
        ),
        strict=strict,
        warnings=warnings,
    )
    exact_bias_valid = (
        not r.exact_bias_free_recompute
        or (r.preserve_projection_bias and r.legacy_kv_bits < 32)
    )
    _status(
        exact_bias_valid,
        "HF exact bias-free recompute",
        (
            "disabled (default post-forward approximate bias preservation)"
            if not r.exact_bias_free_recompute
            else (
                "enabled for builtin nn.Linear K/V projections; "
                "QAT_KV_EXACT_BIAS_FREE=1 (extra projection GEMM)"
            )
        )
        if exact_bias_valid
        else (
            "requires --preserve-projection-bias and a sub-32-bit KV setting; "
            f"got preserve_projection_bias={r.preserve_projection_bias}, legacy_kv_bits={r.legacy_kv_bits}"
        ),
        strict=True,
        warnings=warnings,
    )
    _status(r.reward_top_k > 0, "MOPD reward top-k", f"{r.reward_top_k} (student, untruncated distribution)", strict=True, warnings=warnings)
    _status(
        math.isfinite(r.temperature) and r.temperature > 0.0,
        "rollout temperature",
        str(r.temperature),
        strict=True,
        warnings=warnings,
    )
    # vLLM uses -1 for an unbounded generation vocabulary; the HF sampler
    # uses 0 for the same meaning.  Keep this generation knob independent of
    # the positive MOPD reward candidate count above.
    sampling_top_k_valid = (
        (r.sampling_top_k == -1 or r.sampling_top_k > 0)
        if r.rollout_engine == "vllm"
        else (r.sampling_top_k == 0 or r.sampling_top_k > 0)
    )
    _status(
        sampling_top_k_valid,
        "rollout sampling top-k",
        f"{r.sampling_top_k} (separate from reward top-k; {'vLLM -1' if r.rollout_engine == 'vllm' else 'HF 0'} disables)",
        strict=True,
        warnings=warnings,
    )
    if not r.native_turboquant_rollout:
        _status(
            r.legacy_kv_bits in (2, 4, 8, 16, 32),
            "legacy KV bits",
            str(r.legacy_kv_bits),
            strict=True,
            warnings=warnings,
        )
    # NVFP4 weight/activation fake-quant.  Validated and exported here so a
    # bare CLI invocation behaves the same as the sbatch, which also exports
    # these; qat_kv.wa_scheme reads them inside the FSDP worker.
    if r.wa_scheme:
        state = r.wa_state or os.environ.get("QAT_WA_STATE", "")
        _status(
            bool(state) and Path(state).is_file(),
            "actor W/A fake-quant",
            f"{r.wa_scheme} needs a calibrated modelopt state; "
            f"--wa-state={state or 'unset'} is not a file. Produce it with "
            "scripts/ptq_nvfp4.py.",
            strict=True,
            warnings=warnings,
        )
        os.environ["QAT_WA_SCHEME"] = r.wa_scheme
        if state:
            os.environ["QAT_WA_STATE"] = state

        # Import it here, in the driver, rather than discovering three minutes
        # later that the Ray actors cannot (job 22939 died exactly that way).
        _modelopt_ok, _modelopt_why = True, ""
        try:
            import importlib.util

            # modelopt.torch pulls in omegaconf, which lives in the *other*
            # compatibility directory -- both must be on the path or this check
            # fails on a dependency the Ray workers would have resolved fine.
            for _libs in (MODELOPT_LIBS, MOPD_EXTRA_LIBS):
                if _libs.is_dir() and str(_libs) not in sys.path:
                    sys.path.append(str(_libs))
            if importlib.util.find_spec("modelopt.torch.opt") is None:
                _modelopt_ok, _modelopt_why = False, "modelopt.torch.opt not found"
        except Exception as exc:  # noqa: BLE001
            _modelopt_ok = False
            _modelopt_why = f"{type(exc).__name__}: {exc}"
        _status(
            _modelopt_ok,
            "modelopt importable",
            f"--wa-scheme={r.wa_scheme} needs NVIDIA modelopt, but {_modelopt_why}. "
            f"It is not in the venv; it lives in {MODELOPT_LIBS} and is added to "
            "PYTHONPATH only when --wa-scheme is set.",
            strict=True,
            warnings=warnings,
        )
        print(
            f"[OK  ] actor W/A fake-quant: {r.wa_scheme} "
            f"(modelopt TensorQuantizer on every nn.Linear, clipped STE, "
            f"frozen amax from {state})"
        )
    elif os.environ.get("QAT_WA_SCHEME"):
        # An env var left over from another arm would silently quantize this
        # one.  Every arm must name its own scheme.
        _status(
            False,
            "actor W/A fake-quant",
            f"QAT_WA_SCHEME={os.environ['QAT_WA_SCHEME']!r} is set in the "
            "environment but --wa-scheme was not passed; refusing to quantize "
            "an arm that did not ask for it",
            strict=True,
            warnings=warnings,
        )

    scheme = r.kv_scheme
    if scheme and scheme != "symmetric":
        # The whole point of the fake quantizer is that it is the *deployed*
        # one.  A scheme that does not name the rollout's cache dtype trains
        # the student for a quantizer it will never run under.
        _status(
            bool(r.native_turboquant_rollout)
            and scheme == r.native_turboquant_kv_cache_dtype,
            "actor QAT scheme",
            f"{scheme}; rollout kv_cache_dtype="
            f"{r.native_turboquant_kv_cache_dtype or 'none'} "
            "(training and rollout quantizers must be the same preset)",
            strict=True,
            warnings=warnings,
        )
        if r.block_align_coef > 0:
            print(
                f"[OK  ] block alignment: coef={r.block_align_coef} "
                f"every={r.block_align_layer_every} keep={r.block_align_keep_fraction} "
                f"metric={r.block_align_metric} (reference = same weights, quantizer off)"
            )
        if r.kv_align_coef > 0:
            print(
                f"[OK  ] KV alignment: coef={r.kv_align_coef} every={r.kv_align_layer_every} "
                f"keep={r.kv_align_keep_fraction} metric={r.kv_align_metric} "
                f"keys={r.kv_align_align_keys} values={r.kv_align_align_values} "
                "(student quantized K/V -> frozen teacher full-precision K/V)"
            )
        print(
            f"[OK  ] actor fake-quant: {scheme} installed post-RoPE inside "
            "attention (k_proj/v_proj projection hook stays disabled; "
            "TurboQuant quantizes post-RoPE keys)"
        )
    else:
        _status(
            r.block_align_coef <= 0 or bool(r.wa_scheme),
            "block alignment",
            f"coef={r.block_align_coef} requires --kv-scheme or --wa-scheme; "
            "without a quantizer the reference equals the student and the term "
            "is identically zero",
            strict=True,
            warnings=warnings,
        )
        _status(
            r.kv_align_coef <= 0,
            "KV alignment",
            f"coef={r.kv_align_coef} requires --kv-scheme; without a quantizer "
            "the student K/V equal the teacher K/V and the term is identically zero",
            strict=True,
            warnings=warnings,
        )
        print(
            "[INFO] actor fake-quant: disabled (no --kv-scheme); the actor forward is full precision and gradients do not see quantization"
        )
    if r.native_turboquant_rollout:
        native_ok = (
            r.rollout_engine == "vllm"
            and r.legacy_kv_bits == 32
            and not r.quantize_vllm_rollout
            and not r.kivi_hf_rollout
            and r.native_turboquant_kv_cache_dtype in NATIVE_TURBOQUANT_DTYPES
        )
        _status(
            native_ok,
            "native TurboQuant rollout",
            (
                f"dtype={r.native_turboquant_kv_cache_dtype}, engine={r.rollout_engine}, "
                f"legacy_kv_bits={r.legacy_kv_bits}; old HF QAT/KIVI hooks disabled and vLLM "
                "constructor/cache dtype must be verified at runtime"
            )
            if native_ok
            else (
                "requires vLLM engine, --kv-bits 32, no --quantize-vllm-rollout, "
                "and no KIVI HF rollout"
            ),
            strict=True,
            warnings=warnings,
        )
    if r.disable_torch_compile:
        # L20D reports capability 10.3 (sm_103a), while the pinned Triton
        # wheel's bundled ptxas cannot assemble that target.  Disabling
        # Dynamo before vLLM imports makes its V1 sampler use the eager
        # implementation and avoids a late, non-actionable PTXAS crash.
        print("[OK  ] torch.compile guard: TORCH_COMPILE_DISABLE=1 (eager vLLM sampler for sm_103a compatibility)")
        if r.rollout_cuda_graph:
            # TORCH_COMPILE_DISABLE=1 pins compilation mode to NONE, which
            # silently downgrades any PIECEWISE cudagraph mode to NONE.
            # FULL_DECODE_ONLY does not require piecewise compilation and
            # matches FlashAttention-2's UNIFORM_BATCH cudagraph support.
            print(
                "[OK  ] rollout CUDA graphs: enabled "
                "(enforce_eager=false, cudagraph_mode=FULL_DECODE_ONLY); "
                "decode is captured, prefill stays eager"
            )
    else:
        warning = (
            "torch.compile enabled (advisory): the L20D/ptxas concern was "
            "checked by job 22447 -- a full AIME eval on the compiled "
            "turboquant_k2v3_nc path scored 0.32188 vs 0.30521 eager "
            "(+0.79 SE, no regression) in 2154 s vs 3437 s"
        )
        print(f"[WARN] {warning}")
        warnings.append(warning)

    # The model-specific vLLM patch is deliberately opt-in.  Check its
    # static availability and hard constraints here, while leaving runtime
    # model-hit validation to qat_kv.verl_plugin inside the Ray worker.
    vllm_opt_in_valid = (
        not r.quantize_vllm_rollout
        or (
            r.rollout_engine == "vllm"
            and r.legacy_kv_bits < 32
            and r.vllm_kv_patch_available
        )
    )
    if not r.quantize_vllm_rollout:
        print("[OK  ] vLLM KV patch opt-in: disabled (QAT_KV_VLLM_ENABLE=0)")
    else:
        _status(
            vllm_opt_in_valid,
            "vLLM KV patch opt-in",
            (
                f"QAT_KV_VLLM_ENABLE=1, patch_available={r.vllm_kv_patch_available}, "
                f"engine={r.rollout_engine}, legacy_kv_bits={r.legacy_kv_bits}; "
                "prototype requires Qwen2 + TP=1 + external_launcher + enforce_eager"
            ),
            strict=True,
            warnings=warnings,
        )

    policy_mismatch_fatal = _rollout_policy_mismatch_is_fatal(r, strict=strict)
    if r.rollout_engine == "vllm":
        if r.quantize_vllm_rollout:
            mismatch_detail = (
                "vLLM KV patch explicitly requested: "
                f"status={r.rollout_kv_status}, actor_kv_hook_expected={r.actor_kv_hook_expected}, "
                f"rollout_kv_quantized_expected={r.rollout_kv_quantized}, "
                f"rollout_policy_mismatch_expected={r.rollout_policy_mismatch}. "
                "The Ray-worker plugin must validate a constructed Qwen2 model and record hits; "
                "cache remains floating-point fake/dequant, not packed INT KV."
            )
            policy_ok = (
                r.rollout_kv_quantized
                if r.require_quantized_rollout
                else not policy_mismatch_fatal
            )
            _status(
                policy_ok,
                "rollout KV policy",
                mismatch_detail,
                strict=(
                    strict
                    if (r.require_quantized_rollout or policy_mismatch_fatal)
                    else False
                ),
                warnings=warnings,
            )
            warning = (
                "vLLM rollout patch requested but preflight cannot verify child-worker model hits; "
                "inspect qat_kv_metadata.rollout_kv_status before treating policy identity as proven"
            )
            warnings.append(warning)
            print(f"[WARN] rollout engine: vllm ({mismatch_detail})")
        else:
            mismatch_detail = (
                f"native vLLM TurboQuant cache requested ({r.native_turboquant_kv_cache_dtype}); "
                "the constructed engine must report the same cache dtype before generation"
                if r.native_turboquant_rollout
                else (
                    "vLLM rollout does not execute HF k_proj/v_proj hooks; "
                    f"actor_kv_hook_expected={r.actor_kv_hook_expected}, "
                    f"rollout_kv_quantized={r.rollout_kv_quantized}, "
                    f"rollout_policy_mismatch={r.rollout_policy_mismatch}, exact_opd=false. "
                    "QAT_KV_BITS applies to the HF actor/logprob path only; this is not an "
                    "end-to-end quantized-cache deployment result."
                )
            )
            policy_ok = (
                r.rollout_kv_quantized
                if r.require_quantized_rollout
                else not policy_mismatch_fatal
            )
            _status(
                policy_ok,
                "rollout KV policy",
                (
                    mismatch_detail
                    if not r.require_quantized_rollout
                    else mismatch_detail + "; --require-quantized-rollout rejects this"
                ),
                # The default is an explicit, reviewable advisory because the
                # standard Open-MOPD vLLM path is still useful for a QAT actor
                # study.  Main runs now need the explicit allow switch to
                # retain that diagnostic actor-only behavior.
                strict=(
                    strict
                    if (r.require_quantized_rollout or policy_mismatch_fatal)
                    else False
                ),
                warnings=warnings,
            )
            if not policy_mismatch_fatal and not r.native_turboquant_rollout:
                print("[WARN] rollout engine: vllm (actor hook expected, external rollout unquantized; explicit mismatch metadata)")
    else:
        if r.quantize_vllm_rollout:
            _status(
                False,
                "vLLM KV patch opt-in",
                "--quantize-vllm-rollout requires --rollout-engine vllm; refusing to set QAT_KV_VLLM_ENABLE for HF",
                strict=True,
                warnings=warnings,
            )
        hf_detail = (
            "HF rollout can share the actor hook only if the fork's HF rollout registry is present; "
            f"actor_kv_hook_expected={r.actor_kv_hook_expected}, "
            f"rollout_kv_quantized={r.rollout_kv_quantized}, "
            f"rollout_policy_mismatch={r.rollout_policy_mismatch}"
        )
        policy_ok = (
            r.rollout_kv_quantized
            if r.require_quantized_rollout
            else not policy_mismatch_fatal
        )
        _status(
            policy_ok,
            "rollout KV policy",
            hf_detail,
            strict=(
                strict
                if (r.require_quantized_rollout or policy_mismatch_fatal)
                else False
            ),
            warnings=warnings,
        )
        warning = (
            "HF rollout is an opt-in compatibility path; this pinned Open-MOPD fork does not register hf in its default rollout registry."
        )
        warnings.append(warning)
        print(f"[WARN] rollout engine: hf ({hf_detail}; registry must be verified before --run)")

    _dependency_report(r, strict=strict, warnings=warnings, skip=skip_deps)
    steps_per_epoch = rows // r.train_batch_size if rows is not None else None
    if steps_per_epoch is not None and steps_per_epoch > 0:
        planned_epochs = max(1, math.ceil(r.updates / steps_per_epoch))
        # ``StatefulDataLoader(..., drop_last=True)`` is used by the pinned
        # trainer, so the effective pass count is based on complete batches,
        # not the nominal row count alone.  Report both quantities to avoid
        # silently calling a 150-update run on 16K rows "one epoch".
        consumed_examples = r.updates * r.train_batch_size
        nominal_passes = consumed_examples / rows if rows else float("inf")
        complete_epoch_examples = steps_per_epoch * r.train_batch_size
        dataloader_passes = r.updates / steps_per_epoch
        print(
            "[INFO] data plan: "
            f"{steps_per_epoch} complete optimizer steps/epoch (drop_last), "
            f"updates={r.updates}, trainer.total_epochs={planned_epochs}, "
            f"consumed_examples≈{consumed_examples:,}, nominal_passes≈{nominal_passes:.3f}, "
            f"dataloader_passes≈{dataloader_passes:.3f} "
            f"({complete_epoch_examples:,} usable rows/epoch)"
        )
        one_epoch_ok = r.updates <= steps_per_epoch
        if r.one_epoch:
            _status(
                one_epoch_ok,
                "one-epoch constraint",
                (
                    f"updates={r.updates} fit within {steps_per_epoch} steps"
                    if one_epoch_ok
                    else f"updates={r.updates} exceed one epoch ({steps_per_epoch} steps); "
                    "remove --one-epoch or lower --updates"
                ),
                strict=strict,
                warnings=warnings,
            )
        else:
            print(
                "[INFO] one-epoch guard: not requested; "
                f"updates={r.updates} {'fit within' if one_epoch_ok else 'span beyond'} "
                f"the first {steps_per_epoch}-step drop-last pass"
            )
    elif r.one_epoch:
        _status(
            False,
            "one-epoch constraint",
            "cannot verify rows//train_batch_size because training row count is unknown or zero",
            strict=strict,
            warnings=warnings,
        )
    else:
        print("[INFO] data plan: optimizer steps/epoch unknown until the Parquet dataloader is built")

    # Checkpoint/evaluation schedule is external to the pinned trainer's
    # validation loop.  A requested eval step must either be the final step or
    # a multiple of save_freq; otherwise no checkpoint with that exact global
    # step is guaranteed to exist for the profile's external evaluator.
    _status(
        r.save_freq > 0,
        "checkpoint frequency",
        f"trainer.save_freq={r.save_freq}",
        strict=True,
        warnings=warnings,
    )
    pending_evals = [step for step in r.eval_checkpoints if step <= r.updates]
    missing_ckpts = [
        step for step in pending_evals
        if step != r.updates and step % r.save_freq != 0
    ]
    if missing_ckpts:
        _status(
            False,
            "external eval checkpoints",
            f"steps {missing_ckpts} are not guaranteed by save_freq={r.save_freq}; "
            "choose --save-freq dividing each requested step or adjust --eval-checkpoints",
            strict=strict,
            warnings=warnings,
        )
    else:
        future = [step for step in r.eval_checkpoints if step > r.updates]
        suffix = f"; future steps not in this stage: {future}" if future else ""
        print(
            f"[OK  ] external eval schedule: {list(r.eval_checkpoints)} "
            f"(available in this stage: {pending_evals}; save_freq={r.save_freq}){suffix}"
        )

    # The launcher cannot infer Adam/FSDP optimizer-state precision from a
    # generic Hydra fork.  Never claim FP32 implicitly.  The optional strict
    # switch is intentionally a diagnostic gate and does not inject a dtype
    # override; callers must provide and verify one explicitly in --override.
    fp32_message = (
        "FP32 optimizer-state/teacher arithmetic is not runtime-verified; "
        f"actor initialization contract is model_dtype={r.actor_model_dtype} with "
        f"FSDP param={ACTOR_FSDP_PARAM_DTYPE}, reduce={ACTOR_FSDP_REDUCE_DTYPE}, "
        f"buffer={ACTOR_FSDP_BUFFER_DTYPE}"
    )
    _status(
        (not r.strict_fp32_arithmetic) or actor_fp32_contract,
        "FP32 arithmetic check",
        fp32_message if not r.strict_fp32_arithmetic else fp32_message + "; strict actor contract requested",
        strict=strict,
        warnings=warnings,
    )
    return warnings, steps_per_epoch


def _rollout_skip_layer_args(r: Resolved) -> list[str]:
    """Engine kwargs pinning which layers the rollout leaves unquantized.

    vLLM applies the boundary-skip convention itself, but only for TurboQuant::

        if resolved_cache_dtype.startswith("turboquant_"):
            boundary = TurboQuantConfig.get_boundary_skip_layers(model_config)

    ``int4_per_token_head`` is an upstream preset and gets no such treatment, so
    its engine quantizes all 28 layers while the training-side hook -- which
    uses the shared ``skip_layer_indices`` regardless of preset -- fake-quantizes
    only 24.  That is a train/serve mismatch on a quarter of the boundary
    layers, and it is invisible: both sides run, losses look plausible.  Job
    12394's verifier caught it ("training-side skip layers [0, 1, 26, 27] !=
    engine skip list []").

    Passing the skip list explicitly is also what makes the int4 arm comparable
    to the TurboQuant arms: same layers quantized, only the quantizer differs.

    Returns an empty list for TurboQuant presets, where vLLM already does this
    and a second source of truth could only drift.
    """

    # preflight runs before the launcher exports PYTHONPATH, so the repo is not
    # importable yet (job 12529 died here with ModuleNotFoundError two seconds
    # in).  Import the real thing anyway rather than restating the skip rule --
    # a second copy of it is exactly the drift this function exists to close.
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from qat_kv.int4_ptkh_fake_quant import is_int4_ptkh_scheme
    from qat_kv.turboquant_fake_quant import skip_layer_indices

    dtype = r.native_turboquant_kv_cache_dtype
    if not is_int4_ptkh_scheme(dtype):
        return []

    config_path = Path(r.model) / "config.json"
    if not config_path.is_file():
        raise SystemExit(
            f"{config_path} not found; cannot determine how many layers to skip for "
            f"{dtype}.  Refusing to launch a rollout whose skip set would silently "
            "differ from the actor's."
        )
    config = json.loads(config_path.read_text())
    num_layers = config.get("num_hidden_layers")
    if not isinstance(num_layers, int) or num_layers <= 0:
        raise SystemExit(
            f"{config_path} has num_hidden_layers={num_layers!r}; cannot compute the "
            f"skip set for {dtype}."
        )

    # vLLM compares ``str(layer_idx)`` against the entries, so these must be
    # strings; a list of ints would match nothing and skip no layer at all.
    skip = ",".join(f'"{i}"' for i in skip_layer_indices(num_layers))
    return [
        "+actor_rollout_ref.rollout.engine_kwargs.vllm.kv_cache_dtype_skip_layers="
        f"[{skip}]"
    ]


def _rollout_logprobs_k(r: Resolved) -> int:
    """Width of the top-k vLLM returns from the rollout: the OPD S_logp source.

    Default is ``reward_top_k``, which makes ray_trainer take its native path
    and alias ``student_top_k_log_probs`` straight from
    ``rollout_top_k_log_probs`` -- a vLLM number.  The teacher side of that
    same reward term (``T_on_S``) is an HF forward, so the front K of the union
    reward compares across engines while the back K (``S_on_T - T_logp``) does
    not.

    Job 19427 measured that cross-engine gap at 0.239 nat on k2v3 and showed it
    is not an implementation bug: perturbing only bf16-vs-fp32 inside HF
    reproduces 0.239 through the same quantizer, 7.4x the 0.032 the same
    perturbation produces with the quantizer off.  A 2-bit codebook is a
    discrete map, so a rounding difference flips centroid indices and two
    correct implementations still disagree.  It is irreducible; matching the
    implementations cannot remove it.

    Setting this to 0 withholds the rollout top-k, dropping ray_trainer into
    its ``compute_log_prob`` branch: ``student_top_k_*`` then come from the HF
    actor's own forward, the same forward that already produces ``S_on_T``.
    Both halves of the union reward become same-engine and the quantizer noise
    cancels inside the difference instead of entering it.

    Costs one extra actor forward per step and leaves every other knob alone.
    """
    raw = os.environ.get("MOPD_ROLLOUT_LOGPROBS", "").strip()
    if not raw:
        return int(r.reward_top_k)
    k = int(raw)
    if k < 0:
        raise ValueError(f"MOPD_ROLLOUT_LOGPROBS must be non-negative, got {k}")
    if k not in (0, int(r.reward_top_k)):
        raise ValueError(
            "MOPD_ROLLOUT_LOGPROBS must be 0 (HF actor top-k) or "
            f"{r.reward_top_k} (vLLM rollout top-k); ray_trainer rejects any "
            f"other width against log_prob_top_k, got {k}"
        )
    return k


def _rollout_override(r: Resolved, key: str, value: object) -> str:
    """Build a Hydra override for one OPD rollout option.

    Open-MOPD revisions differ in whether the extra top-k fields are declared
    in ``trainer/config/rollout/rollout.yaml``.  Hydra's struct mode requires
    ``+`` for fields absent from that YAML, while newer revisions accept the
    ordinary path.  Inspect the pinned checkout instead of hard-coding one
    syntax; this keeps the printed command portable and lets the config probe
    catch a genuinely incompatible fork.  The same rule applies to
    ``reward_mode``: if a newer checkout declares it, use the ordinary
    override; otherwise use Hydra's append form.
    """

    declared = False
    rollout_yaml = r.verl_dir / "verl/trainer/config/rollout/rollout.yaml"
    try:
        text = rollout_yaml.read_text(encoding="utf-8")
        declared = re.search(rf"(?m)^\s*{re.escape(key)}\s*:", text) is not None
    except (OSError, UnicodeError):
        # A missing checkout is reported by dependency preflight; use the
        # fail-closed append form here so command construction itself remains
        # deterministic and does not accidentally overwrite another key.
        declared = False
    prefix = "" if declared else "+"
    return f"{prefix}actor_rollout_ref.rollout.{key}={value}"


def _hydra_command(r: Resolved, steps_per_epoch: int | None) -> list[str]:
    """Build the complete command using the official Open-MOPD trainer."""

    planned_epochs = max(1, math.ceil(r.updates / max(1, steps_per_epoch or 1)))
    if r.one_epoch:
        # The preflight rejects an incompatible update budget.  Keeping this
        # explicit here makes the generated command self-documenting and
        # avoids relying on a caller to infer the intent from total_steps.
        epochs = 1
    else:
        epochs = planned_epochs
    # Keep a little slack when a filtered dataset's final dataloader length is
    # not known at dry-run time.  The trainer stops at total_training_steps.
    if steps_per_epoch is None:
        epochs = max(1, epochs)
    native_tq = r.native_turboquant_rollout
    rollout_context_length = r.profile.prompt_length + r.profile.response_length
    # Ray workers do not reliably inherit the shell PATH from a Slurm
    # launcher. Keep the vLLM environment explicit in both the trainer
    # process and Ray runtime_env so subprocesses can resolve ninja. The
    # FlashInfer sampler JIT is disabled because this environment lacks
    # matching Python development headers; TurboQuant KV remains enabled.
    ray_path = os.environ.get("MOPD_RAY_PATH") or os.environ.get("PATH", "")
    python_include = os.environ.get(
        "MOPD_PYTHON_INCLUDE",
        "/historical/shared/qat/tmp/python-devel/root/usr/include/python3.12",
    )
    include_path = os.pathsep.join(
        item for item in (python_include, os.environ.get("CPATH", "")) if item
    )
    # Keep a 32K active scheduler window for long requests; chunked prefill
    # handles the remainder up to max_model_len.
    rollout_enable_chunked_prefill = rollout_context_length > 32768
    rollout_max_num_batched_tokens = min(32768, rollout_context_length)
    env_prefix = [
        "env",
        f"PATH={ray_path}",
        f"CPATH={include_path}",
        f"C_INCLUDE_PATH={include_path}",
        f"CPLUS_INCLUDE_PATH={include_path}",
        "VLLM_USE_FLASHINFER_SAMPLER=0",
        f"PYTHONPATH={_pythonpath(r)}",
        "PYTHONUNBUFFERED=1",
        "TOKENIZERS_PARALLELISM=false",
        *(["VERL_USE_EXTERNAL_MODULES=qat_kv.verl_plugin"] if _needs_legacy_kv_plugin(r) else []),
        *(
            []
            if native_tq
            else [f"QAT_KV_BITS={r.legacy_kv_bits}", "QAT_KV_REQUIRE_BITS=1"]
        ),
        "QAT_KV_LAYERWISE=0",
        f"QAT_KV_PRESERVE_BIAS={'1' if (r.preserve_projection_bias and not native_tq) else '0'}",
        f"QAT_KV_EXACT_BIAS_FREE={'1' if (r.exact_bias_free_recompute and not native_tq) else '0'}",
        f"QAT_KV_KIVI_HF_ENABLE={'1' if (r.kivi_hf_rollout and not native_tq) else '0'}",
        "QAT_KV_KIVI_GROUP_SIZE=32",
        "QAT_KV_KIVI_RESIDUAL_LENGTH=32",
        f"MOPD_OBJECTIVE={MOPD_OBJECTIVE_NAME}",
        f"MOPD_ACTOR_KV_HOOK_EXPECTED={'1' if r.actor_kv_hook_expected else '0'}",
        f"MOPD_ACTOR_MODEL_DTYPE={r.actor_model_dtype}",
        f"MOPD_OPTIMIZER_STATE_EXPECTATION={'fp32_master_params' if r.actor_model_dtype == 'fp32' else 'unverified_non_fp32_model_init'}",
        f"MOPD_ROLLOUT_KV_QUANTIZED={'1' if r.rollout_kv_quantized else '0'}",
        f"MOPD_ROLLOUT_POLICY_MISMATCH={'1' if r.rollout_policy_mismatch else '0'}",
        f"MOPD_ALLOW_ROLLOUT_POLICY_MISMATCH={'1' if r.allow_rollout_policy_mismatch else '0'}",
        f"MOPD_ROLLOUT_KV_STATUS={r.rollout_kv_status}",
        f"MOPD_VLLM_PATCH_REQUESTED={'1' if r.quantize_vllm_rollout else '0'}",
        f"MOPD_NATIVE_TURBOQUANT_ROLLOUT={'1' if native_tq else '0'}",
        f"MOPD_NATIVE_TURBOQUANT_KV_CACHE_DTYPE={r.native_turboquant_kv_cache_dtype or ''}",
        *(
            [f"MOPD_ROLLOUT_KV_VERIFICATION_RECORD={r.output / ROLLOUT_KV_VERIFICATION_FILENAME}"]
            if native_tq
            else []
        ),
        f"MOPD_EVAL_GATE_SUMMARY={r.eval_gate_summary if r.eval_gate_summary is not None else ''}",
        f"MOPD_REQUIRE_EVAL_GATE={'1' if r.eval_gate_required_for_run else '0'}",
        f"MOPD_ALLOW_MISSING_EVAL_GATE={'1' if r.allow_missing_eval_gate else '0'}",
        f"QAT_KV_VLLM_ENABLE={'1' if r.quantize_vllm_rollout and not native_tq else '0'}",
        # vLLM 0.11 may expose TP=1 external_launcher as ``uni`` after
        # normalization.  This explicit marker distinguishes that governed
        # path from an ordinary standalone uni engine; force it to 0 when the
        # patch is disabled so an inherited setting cannot be trusted.
        f"QAT_KV_VLLM_EXTERNAL_LAUNCHER={'1' if r.quantize_vllm_rollout and not native_tq else '0'}",
        # Keep the runtime policy gate separate from the patch opt-in.  The
        # latter only installs a class seam; this flag makes the worker reject
        # a child/process path that never exposes a validated model hit.
        f"QAT_KV_REQUIRE_QUANTIZED_ROLLOUT={'1' if r.rollout_quantized_required_for_run else '0'}",
        f"TORCH_COMPILE_DISABLE={'1' if r.disable_torch_compile else '0'}",
        f"TORCHDYNAMO_DISABLE={'1' if r.disable_torch_compile else '0'}",
        # The pinned reward worker historically hard-coded flash-attn.  The
        # external shim reads this non-secret switch and temporarily selects
        # the requested backend while building the frozen teacher.  SDPA is
        # the safe 16K default: eager Qwen2 attention materializes a
        # 1 x 28 x 16384 x 16384 FP32 softmax tensor (28 GiB) per layer.
        f"QAT_KV_SCHEME={r.kv_scheme or ''}",
        f"QAT_KV_REWARD_ATTN={r.reward_attention}",
        f"QAT_OPD_POSITION_RAMP={os.environ.get('QAT_OPD_POSITION_RAMP', '')}",
        f"QAT_OPD_POSITION_RAMP_NORMALIZE={os.environ.get('QAT_OPD_POSITION_RAMP_NORMALIZE', '0')}",
        f"QAT_BLOCK_ALIGN_STATIC_REF={os.environ.get('QAT_BLOCK_ALIGN_STATIC_REF', '1')}",
        f"QAT_SKIP_TEACHER_ENTROPY={os.environ.get('QAT_SKIP_TEACHER_ENTROPY', '0')}",
        f"QAT_PROBE_OBJECTIVE={os.environ.get('QAT_PROBE_OBJECTIVE', 'both')}",
        f"QAT_GRAD_PROBE={os.environ.get('QAT_GRAD_PROBE', '0')}",
        f"SWANLAB_MODE={r.swanlab_mode}",
        f"SWANLAB_LOG_DIR={r.output / 'swanlog'}",
    ]
    command = env_prefix + [str(r.python), "-m", "verl.trainer.main_ppo"]
    # Ray's pinned YAML leaves ``num_cpus`` null (all visible CPUs), which is
    # unsafe on this cluster.  Keep its session files on the shared QAT
    # volume.  Do not pass ``object_spilling_config`` as a top-level
    # ``ray.init`` kwarg: Ray 2.58 rejects that spelling (the setting belongs
    # to its system config in some releases).  The explicit spill directory
    # is still created/recorded for provenance, while this command lets Ray
    # use its version-compatible default spill location beneath ``_temp_dir``.
    overrides = [
        "algorithm.adv_estimator=token_reward_direct",
        "algorithm.gamma=1.0",
        "algorithm.lam=1.0",
        "algorithm.norm_adv_by_std_in_grpo=false",
        "algorithm.use_kl_in_reward=false",
        # Dense token rewards are consumed directly by PPO; no value model is
        # needed (and leaving this null would make the trainer infer it from
        # the estimator, which is easy to misread in a dry-run).
        "critic.enable=false",
        f"ray_kwargs.ray_init.num_cpus={r.ray_num_cpus}",
        f"+ray_kwargs.ray_init._temp_dir={r.ray_temp_dir}",
        f"+ray_kwargs.ray_init.runtime_env.env_vars.PATH={ray_path}",
        f'+ray_kwargs.ray_init.runtime_env.env_vars.CPATH="{include_path}"',
        f'+ray_kwargs.ray_init.runtime_env.env_vars.C_INCLUDE_PATH="{include_path}"',
        f'+ray_kwargs.ray_init.runtime_env.env_vars.CPLUS_INCLUDE_PATH="{include_path}"',
        '+ray_kwargs.ray_init.runtime_env.env_vars.VLLM_USE_FLASHINFER_SAMPLER="0"',
        f'+ray_kwargs.ray_init.runtime_env.env_vars.QAT_PROBE_OBJECTIVE="{os.environ.get("QAT_PROBE_OBJECTIVE", "both")}"',
        f'+ray_kwargs.ray_init.runtime_env.env_vars.QAT_GRAD_PROBE="{os.environ.get("QAT_GRAD_PROBE", "0")}"',
        # Read by fsdp_workers.get_sharding_strategy inside the worker.  Empty
        # keeps torch FSDP's FULL_SHARD (ZeRO-3); SHARD_GRAD_OP is ZeRO-2,
        # which stops resharding parameters between micro-batches.
        f'+ray_kwargs.ray_init.runtime_env.env_vars.QAT_FSDP_SHARDING_STRATEGY="{os.environ.get("QAT_FSDP_SHARDING_STRATEGY", "")}"',
        f'+ray_kwargs.ray_init.runtime_env.env_vars.QAT_OPD_POSITION_RAMP="{os.environ.get("QAT_OPD_POSITION_RAMP", "")}"',
        f'+ray_kwargs.ray_init.runtime_env.env_vars.QAT_OPD_POSITION_RAMP_NORMALIZE="{os.environ.get("QAT_OPD_POSITION_RAMP_NORMALIZE", "0")}"',
        f'+ray_kwargs.ray_init.runtime_env.env_vars.QAT_BLOCK_ALIGN_STATIC_REF="{os.environ.get("QAT_BLOCK_ALIGN_STATIC_REF", "1")}"',
        f'+ray_kwargs.ray_init.runtime_env.env_vars.QAT_SKIP_TEACHER_ENTROPY="{os.environ.get("QAT_SKIP_TEACHER_ENTROPY", "0")}"',
        f"data.train_files={r.train}",
        f"data.val_files={r.val}",
        "data.prompt_key=prompt",
        "data.train_max_samples=-1",
        "data.shuffle=false",
        "data.seed=42",
        f"data.max_prompt_length={r.profile.prompt_length}",
        f"data.max_response_length={r.profile.response_length}",
        f"data.train_batch_size={r.train_batch_size}",
        "data.filter_overlong_prompts=true",
        "data.truncation=error",
        "data.val_batch_size=32",
        f"actor_rollout_ref.model.path={r.model}",
        "actor_rollout_ref.model.trust_remote_code=true" if r.trust_remote_code else "actor_rollout_ref.model.trust_remote_code=false",
        *(["actor_rollout_ref.model.external_lib=qat_kv.verl_plugin"] if _needs_legacy_kv_plugin(r) else []),
        # The KV hook attaches to the HF k_proj/v_proj modules and is
        # independent of the attention kernel.  Use memory-efficient SDPA by
        # default because eager attention is quadratic.  Long-response
        # profiles enable the model's existing gradient-checkpointing path;
        # without it a 16K actor update can exceed one GPU even at micro-batch
        # size one.  Compile and fused/remove-padding paths remain disabled.
        (
            "actor_rollout_ref.model.enable_gradient_checkpointing="
            f"{'true' if (rollout_context_length >= 32768 or r.profile.response_length >= 16384) else 'false'}"
        ),
        "actor_rollout_ref.model.use_remove_padding=false",
        "actor_rollout_ref.model.use_fused_kernels=false",
        f"+actor_rollout_ref.model.override_config.attn_implementation={r.actor_attention}",
        "actor_rollout_ref.actor.strategy=fsdp",
        # Request the pinned worker's FP32 actor initialization contract. The
        # mixed policy asks for BF16 forward parameters and FP32
        # reductions/buffers; exact parameter/optimizer residency is recorded
        # as an expectation and must be checked from runtime telemetry. FP32
        # storage can be roughly 2x BF16 (before sharding), with optimizer
        # master/moment state adding more peak memory.
        f"actor_rollout_ref.actor.fsdp_config.model_dtype={r.actor_model_dtype}",
        (
            "+actor_rollout_ref.actor.fsdp_config.mixed_precision="
            f"{{param_dtype:{ACTOR_FSDP_PARAM_DTYPE},"
            f"reduce_dtype:{ACTOR_FSDP_REDUCE_DTYPE},"
            f"buffer_dtype:{ACTOR_FSDP_BUFFER_DTYPE}}}"
        ),
        "actor_rollout_ref.actor.fsdp_config.use_torch_compile=false",
        "actor_rollout_ref.actor.use_torch_compile=false",
        f"actor_rollout_ref.actor.optim.lr={r.lr:g}",
        "actor_rollout_ref.actor.optim.weight_decay=0.0",
        "actor_rollout_ref.actor.optim.lr_warmup_steps=5",
        "actor_rollout_ref.actor.optim.lr_scheduler_type=constant",
        f"actor_rollout_ref.actor.ppo_mini_batch_size={r.ppo_mini_batch_size}",
        f"actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu={r.micro_batch_size_per_gpu}",
        "actor_rollout_ref.actor.ppo_epochs=1",
        # Match Open-MOPD's asymmetric PPO clipping: lower 0.20, upper 0.28.
        # Keep clip_ratio at the lower bound for older VeRL code paths.
        "actor_rollout_ref.actor.clip_ratio=0.2",
        "actor_rollout_ref.actor.clip_ratio_low=0.2",
        "actor_rollout_ref.actor.clip_ratio_high=0.28",
        "actor_rollout_ref.actor.loss_agg_mode=token-mean",
        "actor_rollout_ref.actor.use_kl_loss=false",
        *(
            [
                f"actor_rollout_ref.actor.block_align_coef={r.block_align_coef}",
                f"actor_rollout_ref.actor.block_align_layer_every={r.block_align_layer_every}",
                f"actor_rollout_ref.actor.block_align_keep_fraction={r.block_align_keep_fraction}",
                f"actor_rollout_ref.actor.block_align_metric={r.block_align_metric}",
                f"+actor_rollout_ref.actor.block_align_last_only={str(r.block_align_last_only).lower()}",
            ]
            if r.block_align_coef > 0
            else []
        ),
        *(
            [
                f"actor_rollout_ref.actor.kv_align_coef={r.kv_align_coef}",
                f"actor_rollout_ref.actor.kv_align_layer_every={r.kv_align_layer_every}",
                f"actor_rollout_ref.actor.kv_align_keep_fraction={r.kv_align_keep_fraction}",
                f"actor_rollout_ref.actor.kv_align_metric={r.kv_align_metric}",
                f"actor_rollout_ref.actor.kv_align_align_keys={str(r.kv_align_align_keys).lower()}",
                f"actor_rollout_ref.actor.kv_align_align_values={str(r.kv_align_align_values).lower()}",
            ]
            if r.kv_align_coef > 0
            else []
        ),
        f"actor_rollout_ref.actor.ppo_max_token_len_per_gpu={r.profile.prompt_length + r.profile.response_length}",
        "actor_rollout_ref.rollout.name=vllm" if r.rollout_engine == "vllm" else "actor_rollout_ref.rollout.name=hf",
        "actor_rollout_ref.rollout.mode=sync",
        f"actor_rollout_ref.rollout.temperature={r.temperature}",
        f"actor_rollout_ref.rollout.top_k={r.sampling_top_k}",
        f"actor_rollout_ref.rollout.top_p={r.sampling_top_p}",
        "actor_rollout_ref.rollout.do_sample=true",
        "actor_rollout_ref.rollout.n=1",
        f"actor_rollout_ref.rollout.prompt_length={r.profile.prompt_length}",
        f"actor_rollout_ref.rollout.response_length={r.profile.response_length}",
        f"actor_rollout_ref.rollout.max_model_len={r.profile.prompt_length + r.profile.response_length}",
        "actor_rollout_ref.rollout.tensor_model_parallel_size=1",
        f"actor_rollout_ref.rollout.gpu_memory_utilization={r.rollout_gpu_memory_utilization}",
        "actor_rollout_ref.rollout.enforce_eager="
        f"{'false' if r.rollout_cuda_graph else 'true'}",
        *(
            [
                # Passed straight through to LLM(**engine_kwargs); verl only
                # builds its own compilation_config when cudagraph_capture_sizes
                # is set (which we do not set), so there is no duplicate kwarg.
                "+actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config="
                "{cudagraph_mode:FULL_DECODE_ONLY}",
            ]
            if r.rollout_cuda_graph
            else []
        ),
        f"actor_rollout_ref.rollout.max_num_batched_tokens={rollout_max_num_batched_tokens}",
        f"actor_rollout_ref.rollout.enable_chunked_prefill={'true' if rollout_enable_chunked_prefill else 'false'}",
        "actor_rollout_ref.rollout.enable_prefix_caching=false",
        "actor_rollout_ref.rollout.load_format=dummy",
        # The pinned HF rollout batches the whole incoming DataProto unless
        # this legacy field is present.  Long-context training must generate
        # one prompt at a time; vLLM has its own scheduler and ignores it.
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="
        f"{r.log_prob_micro_batch_size_per_gpu}",
        *(
            [
                "+actor_rollout_ref.rollout.engine_kwargs.vllm.kv_cache_dtype="
                f"{r.native_turboquant_kv_cache_dtype}",
                *_rollout_skip_layer_args(r),
                # vLLM defaults to exposing at most 20 sample logprobs.  OPD
                # needs the requested student Top-64 candidates from the
                # same quantized rollout, so raise the engine-side limit to
                # the exact value carried through the VeRL DataProto.
                f"+actor_rollout_ref.rollout.engine_kwargs.vllm.max_logprobs={r.reward_top_k}",
                _rollout_override(r, "calculate_log_probs", "true"),
                _rollout_override(r, "logprobs", str(_rollout_logprobs_k(r))),
                "+algorithm.rollout_correction.bypass_old_logprob_for_rollout=true",
            ]
            if native_tq
            else []
        ),
        _rollout_override(r, "reward_mode", "opd_kl"),
        _rollout_override(r, "log_prob_top_k", r.reward_top_k),
        _rollout_override(r, "top_k_strategy", r.top_k_strategy),
        _rollout_override(r, "reward_weight_mode", r.reward_weight_mode),
        _rollout_override(r, "teacher_temperature", "1.0"),
        _rollout_override(r, "top_p_intersec_p", "0.99"),
        "reward_model.enable=true",
        "reward_model.enable_resource_pool=false",
        "reward_model.strategy=fsdp",
        # This is a distillation-only run.  The reward-model worker has already
        # materialized the MOPD ``rm_scores`` tensor; invoking the generic
        # rule-reward dispatcher afterwards is both redundant and unsafe for
        # arbitrary dataset identifiers (e.g. Skywork's
        # ``train-math-deepscaler``).  NaiveRewardManager accepts this flag and
        # returns rm_scores without decoding/scoring the response.  Keep the
        # override nested under reward_kwargs because that is the constructor
        # interface used by load_reward_manager; the leading ``+`` is needed
        # on the pinned fork, whose base reward YAML does not declare the map.
        "+reward_model.reward_kwargs.compute_true_reward=false",
        "reward_model.model.path=" + str(r.teacher),
        # RewardModelWorker currently defaults to BF16; keep the frozen
        # teacher precision explicit in the generated VeRL command.
        "+reward_model.model.dtype=bf16",
        "reward_model.model.input_tokenizer=null",
        # The pinned RewardModelWorker now reads this direct field; keep the
        # external env shim below as a compatibility fallback for older forks
        # whose worker still hard-codes flash-attn.  ``override_config`` is
        # also retained for forks that expose that nesting instead.
        f"+reward_model.model.attn_implementation={r.reward_attention}",
        *(["reward_model.model.external_lib=qat_kv.verl_plugin"] if _needs_legacy_kv_plugin(r) else []),
        "reward_model.model.trust_remote_code=true",
        f"+reward_model.model.override_config.attn_implementation={r.reward_attention}",
        "reward_model.model.use_remove_padding=false",
        "reward_model.model.use_fused_kernels=false",
        "reward_model.model.fsdp_config.param_offload=false",
        "reward_model.use_dynamic_bsz=false",
        f"reward_model.forward_max_token_len_per_gpu={r.profile.prompt_length + r.profile.response_length}",
        f"reward_model.micro_batch_size_per_gpu={r.reward_micro_batch_size_per_gpu}",
        f"trainer.nnodes={r.nodes}",
        f"trainer.n_gpus_per_node={r.gpus}",
        f"trainer.total_training_steps={r.updates}",
        f"trainer.total_epochs={epochs}",
        "trainer.val_before_train=false",
        "trainer.test_freq=-1",
        f"trainer.save_freq={r.save_freq}",
        # Keep disabled/offline smoke runs independent of the optional
        # tracker.  Cloud is the production profile and explicitly enables
        # SwanLab so its metrics (including candidate-count plots) are kept.
        "trainer.logger=[console,swanlab]" if r.swanlab_mode == "cloud" else "trainer.logger=[console]",
        f"trainer.project_name={r.project_name}",
        f"trainer.experiment_name={r.experiment_name}",
        f"trainer.default_local_dir={r.output / 'checkpoints'}",
        "trainer.rollout_data_dir=" + str(r.output / "rollouts"),
        "trainer.validation_data_dir=" + str(r.output / "validation"),
        "trainer.resume_mode=auto" if r.resume else "trainer.resume_mode=disable",
        # Metadata consumed by diagnostics/plugins; reward/PPO code remains the
        # official Open-MOPD implementation above.  ``objective_legacy`` is a
        # compatibility label only: no trainer branch reads it.
        f"+mopd.objective={MOPD_OBJECTIVE_NAME}",
        "+mopd.teacher_count=1",
        "+mopd.teacher_routing=single_frozen_teacher",
        f"+mopd.objective_legacy={MOPD_OBJECTIVE_LEGACY_NAME}",
        "+mopd.objective_legacy_aliases=[topk_student_weighted_teacher_delta,student_topk_mopd_reward,mopd_topk_ppo]",
        "+mopd.objective_legacy_metadata_only=true",
        f"+mopd.reward_top_k_source={MOPD_REWARD_TOP_K_SOURCE}",
        f"+mopd.reward_top_k_source_legacy={MOPD_REWARD_TOP_K_SOURCE_LEGACY}",
        "+mopd.reward_top_k_source_metadata_only=true",
        "+mopd.reward_candidate_truncation=none",
        "+mopd.reward_top_p_applies=false",
        "+mopd.reward_top_p_scope=only_stu_no_extra_filter",
        "+mopd.reward_top_p_intersec_p=0.99",
        f"+mopd.sampling_top_k={r.sampling_top_k}",
        f"+mopd.sampling_top_p={r.sampling_top_p}",
        f"+mopd.reward_top_k={r.reward_top_k}",
        f"+mopd.top_k_strategy={r.top_k_strategy}",
        f"+mopd.reward_weight_mode={r.reward_weight_mode}",
        *([] if native_tq else [f"+mopd.legacy_kv_bits={r.legacy_kv_bits}"]),
        f"+mopd.kv_scheme={r.kv_scheme or 'none'}",
        f"+mopd.actor_lr={r.lr:g}",
        f"+mopd.block_align_coef={r.block_align_coef}",
        f"+mopd.kv_align_coef={r.kv_align_coef}",
        f"+mopd.actor_fake_quant={'true' if r.kv_scheme and r.kv_scheme != 'symmetric' else 'false'}",
        f"+mopd.kivi_hf_rollout={'true' if r.kivi_hf_rollout else 'false'}",
        f"+mopd.kivi_group_size=32",
        f"+mopd.kivi_residual_length=32",
        f"+mopd.preserve_projection_bias={'true' if r.preserve_projection_bias else 'false'}",
        f"+mopd.exact_bias_free_recompute={'true' if r.exact_bias_free_recompute else 'false'}",
        f"+mopd.hf_bias_preservation_mode={('exact_bias_free_recompute' if r.exact_bias_free_recompute else 'post_forward_approximate' if r.preserve_projection_bias else 'disabled')}",
        f"+mopd.global_batch={r.train_batch_size}",
        f"+mopd.ray_num_cpus={r.ray_num_cpus}",
        "+mopd.ray_spill_dir_wired=false",
        "+mopd.ray_spill_policy=ray_default_under_temp_dir",
        f"+mopd.disable_torch_compile={'true' if r.disable_torch_compile else 'false'}",
        f"+mopd.rollout_cuda_graph={'true' if r.rollout_cuda_graph else 'false'}",
        f"+mopd.rollout_cudagraph_mode="
        f"{'FULL_DECODE_ONLY' if r.rollout_cuda_graph else 'NONE'}",
        f"+mopd.actor_model_dtype={r.actor_model_dtype}",
        f"+mopd.actor_fsdp_param_dtype={ACTOR_FSDP_PARAM_DTYPE}",
        f"+mopd.actor_fsdp_reduce_dtype={ACTOR_FSDP_REDUCE_DTYPE}",
        f"+mopd.actor_fsdp_buffer_dtype={ACTOR_FSDP_BUFFER_DTYPE}",
        f"+mopd.optimizer_state_expectation={'fp32_master_params' if r.actor_model_dtype == 'fp32' else 'unverified_non_fp32_model_init'}",
        f"+mopd.actor_fp32_storage_memory_warning={'true' if r.actor_model_dtype == 'fp32' else 'false'}",
        f"+mopd.actor_attention={r.actor_attention}",
        f"+mopd.reward_attention={r.reward_attention}",
        f"+mopd.gate_updates=20",
        f"+mopd.max_updates={r.updates}",
        f"+mopd.actor_kv_hook_expected={'true' if r.actor_kv_hook_expected else 'false'}",
        f"+mopd.actor_kv_status={'expected_fake_dequant' if r.actor_kv_hook_expected else 'disabled_32'}",
        f"+mopd.vllm_kv_patch_requested={'true' if r.quantize_vllm_rollout else 'false'}",
        f"+mopd.vllm_kv_patch_available={'true' if r.vllm_kv_patch_available else 'false'}",
        f"+mopd.native_turboquant_rollout={'true' if native_tq else 'false'}",
        f"+mopd.native_turboquant_kv_cache_dtype={r.native_turboquant_kv_cache_dtype or 'none'}",
        f"+mopd.rollout_student_logprob_source="
        f"{('vllm_native_turboquant' if _rollout_logprobs_k(r) else 'hf_actor_same_engine') if native_tq else 'hf_or_rollout'}",
        "+mopd.teacher_kv_cache_dtype=bf16",
        f"+mopd.rollout_kv_runtime_verification="
        f"{'required_engine_structural_proof' if native_tq else 'plugin_or_registry'}",
        f"+mopd.rollout_kv_status={r.rollout_kv_status}",
        f"+mopd.rollout_kv_quantized_expected={'true' if r.rollout_kv_quantized else 'false'}",
        f"+mopd.rollout_kv_quantized={'true' if r.rollout_kv_quantized else 'false'}",
        f"+mopd.rollout_policy_mismatch={'true' if r.rollout_policy_mismatch else 'false'}",
        f"+mopd.rollout_policy_mismatch_expected={'true' if r.rollout_policy_mismatch else 'false'}",
        f"+mopd.rollout_quantized_required_for_run={'true' if r.rollout_quantized_required_for_run else 'false'}",
        f"+mopd.allow_rollout_policy_mismatch={'true' if r.allow_rollout_policy_mismatch else 'false'}",
        # Native TurboQuant is verified inside the rollout worker after the
        # engine is built (qat_kv.native_tq_verify); preflight cannot know
        # the answer yet, and must not claim a definitive "false".
        f"+mopd.rollout_kv_verified={'pending_runtime_engine_proof' if native_tq else 'false'}",
        *(
            [
                "+mopd.rollout_kv_verification_record="
                f"{r.output / ROLLOUT_KV_VERIFICATION_FILENAME}"
            ]
            if native_tq
            else []
        ),
        "+mopd.exact_opd=false",
        f"+mopd.eval_gate_summary={r.eval_gate_summary if r.eval_gate_summary is not None else 'none'}",
        f"+mopd.require_eval_gate={'true' if r.eval_gate_required_for_run else 'false'}",
        f"+mopd.allow_missing_eval_gate={'true' if r.allow_missing_eval_gate else 'false'}",
        "+mopd.fp32_arithmetic_verified=false",
        f"+mopd.steps_per_epoch={steps_per_epoch if steps_per_epoch is not None else 'unknown'}",
        f"+mopd.planned_dataset_passes={(r.updates / steps_per_epoch):.6f}" if steps_per_epoch else "+mopd.planned_dataset_passes=unknown",
        "+mopd.planned_pass_definition=updates_over_drop_last_steps_per_epoch",
        f"+mopd.planned_trainer_epochs={planned_epochs}",
        f"+mopd.one_epoch={'true' if r.one_epoch else 'false'}",
        f"+mopd.save_freq={r.save_freq}",
        "+mopd.eval_checkpoints=" + "[" + ",".join(str(x) for x in r.eval_checkpoints) + "]",
        f"+mopd.gradient_accumulation_steps={r.gradient_accumulation_steps}",
        "+mopd.teacher_frozen=true",
        "+mopd.no_critic=true",
    ]
    if r.resume_path:
        overrides.append(f"trainer.resume_from_path={r.resume_path}")
        overrides.append("trainer.resume_mode=resume_path")
    command.extend(overrides)
    command.extend(r.extra_overrides)
    return command


def _print_command(command: Sequence[str], *, swanlab_status: str = "unknown") -> None:
    print("[mopd] complete command (safe/redacted):")
    # `env` assignments are printed one per token; this is both copy/pasteable
    # and makes the absence of SWANLAB_API_KEY explicit.
    print(" ".join(_shell_quote(_safe_text(str(item))) for item in command))
    if swanlab_status in {"inherited", "autokey-line6"}:
        print(f"[mopd] SWANLAB_API_KEY: <{swanlab_status}; redacted>")
    elif swanlab_status == "not-required":
        print("[mopd] SWANLAB_API_KEY: not required for this mode")
    else:
        print("[mopd] SWANLAB_API_KEY: unavailable (cloud launch will be refused)")


def _write_effective_manifest(path: Path, summary: dict[str, Any]) -> None:
    """Atomically write a non-secret effective-settings manifest.

    This is intentionally called only from the real ``--run`` branch after
    strict preflight and output-directory creation.  Dry-runs never create a
    manifest (or any output directory).  The command stored in ``summary`` is
    already redacted; we defensively redact each assignment again here so a
    caller-supplied ``--override ...api_key=...`` cannot leak into the file.
    """

    path = path.expanduser().resolve(strict=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    # The caller has already classified the output as fresh, but another
    # process could create it between that check and this write.  Refuse an
    # overwrite rather than silently replacing provenance from a concurrent
    # run; ``--resume`` is the explicit path for an existing artifact.
    if path.exists():
        raise FileExistsError(f"effective-settings manifest already exists: {path}")
    payload = dict(summary)
    payload.update(
        {
            "manifest_schema_version": 1,
            "manifest_kind": "mopd_effective_settings",
            "run_mode": "run",
            "dry_run": False,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "secret_policy": {
                "swanlab_api_key": "in_child_environment_only",
                "command_contains_api_key": False,
            },
        }
    )
    payload["command"] = [_safe_text(str(item)) for item in payload.get("command", ())]
    payload["effective_manifest"] = str(path)
    payload["effective_manifest_written"] = True

    # Use an explicit temporary file in the destination directory so the
    # replace is atomic even when the shared filesystem is mounted on several
    # nodes.  Never put the credential in this payload or its error text.
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


def _hydra_config_probe(
    command: Sequence[str],
    r: Resolved,
    *,
    strict: bool,
    warnings: list[str],
) -> None:
    """Compose the exact Hydra job without entering ``main``/Ray training.

    ``--cfg job`` is Hydra's supported configuration-only mode.  Running it
    catches misspelled custom fields (notably the ``+actor_rollout_ref`` MOPD
    keys) while avoiding model loading and optimizer creation.  A temporary
    Hydra run directory prevents this read-only check from scattering
    ``outputs/`` or ``.hydra/`` files into the reference checkout.
    """

    try:
        module_index = command.index("verl.trainer.main_ppo")
    except ValueError:
        _status(False, "Hydra config", "trainer module missing from generated command", strict=strict, warnings=warnings)
        return
    probe_command = list(command)
    probe_command[module_index + 1 : module_index + 1] = ["--cfg", "job"]
    temp_dir = tempfile.mkdtemp(prefix="mopd-hydra-")
    probe_command.extend([f"hydra.run.dir={temp_dir}", "hydra.output_subdir=null", "hydra.job.chdir=false"])
    env = os.environ.copy()
    env.update(
        {
            "PYTHONPATH": _pythonpath(r),
            "VERL_USE_EXTERNAL_MODULES": "qat_kv.verl_plugin" if _needs_legacy_kv_plugin(r) else "",
            **(
                {}
                if r.native_turboquant_rollout
                else {
                    "QAT_KV_BITS": str(r.legacy_kv_bits),
                    "QAT_KV_REQUIRE_BITS": "1",
                }
            ),
            "QAT_KV_LAYERWISE": "0",
            "QAT_KV_PRESERVE_BIAS": "1" if (r.preserve_projection_bias and not r.native_turboquant_rollout) else "0",
            "QAT_KV_EXACT_BIAS_FREE": "1" if (r.exact_bias_free_recompute and not r.native_turboquant_rollout) else "0",
            "TORCH_COMPILE_DISABLE": "1" if r.disable_torch_compile else "0",
            "TORCHDYNAMO_DISABLE": "1" if r.disable_torch_compile else "0",
            "QAT_KV_SCHEME": r.kv_scheme or "",
            "QAT_KV_REWARD_ATTN": r.reward_attention,
            # Read inside the Ray workers by dp_actor / fsdp_workers; without
            # these the ramp and the static block-align reference would be
            # silently off in the workers while the launcher reported them on.
            "QAT_OPD_POSITION_RAMP": os.environ.get("QAT_OPD_POSITION_RAMP", ""),
            "QAT_OPD_POSITION_RAMP_NORMALIZE": os.environ.get("QAT_OPD_POSITION_RAMP_NORMALIZE", "0"),
            "QAT_BLOCK_ALIGN_STATIC_REF": os.environ.get("QAT_BLOCK_ALIGN_STATIC_REF", "1"),
            "QAT_SKIP_TEACHER_ENTROPY": os.environ.get("QAT_SKIP_TEACHER_ENTROPY", "0"),
            "QAT_PROBE_OBJECTIVE": os.environ.get("QAT_PROBE_OBJECTIVE", "both"),
            "QAT_GRAD_PROBE": os.environ.get("QAT_GRAD_PROBE", "0"),
            "QAT_FSDP_SHARDING_STRATEGY": os.environ.get("QAT_FSDP_SHARDING_STRATEGY", ""),
            "QAT_KV_VLLM_ENABLE": "1" if (r.quantize_vllm_rollout and not r.native_turboquant_rollout) else "0",
            "QAT_KV_VLLM_EXTERNAL_LAUNCHER": "1" if (r.quantize_vllm_rollout and not r.native_turboquant_rollout) else "0",
            "QAT_KV_REQUIRE_QUANTIZED_ROLLOUT": (
                "1" if r.rollout_quantized_required_for_run else "0"
            ),
            "MOPD_REQUIRE_EVAL_GATE": "1" if r.eval_gate_required_for_run else "0",
            "MOPD_ALLOW_MISSING_EVAL_GATE": (
                "1" if r.allow_missing_eval_gate else "0"
            ),
            "MOPD_ALLOW_ROLLOUT_POLICY_MISMATCH": (
                "1" if r.allow_rollout_policy_mismatch else "0"
            ),
            # ``--cfg job`` should be read-only even on a busy node.  Hide
            # GPUs and cap BLAS/OpenMP fan-out so importing the pinned fork
            # cannot reserve CUDA memory or spawn hundreds of CPU threads.
            "CUDA_VISIBLE_DEVICES": "",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "HYDRA_FULL_ERROR": "1",
        }
    )
    try:
        result = subprocess.run(
            probe_command,
            cwd=str(r.verl_dir),
            env=env,
            check=False,
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode == 0:
            print("[OK  ] Hydra config: --cfg job composition succeeded (training not entered)")
        else:
            detail = (result.stderr or result.stdout).strip().splitlines()
            excerpt = " | ".join(detail[-6:]) if detail else f"exit={result.returncode}"
            _status(False, "Hydra config", excerpt, strict=strict, warnings=warnings)
    except Exception as exc:
        _status(False, "Hydra config", str(exc), strict=strict, warnings=warnings)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--profile", choices=sorted(_profiles()), default="deepseek_skywork")
    parser.add_argument("--stage", choices=("gate", "main"), default="gate")
    parser.add_argument("--updates", type=_positive_int, default=None)
    parser.add_argument("--nodes", type=_positive_int, default=1)
    parser.add_argument("--gpus", type=_positive_int, default=8)
    parser.add_argument(
        "--ray-num-cpus",
        type=_positive_int,
        # ``None`` means a topology-aware cap is derived in ``_resolve``:
        # one CPU per GPU worker plus one TaskRunner CPU on the head node.
        # This keeps the null/all-CPU default from pre-starting hundreds of
        # workers while still supporting the documented two-node profile.
        default=None,
        help="fixed Ray CPU capacity (auto: nodes*GPUs+1; must be >= that floor)",
    )
    parser.add_argument(
        "--ray-temp-dir",
        help="Ray _temp_dir (defaults to a short hashed /historical/shared/qat/ray/.../tmp; must be under /shared for --run)",
    )
    parser.add_argument(
        "--ray-spill-dir",
        help="Ray filesystem object-spilling directory (defaults to the matching short hashed /historical/shared/qat/ray/.../spill)",
    )
    # The active experiments use vLLM's native KV-cache implementation.  Keep
    # the old HF/KIVI path available only when it is explicitly requested;
    # an omitted option must never silently launch that deprecated path.
    parser.add_argument("--kv-bits", type=int, choices=(2, 4, 8, 16, 32), default=32)
    parser.add_argument(
        "--block-align-coef",
        type=float,
        default=0.0,
        help=(
            "weight of the per-block alignment between the quantized and "
            "full-precision forwards (0 disables). Requires --kv-scheme; the "
            "reference is the same weights with the quantizer switched off, "
            "so it costs one extra forward and no extra model copy."
        ),
    )
    parser.add_argument("--block-align-layer-every", type=int, default=4,
                        help="align every N-th block (activation memory scales with 1/N)")
    parser.add_argument("--block-align-keep-fraction", type=float, default=0.125,
                        help="fraction of response positions scored per step")
    parser.add_argument("--block-align-metric", default="cosine", choices=("cosine", "nmse"),
                        help="scale-free distance between block outputs")
    parser.add_argument("--block-align-last-only", action="store_true",
                        help="align only the final layer's hidden state (pre-logits "
                             "representation) instead of every layer_every-th block; "
                             "the 'fit the last hidden state only' variant")
    parser.add_argument(
        "--kv-align-coef",
        type=float,
        default=0.0,
        help=(
            "weight of the per-layer alignment between the student's quantized "
            "K/V and the frozen teacher's full-precision K/V (0 disables). "
            "Requires --kv-scheme; the reference is the frozen theta_0 copy "
            "with the quantizer bypassed."
        ),
    )
    parser.add_argument("--kv-align-layer-every", type=int, default=1,
                        help="align every N-th quantized layer (activation memory scales with 1/N)")
    parser.add_argument("--kv-align-keep-fraction", type=float, default=1.0,
                        help="fraction of response positions scored per step")
    parser.add_argument("--kv-align-metric", default="nmse", choices=("nmse", "cosine", "mse"),
                        help="distance between K/V vectors")
    parser.add_argument("--kv-align-align-keys", action="store_true", default=True,
                        help="score the key stream (default)")
    parser.add_argument("--kv-align-no-align-keys", dest="kv_align_align_keys",
                        action="store_false", help="do not score the key stream")
    parser.add_argument("--kv-align-align-values", action="store_true", default=True,
                        help="score the value stream (default)")
    parser.add_argument("--kv-align-no-align-values", dest="kv_align_align_values",
                        action="store_false", help="do not score the value stream")
    # NVFP4 weight/activation fake-quant.  --wa-state must point at the
    # modelopt_state.pt written by scripts/ptq_nvfp4.py: NVFP4_DEFAULT_CFG
    # calibrates a per-tensor amax, and re-deriving it per job would give each
    # arm a different quantizer.
    parser.add_argument(
        "--wa-scheme",
        dest="wa_scheme",
        default=None,
        choices=("nvfp4", "nvfp4_w4a16"),
        help="NVFP4 weight/activation fake-quant tier for the actor",
    )
    parser.add_argument(
        "--wa-state",
        dest="wa_state",
        default=None,
        help="path to the calibrated modelopt_state.pt for --wa-scheme",
    )
    parser.add_argument(
        "--kv-scheme",
        dest="kv_scheme",
        default=None,
        # Kept in step with qat_kv.turboquant_fake_quant.TQ_FAKE_QUANT_PRESETS
        # plus qat_kv.int4_ptkh_fake_quant.INT4_PTKH_PRESETS;
        # tests/test_turboquant_fake_quant.py drives this parser to enforce it.
        choices=(
            "symmetric",
            "turboquant_3bit_nc",
            "turboquant_k3v4_nc",
            "turboquant_4bit_nc",
            "turboquant_k2v3_nc",
            "turboquant_2bit_nc",
            "int4_per_token_head",
        ),
        help=(
            "training-side KV fake-quant scheme (QAT). A turboquant_* preset "
            "or int4_per_token_head installs the post-RoPE quantizer that "
            "mirrors the vLLM inference kernels and must match "
            "--native-turboquant-kv-cache-dtype. "
            "Omit for no actor-side quantization (the pre-2026-08-30 behaviour)."
        ),
    )
    bias_group = parser.add_mutually_exclusive_group()
    bias_group.add_argument(
        "--preserve-projection-bias",
        dest="preserve_projection_bias",
        action="store_true",
        default=True,
        help=(
            "quantize Qwen2 K/V residuals and add the learned bias at the "
            "projection output dtype (HF post-forward seam is approximate; default)"
        ),
    )
    bias_group.add_argument(
        "--no-preserve-projection-bias",
        dest="preserve_projection_bias",
        action="store_false",
        help="use the unmodified LLM-QAT projection-output quantizer (diagnostic only)",
    )
    parser.add_argument(
        "--exact-bias-free-recompute",
        dest="exact_bias_free_recompute",
        action="store_true",
        help=(
            "opt in to the HF builtin-Linear bias-free K/V recompute before "
            "SymQuantizer (sets QAT_KV_EXACT_BIAS_FREE=1; requires "
            "--preserve-projection-bias and adds projection GEMMs)"
        ),
    )
    parser.add_argument("--sampling-top-k", type=_nonnegative_int, default=0,
                        help="rollout generation filter; 0 disables it for HF (independent of MOPD reward top-k)")
    # Open-MOPD's training recipes use unrestricted nucleus sampling.  Keep
    # this separate from the 0.99 top-p used by the AIME evaluator; the OPD
    # reward's Top-64 candidates must come from the student's untruncated
    # distribution.
    # Was hardcoded at 1e-6.  Over 100 steps at that value grad_norm halved
    # and perplexity fell 8.4%, but the distillation reward did not move,
    # so the rate is worth varying.  The default is unchanged.
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--sampling-top-p", type=_float_01, default=1.0)
    parser.add_argument("--reward-top-k", type=_positive_int, default=64)
    parser.add_argument("--top-k-strategy", default="only_stu",
                        choices=("only_stu", "only_tch", "intersection", "union", "union-intersection"))
    parser.add_argument("--reward-weight-mode", default="student_p",
                        choices=("student_p", "teacher_p", "none"))
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--prompt-length",
        "--max-prompt-length",
        dest="prompt_length",
        type=_positive_int,
        default=None,
        help="override the profile prompt-token budget (default: profile value)",
    )
    parser.add_argument(
        "--response-length",
        "--max-response-length",
        dest="response_length",
        type=_positive_int,
        default=None,
        help="override the profile response-token budget (default: profile value)",
    )
    parser.add_argument("--train-batch-size", type=_positive_int, default=256)
    parser.add_argument("--micro-batch-size-per-gpu", type=_positive_int, default=1)
    # The reward (frozen-teacher) forward is a separate, purely inferential
    # pass, so its micro-batch is a throughput knob and not an optimisation
    # one: sequences are independent and padded positions are masked, so
    # batching more of them per forward is arithmetically identical.  Measured
    # on j15414 it was 27% of step time at the value 1 (reward 97s +
    # compute_rm_score 39s of a 508s step), because 64 sequences on 4 GPUs ran
    # as 16 serial forwards per GPU.  Default stays 1 so existing runs are
    # bit-for-bit unchanged unless a launcher opts in.
    parser.add_argument(
        "--reward-micro-batch-size-per-gpu", type=_positive_int, default=1
    )
    # rollout.log_prob_micro_batch_size_per_gpu was pinned at 1 for the legacy
    # HF rollout, which batches the whole DataProto unless it is set.  We serve
    # with vLLM, which has its own scheduler and ignores the field -- but two
    # *actor* forwards read it: compute_log_prob and compute_distillation_reward
    # (fsdp_workers.compute_distillation_reward sets meta_info["micro_batch_size"]
    # from it).  On j22684 those were 12.9s + 39.8s of a 226s step, all running
    # one sequence at a time.  Like the reward micro-batch this is purely a
    # throughput knob: the forwards are inferential and position-masked, so
    # batching is arithmetically identical.  Default stays 1 so existing runs
    # are unchanged unless a launcher opts in.
    parser.add_argument(
        "--log-prob-micro-batch-size-per-gpu", type=_positive_int, default=1
    )
    parser.add_argument("--ppo-mini-batch-size", dest="ppo_mini_batch_size", type=_positive_int, default=64)
    parser.add_argument(
        "--rollout-gpu-memory-utilization",
        type=_float_01,
        default=0.45,
        help="vLLM GPU memory fraction reserved for rollout; load probes may raise this",
    )
    parser.add_argument("--model")
    parser.add_argument("--teacher")
    parser.add_argument("--train")
    parser.add_argument("--val")
    parser.add_argument("--output")
    parser.add_argument("--python")
    parser.add_argument("--verl-dir")
    parser.add_argument("--py-libs", help="optional compatibility PYTHONPATH (defaults to entropy-floor pylibs)")
    parser.add_argument("--rollout-engine", choices=("vllm", "hf"), default="vllm")
    parser.add_argument(
        "--native-turboquant-kv-cache-dtype",
        choices=(*NATIVE_TURBOQUANT_DTYPES, "none"),
        default="turboquant_3bit_nc",
        help=(
            "use vLLM's native TurboQuant KV cache for rollout generation; "
            "requires --rollout-engine vllm so the old HF QAT/KIVI hooks are "
            "disabled.  Pass 'none' for an unquantized full-precision control "
            "run -- without it there is no way to select the legacy/fp path, "
            "since this option defaults to a quantized preset."
        ),
    )
    kivi_group = parser.add_mutually_exclusive_group()
    kivi_group.add_argument(
        "--kivi-hf-rollout", dest="kivi_hf_rollout", action="store_true",
        help="use the official KIVI HF packed-cache rollout (the KV2 main path)",
    )
    kivi_group.add_argument(
        "--no-kivi-hf-rollout", dest="kivi_hf_rollout", action="store_false",
        help="disable KIVI HF rollout (for the unquantized control)",
    )
    parser.set_defaults(kivi_hf_rollout=False)
    parser.add_argument(
        "--quantize-vllm-rollout",
        dest="quantize_vllm_rollout",
        action="store_true",
        help=(
            "opt in to the Qwen2 fused-QKV vLLM fake/dequant prototype "
            "(sets QAT_KV_VLLM_ENABLE=1; TP=1/external_launcher/enforce_eager only; "
            "runtime validation is still required)"
        ),
    )
    parser.add_argument(
        "--require-quantized-rollout",
        action="store_true",
        help=(
            "strictly reject external vLLM/SGlang rollout when the actor KV hook "
            "cannot be executed; main --run enables this requirement by default "
            "unless --allow-rollout-policy-mismatch is explicit"
        ),
    )
    parser.add_argument(
        "--allow-rollout-policy-mismatch",
        action="store_true",
        help=(
            "allow a real main run to continue with an unquantized external/HF-unverified "
            "rollout (diagnostic actor-only mode; default is fail-closed)"
        ),
    )
    parser.add_argument(
        "--actor-attention",
        choices=("sdpa", "eager", "flash_attention_2"),
        default="sdpa",
        help="HF actor attention backend (default sdpa; eager is unsafe for the 16K profiles)",
    )
    parser.add_argument(
        "--actor-model-dtype",
        choices=("fp32", "bf16", "fp16"),
        default="fp32",
        help=(
            "dtype used to instantiate the trainable HF actor before FSDP; "
            "fp32 preserves the optimizer's master parameters while FSDP "
            "forwards use the explicit bf16 mixed-precision policy"
        ),
    )
    parser.add_argument(
        "--reward-attention",
        choices=("sdpa", "eager", "flash_attention_2"),
        default="sdpa",
        help="frozen-teacher attention backend selected by the external shim (default sdpa)",
    )
    parser.add_argument(
        "--rollout-cuda-graph",
        dest="rollout_cuda_graph",
        action="store_true",
        default=True,
        help=(
            "capture vLLM decode CUDA graphs "
            "(enforce_eager=false, cudagraph_mode=FULL_DECODE_ONLY). "
            "FULL_DECODE_ONLY is required rather than PIECEWISE because "
            "TORCH_COMPILE_DISABLE=1 pins compilation mode to NONE, which "
            "silently downgrades any piecewise cudagraph mode to NONE."
        ),
    )
    parser.add_argument(
        "--rollout-enforce-eager",
        dest="rollout_cuda_graph",
        action="store_false",
        help="fully eager vLLM rollout (the pre-2026-08-30 default)",
    )
    parser.add_argument(
        "--disable-torch-compile",
        dest="disable_torch_compile",
        action="store_true",
        default=True,
        help="disable TorchDynamo/Triton globally (default: required for L20D sm_103a with the pinned Triton wheel)",
    )
    parser.add_argument(
        "--enable-torch-compile",
        dest="disable_torch_compile",
        action="store_false",
        help="opt out of the L20D Triton guard (only after validating a compatible ptxas)",
    )
    parser.add_argument(
        "--strict-fp32-arithmetic",
        action="store_true",
        help=(
            "fail preflight if the actor is not initialized with the FP32-master/FSDP "
            "mixed-precision contract; runtime optimizer-state and teacher arithmetic "
            "remain explicitly unverified"
        ),
    )
    parser.add_argument(
        "--one-epoch",
        action="store_true",
        help=(
            "require updates to fit in one drop-last dataloader epoch (fails when "
            "updates exceed rows//train-batch-size)"
        ),
    )
    parser.add_argument(
        "--save-freq",
        type=_positive_int,
        default=20,
        help="trainer checkpoint frequency; 20 covers the requested every-20-step evaluation schedule",
    )
    parser.add_argument(
        "--eval-checkpoints",
        type=_checkpoint_list,
        default=MOPD_EVAL_CHECKPOINTS_DEFAULT,
        help="comma-separated optimizer steps for the profile's external evaluator (metadata/schedule only)",
    )
    parser.add_argument(
        "--eval-gate-summary",
        help=(
            "path to scripts/eval_kv_gate.py gate_selection.json; when supplied, "
            "the profile/model/selected KV bits are checked before launch"
        ),
    )
    parser.add_argument(
        "--require-eval-gate",
        action="store_true",
        help="refuse --run unless a complete FP32/KV2(/KV4) selection summary is supplied and matches this profile",
    )
    parser.add_argument(
        "--allow-missing-eval-gate",
        action="store_true",
        help=(
            "diagnostic escape hatch for a real main run with no selection summary; "
            "a supplied stale/mismatched summary is still rejected"
        ),
    )
    parser.add_argument("--resume", action="store_true", help="allow a non-empty output and use trainer resume_mode=auto")
    parser.add_argument("--resume-path")
    parser.add_argument("--project")
    parser.add_argument("--experiment")
    parser.add_argument("--swanlab-mode", choices=("cloud", "offline", "disabled"), default="cloud")
    parser.add_argument("--no-trust-remote-code", action="store_true")
    parser.add_argument("--override", action="append", help="extra Hydra override (repeatable)")
    parser.add_argument("--materialize-jsonl", action="store_true", help="stream prepared JSONL to the profile's Parquet path")
    parser.add_argument("--overwrite-data", action="store_true", help="allow replacing an explicitly named Parquet companion")
    parser.add_argument("--skip-dependency-check", action="store_true")
    parser.add_argument("--skip-config-check", action="store_true", help="skip Hydra --cfg job composition probe")
    parser.add_argument("--run", action="store_true", help="execute after strict preflight; default is dry-run")
    parser.add_argument("--dry-run", action="store_true", help="explicitly select dry-run (the default)")
    parser.add_argument("--json", action="store_true", help="emit a machine-readable summary after human output")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    # Preserve the caller's explicit mode.  In particular, ``disabled`` must
    # not silently become offline (which would still import and initialize
    # SwanLab in the trainer).
    swanlab_mode = args.swanlab_mode
    args.swanlab_mode = swanlab_mode
    resolved = _resolve(args)

    runtime_env = dict(os.environ)
    runtime_env["SWANLAB_MODE"] = swanlab_mode
    # Keep an explicit unavailable state until credential resolution succeeds;
    # otherwise an exception while reading .autokey could be misreported as
    # ``not-required`` in the redacted command/manifest.
    swanlab_status = "unavailable"
    credential_error: str | None = None
    try:
        runtime_env, swanlab_status = _load_swanlab_key(runtime_env)
    except RuntimeError as exc:
        credential_error = str(exc)

    # Remember whether this is a genuinely fresh artifact directory.  The
    # effective-settings manifest is written only for a fresh real run; a
    # resumed run must preserve the original contract instead of overwriting
    # its provenance.
    output_was_nonempty = False
    if resolved.output.exists():
        if resolved.output.is_dir():
            try:
                output_was_nonempty = any(resolved.output.iterdir())
            except OSError:
                output_was_nonempty = True
        else:
            output_was_nonempty = True

    # Materialization is the only write performed before --run, and it targets
    # one explicit companion path.  A real run remains strict after it.
    strict = bool(args.run)
    warnings, steps_per_epoch = preflight(
        resolved,
        strict=strict,
        skip_deps=args.skip_dependency_check,
        materialize=args.materialize_jsonl,
        overwrite_data=args.overwrite_data,
    )
    if credential_error is not None:
        _status(False, "SwanLab cloud credential", credential_error, strict=strict, warnings=warnings)
    elif resolved.swanlab_mode == "cloud":
        print(f"[OK  ] SwanLab cloud credential: {swanlab_status} (redacted)")
    command = _hydra_command(resolved, steps_per_epoch)
    safe_command = [_safe_text(str(item)) for item in command]
    effective_manifest_path = (resolved.output / "effective_settings.json").resolve(strict=False)
    if not args.skip_config_check and not args.skip_dependency_check:
        _hydra_config_probe(command, resolved, strict=strict, warnings=warnings)
    _print_command(command, swanlab_status=swanlab_status)
    print(
        "[mopd] objective: %s (single frozen teacher; metadata-only legacy aliases: %s)"
        % (MOPD_OBJECTIVE_NAME, ", ".join(MOPD_OBJECTIVE_LEGACY_ALIASES))
    )
    print("[mopd] paper/implementation path: official Open-MOPD compute_distillation_reward + token_reward_direct PPO")
    print(
        "[mopd] knobs: reward top-k=%d from student untruncated distribution "
        "(only_stu candidate set; reward top-p intersection metadata=0.99 is ignored by only_stu, not an extra nucleus filter); "
        "rollout sampling top-k=%d, top-p=%.4g" % (
            resolved.reward_top_k,
            resolved.sampling_top_k,
            resolved.sampling_top_p,
        )
    )
    print(
        f"[mopd] HF attention: actor={resolved.actor_attention}; "
        f"frozen teacher={resolved.reward_attention} (SDPA is the 16K-safe default)"
    )
    print(
        f"[mopd] actor dtype contract: model_dtype={resolved.actor_model_dtype}; "
        f"FSDP mixed param={ACTOR_FSDP_PARAM_DTYPE}, reduce={ACTOR_FSDP_REDUCE_DTYPE}, "
        f"buffer={ACTOR_FSDP_BUFFER_DTYPE}; optimizer_state_expectation="
        f"{'fp32_master_params' if resolved.actor_model_dtype == 'fp32' else 'unverified_non_fp32_model_init'}; "
        "FP32 params are ~2x BF16 before sharding; optimizer state may add more peak memory (profile gate)"
    )
    print(
        f"[mopd] KV bias handling: preserve_projection_bias={resolved.preserve_projection_bias} "
        "(Qwen2/DeepSeek key-bias adaptation)"
    )
    print(
        "[mopd] evaluation schedule: profile-specific external evaluator at updates "
        f"{list(resolved.eval_checkpoints)}; trainer.test_freq=-1 (external only)"
    )
    print(
        "[mopd] frozen-model eval gate: "
        f"summary={resolved.eval_gate_summary if resolved.eval_gate_summary is not None else 'unset'}; "
        f"required={resolved.eval_gate_required_for_run}; "
        f"explicit_required={resolved.require_eval_gate}; "
        f"allow_missing={resolved.allow_missing_eval_gate}"
    )
    print(
        f"[mopd] KV policy: actor_hook_expected={resolved.actor_kv_hook_expected}; "
        f"rollout_kv_quantized_expected={resolved.rollout_kv_quantized}; "
        f"rollout_policy_mismatch_expected={resolved.rollout_policy_mismatch}; "
        f"allow_rollout_policy_mismatch={resolved.allow_rollout_policy_mismatch}; "
        f"rollout_quantized_required_for_run={resolved.rollout_quantized_required_for_run}; "
        f"vllm_patch_requested={resolved.quantize_vllm_rollout} "
        f"(QAT_KV_VLLM_ENABLE={'1' if resolved.quantize_vllm_rollout else '0'}); "
        f"status={resolved.rollout_kv_status}; runtime_verified="
        f"{'pending (engine proof written to ' + str(resolved.output / ROLLOUT_KV_VERIFICATION_FILENAME) + ')' if resolved.native_turboquant_rollout else 'false'}"
        "; exact_opd=false"
    )
    print(
        f"[mopd] HF bias preservation mode: "
        f"{('exact_bias_free_recompute' if resolved.exact_bias_free_recompute else 'post_forward_approximate' if resolved.preserve_projection_bias else 'disabled')}"
    )
    if warnings:
        print("[mopd] warnings:")
        for warning in dict.fromkeys(warnings):
            print(f"  - {warning}")

    summary = {
        "profile": resolved.profile.name,
        "stage": resolved.stage,
        "updates": resolved.updates,
        "model": str(resolved.model),
        "teacher": str(resolved.teacher),
        "train": str(resolved.train),
        "output": str(resolved.output),
        "prompt_length": resolved.profile.prompt_length,
        "response_length": resolved.profile.response_length,
        "context_length": resolved.profile.prompt_length + resolved.profile.response_length,
        "rollout_max_num_batched_tokens": min(32768, resolved.profile.prompt_length + resolved.profile.response_length),
        "rollout_gpu_memory_utilization": resolved.rollout_gpu_memory_utilization,
        "rollout_cuda_graph": resolved.rollout_cuda_graph,
        "rollout_cudagraph_mode": ("FULL_DECODE_ONLY" if resolved.rollout_cuda_graph else "NONE"),
        "rollout_enable_chunked_prefill": resolved.profile.prompt_length + resolved.profile.response_length > 32768,
        "actor_gradient_checkpointing": (
            resolved.profile.prompt_length + resolved.profile.response_length >= 32768
            or resolved.profile.response_length >= 16384
        ),
        **(
            {}
            if resolved.native_turboquant_rollout
            else {"legacy_kv_bits": resolved.legacy_kv_bits}
        ),
        "kv_scheme": resolved.kv_scheme,
        "block_align_coef": resolved.block_align_coef,
        "block_align_layer_every": resolved.block_align_layer_every,
        "block_align_keep_fraction": resolved.block_align_keep_fraction,
        "block_align_metric": resolved.block_align_metric,
        "block_align_last_only": resolved.block_align_last_only,
        "actor_fake_quant_enabled": bool(
            resolved.kv_scheme and resolved.kv_scheme != "symmetric"
        ),
        "native_turboquant_rollout": resolved.native_turboquant_rollout,
        "native_turboquant_kv_cache_dtype": resolved.native_turboquant_kv_cache_dtype,
        "rollout_student_logprob_source": (
            "vllm_native_turboquant" if resolved.native_turboquant_rollout else "hf_or_rollout"
        ),
        "teacher_kv_cache_dtype": "bf16",
        "rollout_kv_runtime_verification": (
            "required_engine_structural_proof"
            if resolved.native_turboquant_rollout
            else "plugin_or_registry"
        ),
        "preserve_projection_bias": resolved.preserve_projection_bias,
        "exact_bias_free_recompute": resolved.exact_bias_free_recompute,
        "hf_bias_preservation_mode": (
            "exact_bias_free_recompute"
            if resolved.exact_bias_free_recompute
            else "post_forward_approximate"
            if resolved.preserve_projection_bias
            else "disabled"
        ),
        "reward_top_k": resolved.reward_top_k,
        "reward_top_k_source": MOPD_REWARD_TOP_K_SOURCE,
        "reward_top_k_source_legacy": MOPD_REWARD_TOP_K_SOURCE_LEGACY,
        "reward_top_k_source_metadata_only": True,
        "reward_candidate_truncation": "none",
        "reward_top_p_intersec_p": 0.99,
        "reward_top_p_scope": "only_stu_no_extra_filter",
        "reward_top_p_applies": False,
        "sampling_top_k": resolved.sampling_top_k,
        "sampling_top_p": resolved.sampling_top_p,
        "objective": MOPD_OBJECTIVE_NAME,
        "teacher_count": 1,
        "teacher_routing": "single_frozen_teacher",
        "objective_legacy": MOPD_OBJECTIVE_LEGACY_NAME,
        "objective_legacy_metadata_only": True,
        "objective_legacy_aliases": list(MOPD_OBJECTIVE_LEGACY_ALIASES),
        "exact_opd": False,
        "actor_kv_hook_expected": resolved.actor_kv_hook_expected,
        "quantize_vllm_rollout": resolved.quantize_vllm_rollout,
        "vllm_kv_patch_requested": resolved.quantize_vllm_rollout,
        "vllm_kv_patch_available": resolved.vllm_kv_patch_available,
        "require_quantized_rollout": resolved.rollout_quantized_required_for_run,
        "require_quantized_rollout_explicit": resolved.require_quantized_rollout,
        "rollout_kv_status": resolved.rollout_kv_status,
        "rollout_kv_quantized_expected": resolved.rollout_kv_quantized,
        "rollout_kv_quantized": resolved.rollout_kv_quantized,
        "rollout_kv_quantized_runtime": None,
        "rollout_quantized_required_for_run": resolved.rollout_quantized_required_for_run,
        "rollout_policy_mismatch": resolved.rollout_policy_mismatch,
        "rollout_policy_mismatch_expected": resolved.rollout_policy_mismatch,
        "allow_rollout_policy_mismatch": resolved.allow_rollout_policy_mismatch,
        "rollout_policy_mismatch_verified": False,
        "rollout_kv_verified": None if resolved.native_turboquant_rollout else False,
        "rollout_kv_verification_record": (
            str(resolved.output / ROLLOUT_KV_VERIFICATION_FILENAME)
            if resolved.native_turboquant_rollout
            else None
        ),
        "hf_rollout_registry_available": resolved.hf_rollout_registry_available,
        "global_batch": resolved.train_batch_size,
        "world_size": resolved.world_size,
        "gradient_accumulation_steps": resolved.gradient_accumulation_steps,
        "ray_num_cpus": resolved.ray_num_cpus,
        "ray_num_cpus_floor": max(8, resolved.nodes * resolved.gpus + 1),
        "disable_torch_compile": resolved.disable_torch_compile,
        "actor_model_dtype": resolved.actor_model_dtype,
        "actor_fsdp_param_dtype": ACTOR_FSDP_PARAM_DTYPE,
        "actor_fsdp_reduce_dtype": ACTOR_FSDP_REDUCE_DTYPE,
        "actor_fsdp_buffer_dtype": ACTOR_FSDP_BUFFER_DTYPE,
        "optimizer_state_expectation": (
            "fp32_master_params"
            if resolved.actor_model_dtype == "fp32"
            else "unverified_non_fp32_model_init"
        ),
        "actor_fp32_storage_memory_warning": resolved.actor_model_dtype == "fp32",
        "actor_attention": resolved.actor_attention,
        "reward_attention": resolved.reward_attention,
        "save_freq": resolved.save_freq,
        "eval_checkpoints": list(resolved.eval_checkpoints),
        "eval_gate_summary": (
            str(resolved.eval_gate_summary.resolve(strict=False))
            if resolved.eval_gate_summary is not None
            else None
        ),
        "require_eval_gate": resolved.eval_gate_required_for_run,
        "require_eval_gate_explicit": resolved.require_eval_gate,
        "allow_missing_eval_gate": resolved.allow_missing_eval_gate,
        "one_epoch": resolved.one_epoch,
        "strict_fp32_arithmetic": resolved.strict_fp32_arithmetic,
        "fp32_contract_satisfied": resolved.actor_model_dtype == "fp32",
        "fp32_arithmetic_verified": False,
        "steps_per_epoch": steps_per_epoch,
        "planned_trainer_epochs": (
            1
            if resolved.one_epoch
            else (
                max(1, math.ceil(resolved.updates / steps_per_epoch))
                if steps_per_epoch
                else max(1, resolved.updates)
            )
        ),
        "planned_dataset_passes": (
            resolved.updates / steps_per_epoch if steps_per_epoch else None
        ),
        "planned_pass_definition": "updates / floor(rows / global_batch)",
        "planned_consumed_examples": resolved.updates * resolved.train_batch_size,
        "ray_temp_dir": str(resolved.ray_temp_dir),
        "ray_spill_dir": str(resolved.ray_spill_dir),
        "ray_spill_dir_wired": False,
        "ray_spill_policy": "ray_default_under_temp_dir",
        "effective_manifest": str(effective_manifest_path),
        "effective_manifest_written": False,
        "run_mode": "run" if args.run else "dry-run",
        "manifest_policy": "fresh_run_only",
        "runtime_env_contract": {
            **(
                {}
                if resolved.native_turboquant_rollout
                else {"QAT_KV_BITS": str(resolved.legacy_kv_bits)}
            ),
            "QAT_KV_PRESERVE_BIAS": "1" if (resolved.preserve_projection_bias and not resolved.native_turboquant_rollout) else "0",
            "QAT_KV_EXACT_BIAS_FREE": (
                "1" if resolved.exact_bias_free_recompute else "0"
            ),
            "QAT_KV_KIVI_HF_ENABLE": (
                "1" if (resolved.kivi_hf_rollout and not resolved.native_turboquant_rollout) else "0"
            ),
            "QAT_KV_KIVI_GROUP_SIZE": "32",
            "QAT_KV_KIVI_RESIDUAL_LENGTH": "32",
            "QAT_KV_VLLM_ENABLE": "1" if resolved.quantize_vllm_rollout else "0",
            "MOPD_NATIVE_TURBOQUANT_ROLLOUT": "1" if resolved.native_turboquant_rollout else "0",
            "MOPD_NATIVE_TURBOQUANT_KV_CACHE_DTYPE": resolved.native_turboquant_kv_cache_dtype or "",
            "ROLLOUT_STUDENT_LOGPROB_SOURCE": (
                "vllm_native_turboquant" if resolved.native_turboquant_rollout else "hf_or_rollout"
            ),
            "QAT_KV_VLLM_EXTERNAL_LAUNCHER": "1" if resolved.quantize_vllm_rollout else "0",
            "QAT_KV_REQUIRE_QUANTIZED_ROLLOUT": (
                "1" if resolved.rollout_quantized_required_for_run else "0"
            ),
            "MOPD_ROLLOUT_KV_STATUS": resolved.rollout_kv_status,
            "MOPD_ROLLOUT_POLICY_MISMATCH": "1" if resolved.rollout_policy_mismatch else "0",
            "MOPD_ALLOW_ROLLOUT_POLICY_MISMATCH": (
                "1" if resolved.allow_rollout_policy_mismatch else "0"
            ),
            "TORCH_COMPILE_DISABLE": "1" if resolved.disable_torch_compile else "0",
            "QAT_KV_REWARD_ATTN": resolved.reward_attention,
            "MOPD_EVAL_GATE_SUMMARY": (
                str(resolved.eval_gate_summary)
                if resolved.eval_gate_summary is not None
                else ""
            ),
            "MOPD_REQUIRE_EVAL_GATE": (
                "1" if resolved.eval_gate_required_for_run else "0"
            ),
            "MOPD_ALLOW_MISSING_EVAL_GATE": (
                "1" if resolved.allow_missing_eval_gate else "0"
            ),
            "SWANLAB_MODE": resolved.swanlab_mode,
            "SWANLAB_API_KEY": "child_environment_only",
        },
        "warnings": list(dict.fromkeys(warnings)),
        "swanlab_credential": swanlab_status,
        "command": safe_command,
    }
    # Dry-runs can emit the machine-readable plan immediately.  For a real
    # launch we defer JSON until after the strict gate and (on a fresh output)
    # the manifest write, so ``effective_manifest_written`` reflects the file
    # that is actually about to accompany the trainer process.
    if args.json and not args.run:
        print(json.dumps(summary, ensure_ascii=False, indent=2))

    if args.run:
        # Strict mode means any failed check is represented in warnings.  Do
        # not create output or spawn Ray if a required check failed.
        advisory_prefixes = (
            # Empirically validated by job 22447; see the warning text.  Left as
            # a warning rather than removed so every compiled run still says so
            # in its log.
            "torch.compile enabled (advisory)",
            "vLLM rollout",
            "HF rollout",
            "Ray config default num_cpus=null; launcher forces",
            "evaluation gate missing but explicitly allowed",
            "evaluation gate optional:",
        )
        hard_failures = [w for w in warnings if not w.startswith(advisory_prefixes)]
        if hard_failures:
            print("[mopd] refusing --run because strict preflight failed; fix the items above", file=sys.stderr)
            if args.json:
                print(json.dumps(summary, ensure_ascii=False, indent=2))
            return 2
        try:
            resolved.output.mkdir(parents=True, exist_ok=True)
            # Keep explicit shared Ray paths available on every node in
            # real-run mode.  Ray 2.58 currently uses its default spiller
            # beneath ``_temp_dir`` (the separate spill path is provenance /
            # future-version metadata); dry-runs stay read-only.
            resolved.ray_temp_dir.mkdir(parents=True, exist_ok=True)
            resolved.ray_spill_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            print(f"[mopd] refusing --run: cannot create shared Ray directories: {exc}", file=sys.stderr)
            return 2
        if not output_was_nonempty:
            try:
                # Mark the in-memory summary before writing so the JSON printed
                # after launch and the on-disk manifest agree.  The writer
                # still sets this field defensively for callers that invoke it
                # directly.
                summary["effective_manifest_written"] = True
                _write_effective_manifest(effective_manifest_path, summary)
            except Exception as exc:
                summary["effective_manifest_written"] = False
                print(
                    "[mopd] refusing --run: cannot write non-secret effective-settings manifest: "
                    f"{type(exc).__name__}: {exc}",
                    file=sys.stderr,
                )
                return 2
            print(f"[mopd] effective settings manifest: {effective_manifest_path}")
        else:
            print(
                f"[mopd] resume: preserving existing effective-settings provenance "
                f"(not overwriting {effective_manifest_path})"
            )
        if args.json:
            print(json.dumps(summary, ensure_ascii=False, indent=2))
        print("[mopd] launching official trainer; Ctrl-C leaves checkpoints intact")
        # ``runtime_env`` is the only object that contains the credential.  It
        # is passed directly to the child and never serialized/printed.
        env = dict(runtime_env)
        if os.environ.get("MOPD_RAY_PATH"):
            env["PATH"] = os.environ["MOPD_RAY_PATH"]
        python_include = os.environ.get(
            "MOPD_PYTHON_INCLUDE",
            "/historical/shared/qat/tmp/python-devel/root/usr/include/python3.12",
        )
        include_path = os.pathsep.join(
            item for item in (python_include, os.environ.get("CPATH", "")) if item
        )
        env["CPATH"] = include_path
        env["C_INCLUDE_PATH"] = include_path
        env["CPLUS_INCLUDE_PATH"] = include_path
        env["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
        env.update(
            {
                "PYTHONPATH": _pythonpath(resolved),
                "PYTHONUNBUFFERED": "1",
                "TOKENIZERS_PARALLELISM": "false",
                "VERL_USE_EXTERNAL_MODULES": "qat_kv.verl_plugin" if _needs_legacy_kv_plugin(resolved) else "",
                **(
                    {}
                    if resolved.native_turboquant_rollout
                    else {
                        "QAT_KV_BITS": str(resolved.legacy_kv_bits),
                        "QAT_KV_REQUIRE_BITS": "1",
                    }
                ),
                "QAT_KV_LAYERWISE": "0",
                "QAT_KV_PRESERVE_BIAS": "1" if (resolved.preserve_projection_bias and not resolved.native_turboquant_rollout) else "0",
                "QAT_KV_EXACT_BIAS_FREE": (
                    "1" if (resolved.exact_bias_free_recompute and not resolved.native_turboquant_rollout) else "0"
                ),
                "QAT_KV_KIVI_HF_ENABLE": "1" if (resolved.kivi_hf_rollout and not resolved.native_turboquant_rollout) else "0",
                "QAT_KV_KIVI_GROUP_SIZE": "32",
                "QAT_KV_KIVI_RESIDUAL_LENGTH": "32",
                "MOPD_OBJECTIVE": MOPD_OBJECTIVE_NAME,
                "MOPD_ACTOR_KV_HOOK_EXPECTED": "1" if resolved.actor_kv_hook_expected else "0",
                "MOPD_ACTOR_MODEL_DTYPE": resolved.actor_model_dtype,
                "MOPD_OPTIMIZER_STATE_EXPECTATION": (
                    "fp32_master_params"
                    if resolved.actor_model_dtype == "fp32"
                    else "unverified_non_fp32_model_init"
                ),
                "MOPD_ROLLOUT_KV_QUANTIZED": "1" if resolved.rollout_kv_quantized else "0",
                "MOPD_ROLLOUT_POLICY_MISMATCH": "1" if resolved.rollout_policy_mismatch else "0",
                "MOPD_ALLOW_ROLLOUT_POLICY_MISMATCH": (
                    "1" if resolved.allow_rollout_policy_mismatch else "0"
                ),
                "MOPD_ROLLOUT_KV_STATUS": resolved.rollout_kv_status,
                "MOPD_VLLM_PATCH_REQUESTED": "1" if resolved.quantize_vllm_rollout else "0",
                "MOPD_NATIVE_TURBOQUANT_ROLLOUT": "1" if resolved.native_turboquant_rollout else "0",
                "MOPD_NATIVE_TURBOQUANT_KV_CACHE_DTYPE": resolved.native_turboquant_kv_cache_dtype or "",
                **(
                    {
                        "MOPD_ROLLOUT_KV_VERIFICATION_RECORD": str(
                            resolved.output / ROLLOUT_KV_VERIFICATION_FILENAME
                        )
                    }
                    if resolved.native_turboquant_rollout
                    else {}
                ),
                "ROLLOUT_STUDENT_LOGPROB_SOURCE": (
                    "vllm_native_turboquant" if resolved.native_turboquant_rollout else "hf_or_rollout"
                ),
                "QAT_KV_VLLM_ENABLE": "1" if (resolved.quantize_vllm_rollout and not resolved.native_turboquant_rollout) else "0",
                "QAT_KV_VLLM_EXTERNAL_LAUNCHER": "1" if (resolved.quantize_vllm_rollout and not resolved.native_turboquant_rollout) else "0",
                "QAT_KV_REQUIRE_QUANTIZED_ROLLOUT": (
                    "1" if resolved.rollout_quantized_required_for_run else "0"
                ),
                "TORCH_COMPILE_DISABLE": "1" if resolved.disable_torch_compile else "0",
                "TORCHDYNAMO_DISABLE": "1" if resolved.disable_torch_compile else "0",
                "QAT_KV_REWARD_ATTN": resolved.reward_attention,
                # Probe-only objective switches are forwarded to Ray workers;
                # ordinary runs retain the default (both, no gradient probe).
                "QAT_PROBE_OBJECTIVE": os.environ.get("QAT_PROBE_OBJECTIVE", "both"),
                "QAT_GRAD_PROBE": os.environ.get("QAT_GRAD_PROBE", "0"),
            "QAT_FSDP_SHARDING_STRATEGY": os.environ.get("QAT_FSDP_SHARDING_STRATEGY", ""),
                "QAT_OPD_POSITION_RAMP": os.environ.get("QAT_OPD_POSITION_RAMP", ""),
                "QAT_OPD_POSITION_RAMP_NORMALIZE": os.environ.get("QAT_OPD_POSITION_RAMP_NORMALIZE", "0"),
                "QAT_BLOCK_ALIGN_STATIC_REF": os.environ.get("QAT_BLOCK_ALIGN_STATIC_REF", "1"),
                "QAT_SKIP_TEACHER_ENTROPY": os.environ.get("QAT_SKIP_TEACHER_ENTROPY", "0"),
                "MOPD_EVAL_GATE_SUMMARY": (
                    str(resolved.eval_gate_summary)
                    if resolved.eval_gate_summary is not None
                    else ""
                ),
                "MOPD_REQUIRE_EVAL_GATE": (
                    "1" if resolved.eval_gate_required_for_run else "0"
                ),
                "MOPD_ALLOW_MISSING_EVAL_GATE": (
                    "1" if resolved.allow_missing_eval_gate else "0"
                ),
                "SWANLAB_MODE": resolved.swanlab_mode,
                "SWANLAB_LOG_DIR": str(resolved.output / "swanlog"),
            }
        )
        return subprocess.call(command, env=env)
    print("[mopd] dry-run only; pass --run after reviewing the command")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
