# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
The main entry point to run the PPO algorithm
"""

import contextlib
import datetime
import hashlib
import json
import logging
import math
import os
import warnings
from contextlib import contextmanager
from dataclasses import asdict
from typing import Any, Optional

import numpy as np
import psutil
import torch
import torch.distributed
import torch.distributed as dist
import torch.nn.functional as F
from codetiming import Timer
from omegaconf import DictConfig, OmegaConf, open_dict
from peft import AdaLoraConfig, LoraConfig, TaskType, get_peft_model
from safetensors.torch import save_file
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.api import FullStateDictConfig, ShardedStateDictConfig, StateDictType

try:
    # for torch 2.5+
    from torch.distributed.tensor import DTensor
except ImportError:
    from torch.distributed._tensor import DTensor

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.models.transformers.monkey_patch import apply_monkey_patch
from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import Dispatch, make_nd_compute_dataproto_dispatch_fn, register
from verl.utils import hf_processor, hf_tokenizer
from verl.utils.activation_offload import enable_activation_offloading
from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.device import (
    get_device_id,
    get_device_name,
    get_nccl_backend,
    get_torch_device,
    set_expandable_segments,
)
from verl.utils.flops_counter import FlopsCounter
from verl.utils.fs import copy_to_local
from verl.utils.fsdp_utils import (
    CPUOffloadPolicy,
    MixedPrecisionPolicy,
    apply_fsdp2,
    collect_lora_params,
    collect_merged_peft_params,
    fsdp2_load_full_state_dict,
    fsdp_version,
    get_fsdp_wrap_policy,
    get_init_weight_context_manager,
    get_shard_placement_fn,
    init_fn,
    layered_summon_lora_params,
    load_fsdp_model_to_gpu,
    load_fsdp_optimizer,
    offload_fsdp_model_to_cpu,
    offload_fsdp_optimizer,
    replace_lora_wrapper,
)
from verl.utils.import_utils import import_external_libs
from verl.utils.memory_utils import aggressive_empty_cache
from verl.utils.model import compute_position_id_with_mask, convert_weight_keys
from verl.utils.profiler import DistProfiler, DistProfilerExtension, ProfilerConfig, log_gpu_memory_usage, simple_timer
from verl.utils.profiler.performance import reduce_timing, topk_reduce_ratio_min_max
from verl.utils.py_functional import convert_to_regular_types
from verl.utils.ray_utils import get_event_loop
from verl.workers.config import FSDPCriticConfig, FSDPEngineConfig, HFModelConfig, RolloutConfig
from verl.workers.config.model import _resolve_svd_init_aux_path
from verl.workers.config.optimizer import build_optimizer
from verl.workers.rollout import get_rollout_class
from verl.workers.sharding_manager.fsdp_ulysses import FSDPUlyssesShardingManager

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

device_name = get_device_name()


def _debug_base_kl_payload(old_log_prob: torch.Tensor, base_ref_log_prob: torch.Tensor, response_mask: torch.Tensor) -> dict:
    diff = torch.clamp(old_log_prob - base_ref_log_prob, min=-20.0, max=20.0)
    seq_den = response_mask.sum(dim=-1).clamp_min(1)
    seq_base_kl = (-diff * response_mask).sum(dim=-1) / seq_den
    return {
        "old_mean": float(old_log_prob.mean().item()),
        "base_ref_mean": float(base_ref_log_prob.mean().item()),
        "diff_mean": float(diff.mean().item()),
        "diff_abs_mean": float(diff.abs().mean().item()),
        "masked_base_kl": float(verl_F.masked_mean(-diff, response_mask).item()),
        "response_mask_sum": float(response_mask.sum().item()),
        "seq_base_kl_head": seq_base_kl[:8].detach().cpu().tolist(),
    }


def _write_base_kl_debug(stage: str, payload: dict) -> str:
    out_dir = os.getenv(
        "VERL_BASE_KL_DEBUG_DIR",
        "./outputs/base_kl_debug",
    )
    os.makedirs(out_dir, exist_ok=True)
    exp_name = str(payload.get("experiment_name", "unknown_exp"))
    path = os.path.join(out_dir, f"{exp_name}.{stage}.rank{payload.get('rank', 'x')}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, sort_keys=True, indent=2)
    return path


def _write_rollout_policy_debug(stage: str, payload: dict) -> str:
    out_dir = os.getenv(
        "VERL_ROLLOUT_POLICY_DEBUG_DIR",
        "./outputs/rollout_policy_debug",
    )
    os.makedirs(out_dir, exist_ok=True)
    exp_name = str(payload.get("experiment_name", "unknown_exp"))
    path_parts = [exp_name, stage, f"rank{payload.get('rank', 'x')}"]
    if "sample_index" in payload:
        path_parts.append(f"sample{payload['sample_index']}")
    if "request_id" in payload:
        path_parts.append(f"req{payload['request_id']}")
    if "prompt_hash" in payload:
        path_parts.append(f"prompt{payload['prompt_hash']}")
    path = os.path.join(out_dir, ".".join(path_parts) + ".json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, sort_keys=True, indent=2)
    return path


def _prompt_hash(token_ids: list[int]) -> str:
    joined = ",".join(map(str, token_ids)).encode("utf-8")
    return hashlib.sha1(joined).hexdigest()[:12]


def _message_preview(raw_prompt: Any, max_items: int = 2) -> list[dict[str, Any]]:
    if isinstance(raw_prompt, np.ndarray):
        raw_prompt = raw_prompt.tolist()
    if not isinstance(raw_prompt, list):
        return []

    preview = []
    for message in raw_prompt[:max_items]:
        if not isinstance(message, dict):
            preview.append({"value": str(message)[:200]})
            continue
        content = message.get("content")
        content_preview = content[:200] if isinstance(content, str) else str(content)[:200]
        preview.append({"role": message.get("role"), "content_preview": content_preview})
    return preview


def _actor_adapter_state_payload(peft_model, extra: dict | None = None) -> dict:
    payload = {
        "has_peft_config": hasattr(peft_model, "peft_config"),
        "active_adapters": list(getattr(peft_model, "active_adapters", [])) if peft_model is not None else [],
        "disable_adapter_layers": bool(getattr(getattr(peft_model, "base_model", None), "_disable_adapter", False)),
    }
    if extra:
        payload.update(extra)
    return payload


def create_device_mesh(world_size, fsdp_size):
    if fsdp_size < 0 or fsdp_size >= world_size:
        device_mesh = init_device_mesh(device_name, mesh_shape=(world_size,), mesh_dim_names=["fsdp"])
    else:
        device_mesh = init_device_mesh(
            device_name, mesh_shape=(world_size // fsdp_size, fsdp_size), mesh_dim_names=["ddp", "fsdp"]
        )
    return device_mesh


def get_sharding_strategy(device_mesh):
    from torch.distributed.fsdp import ShardingStrategy

    if device_mesh.ndim == 1:
        sharding_strategy = ShardingStrategy.FULL_SHARD
    elif device_mesh.ndim == 2:
        sharding_strategy = ShardingStrategy.HYBRID_SHARD
    else:
        raise NotImplementedError(f"Get device mesh ndim={device_mesh.ndim}, but only support 1 or 2")
    return sharding_strategy


def get_vl_model_vision_tower(vl_model_instance):
    """
    Util to extract Vision Tower from a VL model instance
    """
    if hasattr(vl_model_instance, "model") and hasattr(vl_model_instance.model, "visual"):
        # transformers >= 4.52.0
        return vl_model_instance.model.visual
    elif hasattr(vl_model_instance, "visual"):
        # transformers < 4.52.0
        return vl_model_instance.visual
    return None


def _haar_wavelet_rows(x: torch.Tensor) -> torch.Tensor:
    """Apply in-place 1D Haar wavelet transform row-wise.

    Args:
        x: Tensor with shape [num_rows, num_cols].
    Returns:
        Tensor with the same shape, transformed row-wise.
    """
    if x.ndim != 2:
        raise ValueError(f"Expected 2D tensor for Haar transform, got shape={tuple(x.shape)}")

    num_rows, num_cols = x.shape
    if num_cols <= 1:
        return x

    # Haar decomposition is simplest on power-of-two length.
    target_cols = 1 << (num_cols - 1).bit_length()
    if target_cols != num_cols:
        x = F.pad(x, (0, target_cols - num_cols))

    width = target_cols
    inv_sqrt2 = 1.0 / math.sqrt(2.0)
    while width > 1:
        even = x[:, :width:2]
        odd = x[:, 1:width:2]
        avg = (even + odd) * inv_sqrt2
        diff = (even - odd) * inv_sqrt2
        x[:, : width // 2] = avg
        x[:, width // 2 : width] = diff
        width //= 2

    return x[:, :num_cols]


def _dct_lora_a(rows: int, cols: int, device: torch.device) -> torch.Tensor:
    if rows > cols:
        raise ValueError(f"DCT-LoRA requires rank <= in_features, got rank={rows}, in_features={cols}")
    row_idx = torch.arange(rows, device=device, dtype=torch.float32).unsqueeze(1)
    col_idx = torch.arange(cols, device=device, dtype=torch.float32).unsqueeze(0)
    basis = torch.cos(math.pi * (2.0 * col_idx + 1.0) * row_idx / (2.0 * cols))
    scale = torch.full((rows, 1), math.sqrt(2.0 / cols), device=device, dtype=torch.float32)
    if rows > 0:
        scale[0, 0] = math.sqrt(1.0 / cols)
    return scale * basis


def _init_lora_a_params(module: torch.nn.Module, init_method: str = "default", wavelet: str = "haar") -> int:
    """Initialize LoRA-A weights with a configurable method.

    Returns:
        Number of LoRA-A matrices initialized.
    """
    method = str(init_method).lower()
    if method in {"default", "none"}:
        return 0
    if method not in {"wavelet_orth", "dct", "dct_lora"}:
        raise ValueError(f"Unsupported lora_A_init_method={init_method!r}")

    if method == "wavelet_orth":
        wavelet_name = str(wavelet).lower()
        if wavelet_name != "haar":
            raise ValueError(f"Unsupported lora_A_wavelet={wavelet!r}, only 'haar' is currently supported")

    initialized = 0
    with torch.no_grad():
        for name, param in module.named_parameters():
            if "lora_A" not in name or not name.endswith("weight"):
                continue
            if param.ndim != 2:
                continue

            rows, cols = param.shape
            if rows == 0 or cols == 0:
                continue

            if method in {"dct", "dct_lora"}:
                init_w = _dct_lora_a(rows, cols, param.device)
            else:
                base = torch.randn(rows, cols, device=param.device, dtype=torch.float32)
                base = _haar_wavelet_rows(base)

                if rows <= cols:
                    q, _ = torch.linalg.qr(base.t(), mode="reduced")
                    init_w = q.t()
                else:
                    # rows > cols cannot be fully row-orthogonal; normalize each row instead.
                    init_w = F.normalize(base, p=2, dim=1, eps=1e-12)

            param.copy_(init_w.to(dtype=param.dtype))
            initialized += 1
    return initialized


class ActorRolloutRefWorker(Worker, DistProfilerExtension):
    """
    This worker can be instantiated as a standalone actor or a standalone rollout or a standalone reference policy
    or a hybrid engine based on the config.rollout
    """

    def __init__(self, config: DictConfig, role: str, **kwargs):
        Worker.__init__(self)

        self.config = config
        import torch.distributed

        if not torch.distributed.is_initialized():
            rank = int(os.environ.get("RANK", 0))
            world_size = int(os.environ.get("WORLD_SIZE", 1))
            torch.distributed.init_process_group(
                backend=f"cpu:gloo,{get_device_name()}:{get_nccl_backend()}",
                rank=rank,
                world_size=world_size,
                timeout=datetime.timedelta(seconds=self.config.get("nccl_timeout", 600)),
                init_method=os.environ.get("DIST_INIT_METHOD", None),
            )

        # build device mesh for FSDP
        world_size = torch.distributed.get_world_size()
        # TODO(sgm): support FSDP hybrid shard for larger model
        self.device_mesh = create_device_mesh(world_size=world_size, fsdp_size=self.config.actor.fsdp_config.fsdp_size)

        # build device mesh for Ulysses Sequence Parallel
        self.ulysses_device_mesh = None
        self.ulysses_sequence_parallel_size = self.config.actor.get("ulysses_sequence_parallel_size", 1)
        dp = world_size // self.ulysses_sequence_parallel_size
        if self.ulysses_sequence_parallel_size > 1:
            self.ulysses_device_mesh = init_device_mesh(
                device_name, mesh_shape=(dp, self.ulysses_sequence_parallel_size), mesh_dim_names=["dp", "sp"]
            )

        # create training dispatch
        if self.ulysses_device_mesh is not None:
            is_collect = self.ulysses_device_mesh["sp"].get_local_rank() == 0
            self._register_dispatch_collect_info(
                "actor", dp_rank=self.ulysses_device_mesh["dp"].get_local_rank(), is_collect=is_collect
            )
        else:
            self._register_dispatch_collect_info("actor", dp_rank=self.rank, is_collect=True)

        self.ulysses_sharding_manager = FSDPUlyssesShardingManager(self.ulysses_device_mesh)
        self._lora_rank = self.config.model.get("lora_rank", 0)
        self._is_lora = self.config.model.get("lora_adapter_path") is not None or self._lora_rank > 0
        # True when the base model weights were modified during adapter init (PiSSA/MiLoRA).
        # In that case disable_adapter() returns W_res != W_pretrained, so ref computation
        # must use the saved initial adapter weights instead of disabling the adapter.
        self._is_svd_init = self.config.model.get("lora_residual_base", False)
        self._init_lora_state: dict = {}  # {param_name: CPU tensor} saved before FSDP wrap
        self._svd_ref_adapter_name: str | None = None
        self._debug_step0_base_kl_logged = False

        self.role = role
        assert self.role in ["actor", "rollout", "ref", "actor_rollout", "actor_rollout_ref"]

        self._is_actor = self.role in ["actor", "actor_rollout", "actor_rollout_ref"]
        self._is_rollout = self.role in ["rollout", "actor_rollout", "actor_rollout_ref"]
        self._is_ref = self.role in ["ref", "actor_rollout_ref"]
        self.use_orig_params = self.config.actor.fsdp_config.get("use_orig_params", False)
        if self._is_lora or self.config.model.get("use_dora", False):
            # Keep PEFT trainable params as original tensors so FSDP does not flatten
            # them together with frozen base weights and silently skip updates.
            self.use_orig_params = True

        # TODO(haibin.lin):
        # As of now the type of config is DictConfig, if we assign config.profiler with ProfilerConfig,
        # it will actually convert the ProfilerConfig dataclass back to a DictConfig.
        # We can still use ProfilerConfig for testing purpose (tests/utils/test_nvtx_profile.py)
        # as they provides DictConfig-like interface
        # The benefit of creating the dataclass config is to perform validation during __post_init__
        if self._is_actor:
            omega_profiler_config = config.actor.get("profiler", {})
        elif self._is_rollout:
            # NOTE: In colocation mode, rollout config may not take effect (follow the actor config)
            # This is for extendability in AsyncRL cases
            omega_profiler_config = config.rollout.get("profiler", {})
        elif self._is_ref:
            omega_profiler_config = config.ref.get("profiler", {})
        else:
            raise ValueError(
                f"Invalid role {self.role}, should be one of "
                "['actor', 'rollout', 'ref', 'actor_rollout', 'actor_rollout_ref']"
            )
        # omega_profiler_config is DictConfig
        # profiler_config is a ProfilerConfig dataclass
        profiler_config = omega_conf_to_dataclass(omega_profiler_config, dataclass_type=ProfilerConfig)
        if omega_profiler_config.get("tool", None) in ["npu", "nsys", "torch", "torch_memory"]:
            tool_config = omega_conf_to_dataclass(
                omega_profiler_config.get("tool_config", {}).get(omega_profiler_config.get("tool"))
            )
        else:
            tool_config = None
        DistProfilerExtension.__init__(
            self, DistProfiler(rank=self.rank, config=profiler_config, tool_config=tool_config)
        )

        self._is_offload_param = False
        self._is_offload_optimizer = False
        if self._is_actor:
            self._is_offload_param = self.config.actor.fsdp_config.get("param_offload", False)
            self._is_offload_optimizer = self.config.actor.fsdp_config.get("optimizer_offload", False)
        elif self._is_ref:
            # TODO: it seems that manual offload is slowly than FSDP offload
            self._is_offload_param = self.config.ref.fsdp_config.get("param_offload", False)

        # normalize config
        if self._is_actor:
            self.config.actor.ppo_mini_batch_size *= self.config.rollout.n
            self.config.actor.ppo_mini_batch_size //= self.device_mesh.size() // self.ulysses_sequence_parallel_size
            assert self.config.actor.ppo_mini_batch_size > 0, (
                f"ppo_mini_batch_size {self.config.actor.ppo_mini_batch_size} should be larger than 0 after "
                f"normalization"
            )
            # micro bsz
            if self.config.actor.ppo_micro_batch_size is not None:
                self.config.actor.ppo_micro_batch_size //= (
                    self.device_mesh.size() // self.ulysses_sequence_parallel_size
                )
                self.config.actor.ppo_micro_batch_size_per_gpu = self.config.actor.ppo_micro_batch_size

            if self.config.actor.ppo_micro_batch_size_per_gpu is not None:
                assert self.config.actor.ppo_mini_batch_size % self.config.actor.ppo_micro_batch_size_per_gpu == 0, (
                    f"normalized ppo_mini_batch_size {self.config.actor.ppo_mini_batch_size} should be divisible by "
                    f"ppo_micro_batch_size_per_gpu {self.config.actor.ppo_micro_batch_size_per_gpu}"
                )
                assert self.config.actor.ppo_mini_batch_size // self.config.actor.ppo_micro_batch_size_per_gpu > 0, (
                    f"normalized ppo_mini_batch_size {self.config.actor.ppo_mini_batch_size} should be larger than "
                    f"ppo_micro_batch_size_per_gpu {self.config.actor.ppo_micro_batch_size_per_gpu}"
                )

        # normalize rollout config
        if self._is_rollout and self.config.rollout.log_prob_micro_batch_size is not None:
            self.config.rollout.log_prob_micro_batch_size //= (
                self.device_mesh.size() // self.ulysses_sequence_parallel_size
            )
            self.config.rollout.log_prob_micro_batch_size_per_gpu = self.config.rollout.log_prob_micro_batch_size
        # normalize ref config
        if self._is_ref and self.config.ref.log_prob_micro_batch_size is not None:
            self.config.ref.log_prob_micro_batch_size //= self.device_mesh.size() // self.ulysses_sequence_parallel_size
            self.config.ref.log_prob_micro_batch_size_per_gpu = self.config.ref.log_prob_micro_batch_size

    def _build_model_optimizer(
        self,
        model_path,
        fsdp_config: FSDPEngineConfig,
        optim_config,
        override_model_config,
        use_remove_padding=False,
        use_fused_kernels=False,
        enable_gradient_checkpointing=False,
        trust_remote_code=False,
        use_liger=False,
        role="actor",
        enable_activation_offload=False,
    ):
        from torch.distributed.fsdp import CPUOffload, MixedPrecision
        from transformers import (
            AutoConfig,
            AutoModel,
            AutoModelForCausalLM,
            AutoModelForImageTextToText,
            AutoModelForVision2Seq,
        )

        from verl.utils.model import get_generation_config, print_model_size, update_model_config
        from verl.utils.torch_dtypes import PrecisionType

        assert role in ["actor", "ref"]

        log_gpu_memory_usage(f"Before init {role} from HF AutoModel", logger=logger)
        local_path = model_path
        aux_model_path = local_path
        if self.config.model.get("lora_residual_base", False):
            aux_model_path = _resolve_svd_init_aux_path(local_path, self.config.model.get("lora_adapter_path")) or local_path

        # note that we have to create model in fp32. Otherwise, the optimizer is in bf16, which is incorrect
        # TODO(zhangchi.usc1992): 1. support create from random initialized model. 2. Support init with FSDP directly
        self.tokenizer = hf_tokenizer(aux_model_path, trust_remote_code=trust_remote_code)
        self.processor = hf_processor(aux_model_path, trust_remote_code=trust_remote_code)

        if self.config.model.get("custom_chat_template", None) is not None:
            if self.processor is not None:
                self.processor.chat_template = self.config.model.custom_chat_template
            else:
                self.tokenizer.chat_template = self.config.model.custom_chat_template

        torch_dtype = fsdp_config.get("model_dtype", None)
        if torch_dtype is None:
            torch_dtype = torch.float32 if self._is_actor else torch.bfloat16
        else:
            torch_dtype = PrecisionType.to_dtype(torch_dtype)

        # override model kwargs
        attn_implementation = override_model_config.get("attn_implementation", "flash_attention_2")
        actor_model_config = AutoConfig.from_pretrained(
            aux_model_path, trust_remote_code=trust_remote_code, attn_implementation=attn_implementation
        )
        # TODO: VL models use VisionAttention, which directly uses flash_attention in transformers>=4.53
        # which will be patched by _ulysses_flash_attention_forward, but errorly misses position_ids
        # Maybe support Ulysses in VisionAttention in the future and remove this patch
        if self.ulysses_sequence_parallel_size > 1 and hasattr(actor_model_config, "vision_config"):
            actor_model_config.vision_config._attn_implementation = "eager"

        # patch for kimi-vl
        if getattr(actor_model_config, "model_type", None) == "kimi_vl":
            actor_model_config.text_config.topk_method = "greedy"

        self.generation_config = get_generation_config(aux_model_path, trust_remote_code=trust_remote_code)

        override_config_kwargs = {
            "bos_token_id": self.tokenizer.bos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        override_config_kwargs.update(override_model_config)
        update_model_config(actor_model_config, override_config_kwargs=override_config_kwargs)
        if self.rank == 0:
            print(f"Model config after override: {actor_model_config}")

        # NOTE(fix me): tie_word_embedding causes meta_tensor init to hang
        init_context = get_init_weight_context_manager(
            use_meta_tensor=not actor_model_config.tie_word_embeddings, mesh=self.device_mesh
        )

        with init_context(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            has_remote_code = hasattr(actor_model_config, "auto_map") and any(
                actor_model_config.architectures[0] in val for val in actor_model_config.auto_map.values()
            )
            if has_remote_code:
                auto_class = next(
                    k for k, v in actor_model_config.auto_map.items() if actor_model_config.architectures[0] in v
                )
                match auto_class:
                    case "AutoModelForVision2Seq":
                        actor_module_class = AutoModelForVision2Seq
                    case "AutoModelForCausalLM":
                        actor_module_class = AutoModelForCausalLM
                    case "AutoModelForImageTextToText":
                        actor_module_class = AutoModelForImageTextToText
                    case _:
                        actor_module_class = AutoModel
            else:
                if type(actor_model_config) in AutoModelForVision2Seq._model_mapping.keys():
                    actor_module_class = AutoModelForVision2Seq
                elif type(actor_model_config) in AutoModelForCausalLM._model_mapping.keys():
                    actor_module_class = AutoModelForCausalLM
                elif type(actor_model_config) in AutoModelForImageTextToText._model_mapping.keys():
                    actor_module_class = AutoModelForImageTextToText
                else:
                    actor_module_class = AutoModel

            actor_module = actor_module_class.from_pretrained(
                pretrained_model_name_or_path=local_path,
                torch_dtype=torch_dtype,
                config=actor_model_config,
                trust_remote_code=trust_remote_code,
                attn_implementation=attn_implementation,
            )

            # Apply Liger kernel to the model if use_liger is set to True
            if use_liger:
                from liger_kernel.transformers.monkey_patch import _apply_liger_kernel_to_instance

                _apply_liger_kernel_to_instance(model=actor_module)

            fused_kernel_options = self.config.model.get("fused_kernel_options", None)
            fused_kernels_backend = (
                fused_kernel_options.get("impl_backend", None) if fused_kernel_options is not None else None
            )

            apply_monkey_patch(
                model=actor_module,
                use_remove_padding=use_remove_padding,
                ulysses_sp_size=self.ulysses_sequence_parallel_size,
                use_fused_kernels=use_fused_kernels,
                fused_kernels_backend=fused_kernels_backend,
            )

            # some parameters may not in torch_dtype. TODO(zhangchi.usc1992) remove this after we switch to fsdp2
            actor_module.to(torch_dtype)

            if enable_gradient_checkpointing:
                actor_module.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

        if self._is_lora:
            print("Applying LoRA to actor module")
            actor_module.enable_input_require_grads()

            lora_adapter_path = self.config.model.get("lora_adapter_path")
            if lora_adapter_path is not None:
                from copy import deepcopy

                from peft import PeftConfig, PeftModel, set_peft_model_state_dict
                from peft.utils.save_and_load import load_peft_weights

                print(f"Loading pre-trained LoRA adapter to {role} from: {lora_adapter_path}")

                # Copy adapter to local if needed
                local_adapter_path = copy_to_local(lora_adapter_path, use_shm=self.config.model.get("use_shm", False))

                adapter_config = PeftConfig.from_pretrained(local_adapter_path)
                if self._is_svd_init:
                    # The checkpoint at model.path is already W_res. Re-running
                    # PiSSA/OLoRA initialization here would subtract another SVD
                    # delta before the saved adapter tensors are loaded.
                    adapter_config.init_lora_weights = False

                actor_module = PeftModel.from_pretrained(
                    actor_module,
                    local_adapter_path,
                    is_trainable=True,
                    config=adapter_config,
                    # PiSSA/SVD-init needs real adapter tensors during its
                    # initialization; meta tensors cannot accept the computed
                    # CUDA factors via set_data().
                    low_cpu_mem_usage=not self._is_svd_init,
                )
                peft_config = actor_module.peft_config["default"]
                # Ensure task_type is TaskType enum, not string
                if isinstance(peft_config.task_type, str):
                    peft_config.task_type = TaskType.CAUSAL_LM

                # For SVD-init adapters (PiSSA/MiLoRA) the base layer stores
                # W_res = W_pretrained - delta_init, so disable_adapter() gives
                # the wrong ref. Save the initial adapter and load a frozen copy as
                # a dedicated reference adapter so the correct ref path stays on the
                # standard PEFT forward without FSDP summon_full_params.
                if self._is_svd_init:
                    deferred = 0
                    for name, param in actor_module.named_parameters():
                        if "lora_A" not in name and "lora_B" not in name:
                            continue
                        if getattr(param, "is_meta", False):
                            deferred += 1
                            continue
                        self._init_lora_state[name] = param.data.clone().cpu()

                    self._svd_ref_adapter_name = "_base_ref"
                    # Build the frozen copy from an init-disabled config as well;
                    # load_adapter() would read the on-disk PiSSA config and
                    # mutate W_res a second time.
                    ref_config = deepcopy(peft_config)
                    ref_config.inference_mode = True
                    ref_config.init_lora_weights = False
                    actor_module.add_adapter(
                        self._svd_ref_adapter_name,
                        ref_config,
                        low_cpu_mem_usage=False,
                    )
                    ref_state = load_peft_weights(local_adapter_path, device="cpu")
                    set_peft_model_state_dict(
                        actor_module,
                        ref_state,
                        adapter_name=self._svd_ref_adapter_name,
                    )
                    for name, param in actor_module.named_parameters():
                        if f".{self._svd_ref_adapter_name}." in name:
                            param.requires_grad = False

                    if self.rank == 0:
                        if deferred > 0:
                            print(
                                f"[SVD-init ref] Deferred saving {deferred} initial adapter parameters "
                                "until after FSDP materialization."
                            )
                        else:
                            print(
                                f"[SVD-init ref] Saved {len(self._init_lora_state)} "
                                f"initial adapter parameters for correct ref computation."
                            )
                        print(
                            f"[SVD-init ref] Loaded frozen reference adapter '{self._svd_ref_adapter_name}' "
                            "for correct base-model KL computation."
                        )
                disabled_dropout = self._disable_lora_dropout(actor_module)
                if self.rank == 0 and disabled_dropout > 0:
                    print(f"[LoRA] Disabled {disabled_dropout} LoRA dropout modules on loaded adapter")
                self._maybe_freeze_lora_A(actor_module)

            else:
                # Convert config to regular Python types before creating PEFT model
                lora_config = {
                    "task_type": TaskType.CAUSAL_LM,
                    "r": self.config.model.lora_rank,
                    "lora_alpha": self.config.model.lora_alpha,
                    "use_dora": self.config.model.use_dora,
                    "target_modules": convert_to_regular_types(self.config.model.target_modules),
                    "exclude_modules": convert_to_regular_types(self.config.model.exclude_modules),
                    "bias": "none",
                }
                if self.config.model.get("use_adalora", False):
                    adalora_config = {
                        "task_type": TaskType.CAUSAL_LM,
                        "init_r": self.config.model.get("adalora_init_r") or self.config.model.lora_rank,
                        "target_r": self.config.model.get("adalora_target_r") or self.config.model.lora_rank,
                        "lora_alpha": self.config.model.lora_alpha,
                        "target_modules": convert_to_regular_types(self.config.model.target_modules),
                        "beta1": self.config.model.get("adalora_beta1", 0.85),
                        "beta2": self.config.model.get("adalora_beta2", 0.85),
                        "tinit": self.config.model.get("adalora_tinit", 0),
                        "tfinal": self.config.model.get("adalora_tfinal", 0),
                        "deltaT": self.config.model.get("adalora_delta_t", 1),
                        "total_step": self.config.actor.optim.get("total_training_steps", 0),
                        "orth_reg_weight": self.config.model.get("adalora_orth_reg_weight", 0.5),
                    }
                    actor_module = get_peft_model(actor_module, AdaLoraConfig(**adalora_config))
                    actor_module.to(torch_dtype)
                else:
                    actor_module = get_peft_model(actor_module, LoraConfig(**lora_config))
                disabled_dropout = self._disable_lora_dropout(actor_module)
                if self.rank == 0 and disabled_dropout > 0:
                    print(f"[LoRA] Disabled {disabled_dropout} LoRA dropout modules on actor")
                self._maybe_freeze_lora_A(actor_module)
                init_count = _init_lora_a_params(
                    actor_module,
                    init_method=self.config.model.get("lora_A_init_method", "default"),
                    wavelet=self.config.model.get("lora_A_wavelet", "haar"),
                )
                if self.rank == 0 and init_count > 0:
                    init_desc = f"method={self.config.model.get('lora_A_init_method')}"
                    if self.config.model.get("lora_A_init_method") == "wavelet_orth":
                        init_desc += f" wavelet={self.config.model.get('lora_A_wavelet', 'haar')}"
                    print(f"Applied LoRA-A init {init_desc} to {init_count} actor matrices.")

        self.use_orig_params = fsdp_config.get("use_orig_params", False)
        if self._is_lora or self.config.model.get("use_dora", False) or self.config.model.get("use_adalora", False):
            self.use_orig_params = True
            if self.rank == 0:
                print("[actor model] Enabling use_orig_params=True for LoRA/DoRA/AdaLoRA under FSDP.")
        if self.config.actor.get("freeze_vision_tower", False):
            vision_tower = get_vl_model_vision_tower(actor_module)
            if vision_tower is not None:
                vision_tower.requires_grad_(False)
                self.use_orig_params = True
                if self.rank == 0:
                    print("[actor model] Vision tower is set to not trainable.")
            else:
                if self.rank == 0:
                    print("[actor model] No vision tower found.")

        torch.distributed.barrier()

        if self.rank == 0:
            print_model_size(actor_module)

        log_gpu_memory_usage(f"After init {role} from HF AutoModel", logger=logger)

        # We wrap FSDP for rollout as well
        mixed_precision_config = fsdp_config.get("mixed_precision", None)
        if mixed_precision_config is not None:
            param_dtype = PrecisionType.to_dtype(mixed_precision_config.get("param_dtype", "bf16"))
            reduce_dtype = PrecisionType.to_dtype(mixed_precision_config.get("reduce_dtype", "fp32"))
            buffer_dtype = PrecisionType.to_dtype(mixed_precision_config.get("buffer_dtype", "fp32"))
        else:
            param_dtype = PrecisionType.to_dtype(fsdp_config.dtype)
            reduce_dtype = torch.float32
            buffer_dtype = torch.float32

        mixed_precision = MixedPrecision(param_dtype=param_dtype, reduce_dtype=reduce_dtype, buffer_dtype=buffer_dtype)

        auto_wrap_policy = get_fsdp_wrap_policy(
            module=actor_module,
            config=fsdp_config.get("wrap_policy", None),
            is_lora=self._is_lora
            and not self.config.model.get("use_dora", False)
            and not self.config.model.get("use_adalora", False),
        )

        # if self._is_rollout and self.config.rollout.name == "hf":
        #     # TODO(zhangchi.usc1992, shengguangming) fix me.
        #     Current, auto_wrap_policy causes HFRollout to hang in Gemma
        #     auto_wrap_policy = None

        if self.rank == 0:
            print(f"wrap_policy: {auto_wrap_policy}")

        fsdp_mesh = self.device_mesh
        sharding_strategy = get_sharding_strategy(fsdp_mesh)

        # TODO: add transformer policy
        # We force reference policy to use CPUOffload to save memory.
        # We force turn off CPUOffload for actor because it causes incorrect results when using grad accumulation
        cpu_offload = None if role == "actor" else CPUOffload(offload_params=True)
        fsdp_strategy = self.config.actor.strategy
        if fsdp_strategy == "fsdp":
            actor_module_fsdp = FSDP(
                actor_module,
                cpu_offload=cpu_offload,
                param_init_fn=init_fn,
                auto_wrap_policy=auto_wrap_policy,
                device_id=get_device_id(),
                sharding_strategy=sharding_strategy,  # zero3
                mixed_precision=mixed_precision,
                sync_module_states=True,
                device_mesh=self.device_mesh,
                use_orig_params=self.use_orig_params,
                forward_prefetch=fsdp_config.get("forward_prefetch", False),
            )
        elif fsdp_strategy == "fsdp2":
            assert CPUOffloadPolicy is not None, "PyTorch version >= 2.4 is required for using fully_shard API (FSDP2)"
            mp_policy = MixedPrecisionPolicy(
                param_dtype=param_dtype, reduce_dtype=reduce_dtype, cast_forward_inputs=True
            )
            if role == "actor" and fsdp_config.offload_policy:
                cpu_offload = CPUOffloadPolicy(pin_memory=True)
                self._is_offload_param = False
                self._is_offload_optimizer = False
            else:
                cpu_offload = None if role == "actor" else CPUOffloadPolicy(pin_memory=True)

            fsdp_kwargs = {
                "mesh": fsdp_mesh,
                "mp_policy": mp_policy,
                "offload_policy": cpu_offload,
                "reshard_after_forward": fsdp_config.reshard_after_forward,
                "shard_placement_fn": get_shard_placement_fn(fsdp_size=self.device_mesh.shape[-1]),
            }
            full_state = actor_module.state_dict()
            apply_fsdp2(actor_module, fsdp_kwargs, fsdp_config)
            fsdp2_load_full_state_dict(actor_module, full_state, fsdp_mesh, cpu_offload)
            actor_module_fsdp = actor_module
        else:
            raise NotImplementedError(f"not implement {fsdp_strategy}")

        if enable_activation_offload:
            enable_activation_offloading(actor_module_fsdp, fsdp_strategy, enable_gradient_checkpointing)

        log_gpu_memory_usage(f"After {role} FSDP init", logger=logger)

        # TODO: add more optimizer args into config
        if role == "actor" and optim_config is not None:
            from copy import deepcopy

            from verl.utils.torch_functional import get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup

            optim_config = deepcopy(optim_config)
            if self.config.model.use_dora:
                override_cfg = dict(optim_config.override_optimizer_config or {})
                override_cfg.setdefault("foreach", False)
                optim_config.override_optimizer_config = override_cfg
                if self.rank == 0:
                    print("Disabling foreach optimizer path for DoRA actor training.")

            actor_optimizer = build_optimizer(actor_module_fsdp.parameters(), optim_config)

            total_steps = optim_config.get("total_training_steps", 0)
            num_warmup_steps = int(optim_config.get("lr_warmup_steps", -1))
            lr_scheduler_type = optim_config.get("lr_scheduler_type", "constant")
            min_lr_ratio = optim_config.get("min_lr_ratio", 0.0)
            num_cycles = optim_config.get("num_cycles", 0.5)
            if num_warmup_steps < 0:
                num_warmup_steps_ratio = optim_config.get("lr_warmup_steps_ratio", 0.0)
                num_warmup_steps = int(num_warmup_steps_ratio * total_steps)

            if self.rank == 0:
                print(f"Total steps: {total_steps}, num_warmup_steps: {num_warmup_steps}")

            if lr_scheduler_type == "constant":
                actor_lr_scheduler = get_constant_schedule_with_warmup(
                    optimizer=actor_optimizer, num_warmup_steps=num_warmup_steps
                )
            elif lr_scheduler_type == "cosine":
                actor_lr_scheduler = get_cosine_schedule_with_warmup(
                    optimizer=actor_optimizer,
                    num_warmup_steps=num_warmup_steps,
                    num_training_steps=total_steps,
                    min_lr_ratio=min_lr_ratio,
                    num_cycles=num_cycles,
                )
            else:
                raise NotImplementedError(f"LR scheduler type {lr_scheduler_type} is not supported")

            log_gpu_memory_usage(f"After {role} optimizer init", logger=logger)
        else:
            actor_optimizer = None
            actor_lr_scheduler = None

        return actor_module_fsdp, actor_optimizer, actor_lr_scheduler, actor_model_config

    def _build_rollout(self, trust_remote_code=False):
        from torch.distributed.device_mesh import init_device_mesh

        # 1. parse rollout and huggingface model config
        rollout_config: RolloutConfig = omega_conf_to_dataclass(self.config.rollout)
        model_config: HFModelConfig = omega_conf_to_dataclass(self.config.model, dataclass_type=HFModelConfig)
        self.model_config = model_config

        # 2. build rollout device mesh
        infer_tp = self.config.rollout.tensor_model_parallel_size * self.config.rollout.data_parallel_size
        infer_pp = self.config.rollout.pipeline_model_parallel_size
        infer_world_size = infer_tp * infer_pp
        dp = self.world_size // infer_world_size
        assert self.world_size % infer_world_size == 0, (
            f"rollout world_size: {self.world_size} is not divisible by infer_world_size: {infer_world_size}"
        )
        rollout_device_mesh = init_device_mesh(
            device_name, mesh_shape=(dp, infer_tp, infer_pp), mesh_dim_names=["dp", "infer_tp", "infer_pp"]
        )
        rollout_name = self.config.rollout.name

        self.rollout_device_mesh = rollout_device_mesh

        if rollout_name == "hf":
            self._register_dispatch_collect_info("rollout", dp_rank=self.rank, is_collect=True)
        else:
            is_collect = (
                rollout_device_mesh["infer_tp"].get_local_rank() == 0
                and rollout_device_mesh["infer_pp"].get_local_rank() == 0
            )
            self._register_dispatch_collect_info(
                "rollout", dp_rank=rollout_device_mesh["dp"].get_local_rank(), is_collect=is_collect
            )

        # 3. init trainer and rollout random states
        self.torch_random_states = get_torch_device().get_rng_state()
        gen_dp_rank = rollout_device_mesh["dp"].get_local_rank()
        get_torch_device().manual_seed(gen_dp_rank + 1000)  # make sure all tp ranks have the same random states
        self.gen_random_states = get_torch_device().get_rng_state()
        get_torch_device().set_rng_state(self.torch_random_states)

        # 4. build rollout model
        log_gpu_memory_usage(f"Before building {self.config.rollout.name} rollout", logger=logger)
        self.rollout = get_rollout_class(rollout_config.name, rollout_config.mode)(
            config=rollout_config, model_config=model_config, device_mesh=rollout_device_mesh
        )
        log_gpu_memory_usage(f"After building {self.config.rollout.name} rollout", logger=logger)

        # Full params
        if torch.distributed.get_world_size() == 1 and fsdp_version(self.actor_module_fsdp) == 1:
            FSDP.set_state_dict_type(
                self.actor_module_fsdp,
                state_dict_type=StateDictType.FULL_STATE_DICT,
                state_dict_config=FullStateDictConfig(),
            )
        elif fsdp_version(self.actor_module_fsdp) == 1:
            FSDP.set_state_dict_type(
                self.actor_module_fsdp,
                state_dict_type=StateDictType.SHARDED_STATE_DICT,
                state_dict_config=ShardedStateDictConfig(),
            )

        # used for LoRA
        self.base_sync_done: bool = "dummy" not in self.config.rollout.load_format
        self.layered_summon = self.config.rollout.get("layered_summon", False)

        # 5. switch to trainer mode
        # NOTE: It's critical that hybrid engine in trainer mode initially to load checkpoint.
        # For sync mode, we directly switch to trainer mode here.
        # For async mode, we can't call run_until_complete here, so we will switch to trainer mode in AgentLoopManager.
        if rollout_config.mode == "sync" and self._is_actor:
            loop = get_event_loop()
            loop.run_until_complete(self.trainer_mode())

    async def rollout_mode(self):
        """Context switch hybridengine to rollout mode."""
        aggressive_empty_cache(force_sync=True)

        log_gpu_memory_usage("Before load_fsdp_model_to_gpu", logger=logger)
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)
        log_gpu_memory_usage("After load_fsdp_model_to_gpu", logger=logger)

        peft_config = None
        rollout_sync_mode = "full_weights"
        peft_model = getattr(self.actor_module_fsdp, "_fsdp_wrapped_module", self.actor_module_fsdp)
        if hasattr(peft_model, "peft_config"):  # LoRA / DoRA
            peft_config = peft_model.peft_config.get("default", None)
            if self._is_svd_init and not self.base_sync_done:
                logger.info(
                    "Rollout first sync for SVD-init adapters uses the original pretrained base model "
                    "instead of merged residual weights so vLLM matches the LoRA baseline path."
                )
                params = self._collect_svd_init_pretrained_rollout_params(peft_config)
                # SVD-init variants (PiSSA, MiLoRA, OLoRA residual-base bundles) still run vLLM
                # with LoRA wrappers enabled, so pretrained full weights must target base_layer.*
                # keys just like the normal first-sync LoRA path.
                params = {replace_lora_wrapper(k, peft_config): v for k, v in params.items()}
                peft_config = None
                rollout_sync_mode = "svd_init_original_pretrained_full_weights"
            elif getattr(peft_config, "use_dora", False) or str(getattr(peft_config, "peft_type", "")).endswith(
                "ADALORA"
            ):
                logger.info(
                    "Rollout sync uses merged full weights for DoRA/AdaLoRA because "
                    "vLLM adapter loading is unsupported or unsafe for this PEFT type."
                )
                params = collect_merged_peft_params(module=self.actor_module_fsdp)
                peft_config = None
                rollout_sync_mode = "merged_full_weights"
            else:
                params = collect_lora_params(
                    module=self.actor_module_fsdp,
                    layered_summon=self.config.rollout.get("layered_summon", False),
                    base_sync_done=self.base_sync_done,
                )
                rollout_sync_mode = "lora_delta" if self.base_sync_done else "initial_base_plus_lora_named_weights"
                if not self.base_sync_done:
                    params = {replace_lora_wrapper(k, peft_config): v for k, v in params.items()}
        else:
            params = self.actor_module_fsdp.state_dict()

        if self.rank == 0:
            with contextlib.suppress(Exception):
                sync_stage = "rollout_sync_initial" if not self.base_sync_done else "rollout_sync_latest"
                debug_path = _write_rollout_policy_debug(
                    sync_stage,
                    {
                        "experiment_name": str(getattr(self, "experiment_name", "unknown_exp")),
                        "rank": int(self.rank),
                        "base_sync_done_before": bool(self.base_sync_done),
                        "is_svd_init": bool(self._is_svd_init),
                        "is_lora": bool(self._is_lora),
                        "rollout_sync_mode": rollout_sync_mode,
                        "rollout_model_path": str(self.config.model.path),
                        "rollout_has_peft_config": peft_config is not None,
                        "peft_type": str(getattr(peft_config, "peft_type", "none")) if peft_config is not None else "none",
                        "active_adapter": (
                            peft_model.active_adapters[0] if getattr(peft_model, "active_adapters", None) else None
                        ),
                        "lora_adapter_path": str(self.config.model.get("lora_adapter_path", "")),
                    },
                )
                print(f"[DEBUG rollout policy] wrote {debug_path}")

        params = convert_weight_keys(
            params, getattr(self.actor_module_fsdp, "_fsdp_wrapped_module", self.actor_module_fsdp)
        )

        # Special handling for LoRA with sleep_level=2:
        # When sleep_level=2, base model weights are destroyed during each sleep cycle.
        # separately collect and update LoRA weights and base model weights through their respective interfaces.
        # Here: params contains LoRA weights, base_model_params contains base model weights.
        if peft_config is not None and getattr(self.rollout, "sleep_level", None) == 2:
            base_model_params = collect_lora_params(
                module=self.actor_module_fsdp,
                layered_summon=self.layered_summon,
                base_sync_done=False,
            )
            base_model_params = {replace_lora_wrapper(k, peft_config): v for k, v in base_model_params.items()}
            base_model_params = convert_weight_keys(
                base_model_params, getattr(self.actor_module_fsdp, "_fsdp_wrapped_module", self.actor_module_fsdp)
            )

        log_gpu_memory_usage("Before offload_fsdp_model_to_cpu", logger=logger)
        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)
        log_gpu_memory_usage("After offload_fsdp_model_to_cpu", logger=logger)

        set_expandable_segments(False)

        if peft_config is not None and self.base_sync_done:
            per_tensor_param = params.items() if isinstance(params, dict) else params  # Fixed: handle dict case
        else:
            device = get_device_id()  # used when fsdp2 set cpu_offload_policy
            per_tensor_param = (
                (name, param.to(device, non_blocking=True).full_tensor() if isinstance(param, DTensor) else param)
                for name, param in params.items()
            )

        if self.config.rollout.free_cache_engine:
            await self.rollout.resume(tags=["weights"])
        log_gpu_memory_usage("After resume weights", logger=logger)

        if peft_config is not None and getattr(self.rollout, "sleep_level", None) == 2:
            per_tensor_base_params = (
                (name, param.to(device, non_blocking=True).full_tensor() if isinstance(param, DTensor) else param)
                for name, param in base_model_params.items()
            )
            await self.rollout.update_weights(per_tensor_base_params, base_sync_done=False)
            del base_model_params, per_tensor_base_params

        await self.rollout.update_weights(per_tensor_param, peft_config=peft_config, base_sync_done=self.base_sync_done)
        log_gpu_memory_usage("After update_weights", logger=logger)
        del params, per_tensor_param
        aggressive_empty_cache(force_sync=True)
        if self.config.rollout.free_cache_engine:
            await self.rollout.resume(tags=["kv_cache"])
        log_gpu_memory_usage("After resume kv_cache", logger=logger)

        self.base_sync_done = True
        # important: need to manually set the random states of each tp to be identical.
        self.torch_random_states = get_torch_device().get_rng_state()
        get_torch_device().set_rng_state(self.gen_random_states)

    async def trainer_mode(self):
        """Context switch hybridengine to trainer mode."""
        if self.config.rollout.free_cache_engine:
            log_gpu_memory_usage("Before rollout offload", logger=logger)
            await self.rollout.release()
            log_gpu_memory_usage("After rollout offload", logger=logger)

        self.actor_module_fsdp.train()

        # add empty cache after each compute
        aggressive_empty_cache(force_sync=True)

        set_expandable_segments(True)

        # restore random states
        self.gen_random_states = get_torch_device().get_rng_state()
        get_torch_device().set_rng_state(self.torch_random_states)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        from verl.workers.actor import DataParallelPPOActor

        # This is used to import external_lib into the huggingface systems
        import_external_libs(self.config.model.get("external_lib", None))

        override_model_config = OmegaConf.to_container(OmegaConf.create(self.config.model.get("override_config", {})))
        use_remove_padding = self.config.model.get("use_remove_padding", False)
        use_shm = self.config.model.get("use_shm", False)
        use_fused_kernels = self.config.model.get("use_fused_kernels", False)

        if self._is_actor or self._is_rollout:
            # we need the model for actor and rollout
            if self._is_actor:
                optim_config = self.config.actor.optim
                fsdp_config = omega_conf_to_dataclass(self.config.actor.fsdp_config)
            else:
                optim_config = None
                fsdp_config = FSDPEngineConfig()

            local_path = copy_to_local(self.config.model.path, use_shm=use_shm)
            (
                self.actor_module_fsdp,
                self.actor_optimizer,
                self.actor_lr_scheduler,
                self.actor_model_config,
            ) = self._build_model_optimizer(
                model_path=local_path,
                fsdp_config=fsdp_config,
                optim_config=optim_config,
                override_model_config=override_model_config,
                use_remove_padding=use_remove_padding,
                use_fused_kernels=use_fused_kernels,
                enable_gradient_checkpointing=self.config.model.get("enable_gradient_checkpointing", False),
                trust_remote_code=self.config.model.get("trust_remote_code", False),
                use_liger=self.config.model.get("use_liger", False),
                role="actor",
                enable_activation_offload=self.config.model.get("enable_activation_offload", False),
            )

            # get the original unwrapped module
            if fsdp_version(self.actor_module_fsdp) == 1:
                self.actor_module = self.actor_module_fsdp._fsdp_wrapped_module

            if self._is_offload_param:
                offload_fsdp_model_to_cpu(self.actor_module_fsdp)
                log_gpu_memory_usage("After offload actor model during init", logger=logger)

            if self._is_offload_optimizer:
                offload_fsdp_optimizer(optimizer=self.actor_optimizer)
                log_gpu_memory_usage("After offload actor optimizer during init", logger=logger)

        if self._is_actor:
            actor_cfg = omega_conf_to_dataclass(self.config.actor)
            self.actor = DataParallelPPOActor(
                config=actor_cfg, actor_module=self.actor_module_fsdp, actor_optimizer=self.actor_optimizer
            )

        if self._is_rollout:
            self._build_rollout(trust_remote_code=self.config.model.get("trust_remote_code", False))

        if self._is_ref:
            ref_model_path = self.config.model.path
            ref_model = self.config.ref.get("model", None)
            if ref_model is not None:
                ref_model_path = ref_model.get("path", self.config.model.path)

            if self.rank == 0:
                print("reference model:", ref_model_path)
            local_path = copy_to_local(ref_model_path, use_shm=use_shm)
            self.ref_module_fsdp = self._build_model_optimizer(
                model_path=local_path,
                fsdp_config=omega_conf_to_dataclass(self.config.ref.fsdp_config),
                optim_config=None,
                override_model_config=override_model_config,
                use_remove_padding=use_remove_padding,
                use_fused_kernels=use_fused_kernels,
                trust_remote_code=self.config.model.get("trust_remote_code", False),
                use_liger=self.config.model.get("use_liger", False),
                role="ref",
            )[0]
            OmegaConf.set_struct(self.config.ref, True)
            with open_dict(self.config.ref):
                self.config.ref.use_remove_padding = use_remove_padding
                self.config.ref.use_fused_kernels = use_fused_kernels
            self.ref_policy = DataParallelPPOActor(config=self.config.ref, actor_module=self.ref_module_fsdp)

        if self._is_actor:
            self.flops_counter = FlopsCounter(self.actor_model_config)
            self.checkpoint_manager = FSDPCheckpointManager(
                model=self.actor_module_fsdp,
                optimizer=self.actor.actor_optimizer,
                lr_scheduler=self.actor_lr_scheduler,
                processing_class=self.processor if self.processor is not None else self.tokenizer,
                checkpoint_config=self.config.actor.checkpoint,
            )

        if not self._is_actor and self._is_rollout:
            # If ActorRolloutRefWorker is initialized as a standalone rollout,
            # create a checkpoint manager for FSDP model to allow loading FSDP checkpoints for rollout.

            checkpoint_contents = OmegaConf.create({"load_contents": ["model"], "save_contents": []})
            self.checkpoint_manager = FSDPCheckpointManager(
                model=self.actor_module_fsdp,
                optimizer=None,
                lr_scheduler=None,
                processing_class=self.processor if self.processor is not None else self.tokenizer,
                checkpoint_config=checkpoint_contents,
            )

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    @DistProfiler.annotate(color="red", role="actor_update")
    def update_actor(self, data: DataProto):
        assert self._is_actor
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)
        if self._is_offload_optimizer:
            load_fsdp_optimizer(optimizer=self.actor_optimizer, device_id=get_device_id())

        with self.ulysses_sharding_manager:
            # Base-KL logging can route meta_info through non-dict containers on some Ray paths.
            with contextlib.suppress(Exception):
                from omegaconf import DictConfig as OmegaDictConfig
                from omegaconf import OmegaConf

                if isinstance(data.meta_info, OmegaDictConfig):
                    normalized = OmegaConf.to_container(data.meta_info, resolve=True)
                    if isinstance(normalized, dict):
                        data.meta_info = normalized

            if not isinstance(data.meta_info, dict):
                with contextlib.suppress(Exception):
                    data.meta_info = dict(data.meta_info)
            if not isinstance(data.meta_info, dict):
                data.meta_info = {}

            pre_update_base_model_kl = None
            safe_base_ref_log_prob = None
            if data.meta_info.get("log_base_model_kl", False) and "base_ref_log_prob" not in data.batch:
                data = data.union(self._compute_base_log_prob_from_actor(data, output_key="base_ref_log_prob"))
                if "old_log_probs" in data.batch:
                    response_mask = data.batch.get("response_mask")
                    base_ref_log_prob = data.batch.get("base_ref_log_prob")
                    old_log_prob = data.batch.get("old_log_probs")
                    if response_mask is not None and base_ref_log_prob is not None and old_log_prob is not None:
                        negative_approx_base_kl = torch.clamp(old_log_prob - base_ref_log_prob, min=-20.0, max=20.0)
                        step0_base_model_kl = verl_F.masked_mean(-negative_approx_base_kl, response_mask)
                        pre_update_base_model_kl = float(step0_base_model_kl.detach().item())
                        data.meta_info["debug_step0_base_model_kl"] = pre_update_base_model_kl
                        safe_base_ref_log_prob = base_ref_log_prob.detach().clone().cpu().contiguous()
                        global_step = int(data.meta_info.get("global_steps", -1))
                        if not self._debug_step0_base_kl_logged and self.rank == 0:
                            self._debug_step0_base_kl_logged = True
                            debug_payload = {
                                "stage": "pre_update",
                                "experiment_name": str(data.meta_info.get("experiment_name", "unknown_exp")),
                                "global_step": global_step,
                                "rank": int(self.rank),
                                **_debug_base_kl_payload(old_log_prob, base_ref_log_prob, response_mask),
                            }
                            debug_path = _write_base_kl_debug("pre_update", debug_payload)
                            print(f"[DEBUG pre-update base KL] wrote {debug_path}")
                # Base-model KL logging may offload the actor back to CPU; reload before training forward.
                if self._is_offload_param:
                    load_fsdp_model_to_gpu(self.actor_module_fsdp)

            if safe_base_ref_log_prob is None and "base_ref_log_prob" in data.batch:
                safe_base_ref_log_prob = data.batch["base_ref_log_prob"].detach().clone().cpu().contiguous()

            data = data.to("cpu")  # data will to device with each micro batch on actor.update_policy
            if safe_base_ref_log_prob is not None:
                # TensorDict may be locked after the device move; overwrite the existing slot in-place.
                data.batch.set_("base_ref_log_prob", safe_base_ref_log_prob)

            # perform training
            with Timer(name="update_policy", logger=None) as timer:
                metrics = self.actor.update_policy(data=data)
            if pre_update_base_model_kl is not None:
                metrics["actor/base_model_kl_pre_update"] = pre_update_base_model_kl
            delta_time = timer.last
            global_num_tokens = data.meta_info["global_token_num"]
            estimated_flops, promised_flops = self.flops_counter.estimate_flops(global_num_tokens, delta_time)
            metrics["perf/mfu/actor"] = (
                estimated_flops * self.config.actor.ppo_epochs / promised_flops / self.world_size
            )
            metrics["perf/max_memory_allocated_gb"] = get_torch_device().max_memory_allocated() / (1024**3)
            metrics["perf/max_memory_reserved_gb"] = get_torch_device().max_memory_reserved() / (1024**3)
            metrics["perf/cpu_memory_used_gb"] = psutil.virtual_memory().used / (1024**3)

            lr = self.actor_lr_scheduler.get_last_lr()[0]
            metrics["actor/lr"] = lr.item() if torch.is_tensor(lr) else lr
            self.actor_lr_scheduler.step()

            # TODO: here, we should return all metrics
            output = DataProto(meta_info={"metrics": metrics})

            output = output.to("cpu")

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)
            log_gpu_memory_usage("After offload actor model during update_actor", logger=logger)
        if self._is_offload_optimizer:
            offload_fsdp_optimizer(optimizer=self.actor_optimizer)
            log_gpu_memory_usage("After offload actor optimizer during update_actor", logger=logger)

        return output

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="rollout"))
    @DistProfiler.annotate(color="red", role="rollout_generate")
    def generate_sequences(self, prompts: DataProto):
        # Support all hardwares
        assert self._is_rollout
        prompts = prompts.to(get_device_id())

        meta_info = {
            "eos_token_id": self.generation_config.eos_token_id
            if self.generation_config is not None
            else self.tokenizer.eos_token_id,
            "pad_token_id": self.generation_config.pad_token_id
            if self.generation_config is not None
            else self.tokenizer.pad_token_id,
        }
        prompts.meta_info.update(meta_info)

        if self.rank == 0:
            with contextlib.suppress(Exception):
                sample_index = None
                raw_prompt = None
                raw_prompt_ids = None
                if "index" in prompts.non_tensor_batch:
                    sample_index = int(prompts.non_tensor_batch["index"][0])
                if "raw_prompt" in prompts.non_tensor_batch:
                    raw_prompt = prompts.non_tensor_batch["raw_prompt"][0]
                if "raw_prompt_ids" in prompts.non_tensor_batch:
                    raw_prompt_ids = prompts.non_tensor_batch["raw_prompt_ids"][0]
                    if isinstance(raw_prompt_ids, np.ndarray):
                        raw_prompt_ids = raw_prompt_ids.tolist()
                prompt_ids = prompts.batch["input_ids"][0].detach().cpu().tolist()
                prompt_mask = prompts.batch["attention_mask"][0].detach().cpu().tolist()
                valid_prompt_ids = [token_id for token_id, mask in zip(prompt_ids, prompt_mask, strict=False) if mask]
                debug_path = _write_rollout_policy_debug(
                    "rollout_input",
                    {
                        "experiment_name": str(getattr(self, "experiment_name", "unknown_exp")),
                        "rank": int(self.rank),
                        "sample_index": sample_index,
                        "prompt_hash": _prompt_hash(valid_prompt_ids),
                        "prompt_len": int(len(valid_prompt_ids)),
                        "prompt_ids_head": list(valid_prompt_ids[:128]),
                        "raw_prompt_ids_len": int(len(raw_prompt_ids)) if raw_prompt_ids is not None else None,
                        "raw_prompt_ids_head": list(raw_prompt_ids[:128]) if raw_prompt_ids is not None else None,
                        "raw_prompt_preview": _message_preview(raw_prompt),
                    },
                )
                print(f"[DEBUG rollout input] wrote {debug_path}")

        timing_generate = {}
        if self._is_actor:  # For rollout only, we do not switch context.
            loop = get_event_loop()
            loop.run_until_complete(self.rollout_mode())
            log_gpu_memory_usage("After switch to rollout mode", logger=logger)

        with simple_timer("generate_sequences", timing_generate):
            output = self.rollout.generate_sequences(prompts=prompts)

        if self._is_actor:
            loop.run_until_complete(self.trainer_mode())
            log_gpu_memory_usage("After switch to trainer mode", logger=logger)

        # We calculate the average timing across all ranks
        # to make sure meta_info["timing"] is the same
        timing_generate_topk_ratio, timing_generate_min, timing_generate_max = topk_reduce_ratio_min_max(
            timing_generate["generate_sequences"]
        )
        timing_generate = reduce_timing(timing_generate)
        timing_generate.update(
            {
                "generation_timing/max": timing_generate_max,
                "generation_timing/min": timing_generate_min,
                "generation_timing/topk_ratio": timing_generate_topk_ratio,
            }
        )
        output.meta_info["timing"] = timing_generate
        output = output.to("cpu")

        # clear kv cache
        get_torch_device().empty_cache()
        return output

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    @DistProfiler.annotate(color="blue", role="actor_compute_log_prob")
    def compute_log_prob(self, data: DataProto):
        # when is_lora is True, we use the actor without lora applied to calculate the log_prob
        # which is mostly used for ref log_prob calculation
        assert self._is_actor
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)

        # Support all hardwares
        from contextlib import nullcontext

        is_lora = data.meta_info.pop("is_lora", False)
        adapter_ctx = self.actor.actor_module.disable_adapter() if is_lora else nullcontext()
        peft_model = getattr(self.actor, "actor_module", None)
        if self.rank == 0:
            with contextlib.suppress(Exception):
                debug_path = _write_rollout_policy_debug(
                    "actor_compute_log_prob",
                    {
                        "experiment_name": str(getattr(self, "experiment_name", "unknown_exp")),
                        "rank": int(self.rank),
                        **_actor_adapter_state_payload(peft_model, {"is_lora_flag": bool(is_lora)}),
                    },
                )
                print(f"[DEBUG actor compute_log_prob policy] wrote {debug_path}")
        # we should always recompute old_log_probs when it is HybridEngine
        data.meta_info["micro_batch_size"] = self.config.rollout.log_prob_micro_batch_size_per_gpu
        data.meta_info["max_token_len"] = self.config.rollout.log_prob_max_token_len_per_gpu
        data.meta_info["use_dynamic_bsz"] = self.config.rollout.log_prob_use_dynamic_bsz
        data.meta_info["temperature"] = self.config.rollout.temperature
        # perform recompute log_prob
        with self.ulysses_sharding_manager:
            with adapter_ctx:
                output, entropys = self.actor.compute_log_prob(data=data, calculate_entropy=True)
            output = DataProto.from_dict(
                tensors={"old_log_probs": output, "entropys": entropys},
                meta_info={"temperature": self.config.rollout.temperature},
            )

        output = output.to("cpu")

        # https://pytorch.org/docs/stable/notes/fsdp.html#fsdp-notes
        # unshard the root FSDP module
        if self.world_size > 1 and fsdp_version(self.actor.actor_module) == 1:
            self.actor.actor_module._handle.reshard(True)

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)
            log_gpu_memory_usage("After offload actor model during compute_log_prob", logger=logger)

        return output

    def _disable_lora_dropout(self, model) -> int:
        disabled = 0
        for module in model.modules():
            lora_dropout = getattr(module, "lora_dropout", None)
            if lora_dropout is None:
                continue
            if isinstance(lora_dropout, dict):
                values = lora_dropout.values()
            else:
                values = getattr(lora_dropout, "values", lambda: [])()
            for dropout_layer in values:
                if hasattr(dropout_layer, "p") and dropout_layer.p != 0.0:
                    dropout_layer.p = 0.0
                    disabled += 1
        return disabled

    def _maybe_freeze_lora_A(self, model) -> None:
        if not self.config.model.get("freeze_lora_A", False):
            return
        frozen = 0
        for name, param in model.named_parameters():
            if "lora_A" in name:
                param.requires_grad = False
                frozen += 1
        if self.rank == 0:
            print(f"[LoRA] Frozen lora_A parameters: {frozen}")

    def _collect_svd_init_pretrained_rollout_params(self, peft_config) -> dict[str, torch.Tensor]:
        from transformers import AutoModelForCausalLM

        pretrained_path = getattr(peft_config, "base_model_name_or_path", None) if peft_config is not None else None
        if not pretrained_path:
            raise ValueError("SVD-init rollout sync requires peft_config.base_model_name_or_path")

        local_pretrained_path = copy_to_local(pretrained_path, use_shm=self.config.model.get("use_shm", False))
        if self.rank == 0:
            logger.info(
                "Rollout first sync for SVD-init adapters now loads the original pretrained base model from %s",
                local_pretrained_path,
            )

        pretrained_model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=local_pretrained_path,
            torch_dtype=torch.float32,
            trust_remote_code=self.config.model.get("trust_remote_code", False),
            device_map="cpu",
        )
        try:
            return {name: param.detach().cpu() for name, param in pretrained_model.state_dict().items()}
        finally:
            del pretrained_model

    @contextmanager
    def _svd_init_adapter_ctx(self):
        """Temporarily switch to the frozen initial PiSSA/MiLoRA adapter.

        For SVD-init adapters the residual base stores W_res, so correct base-model KL
        needs W_pretrained = W_res + delta_init. A dedicated frozen adapter gives that
        behavior without triggering FSDP summon_full_params/all-gathers mid-step.
        """
        peft_model = getattr(self.actor, "actor_module", None)
        adapter_name = self._svd_ref_adapter_name

        if peft_model is None or adapter_name is None:
            if self.rank == 0:
                logger.warning(
                    "[SVD-init ref] Frozen reference adapter missing: falling back to current adapter. "
                    "Base-model KL may be inaccurate."
                )
            yield
            return

        previous_adapter = peft_model.active_adapters[0] if peft_model.active_adapters else "default"
        peft_model.base_model.set_adapter(adapter_name, inference_mode=True)
        peft_model.active_adapter = adapter_name
        try:
            yield
        finally:
            peft_model.base_model.set_adapter(previous_adapter, inference_mode=False)
            peft_model.active_adapter = previous_adapter

    def _compute_base_log_prob_from_actor(self, data: DataProto, output_key: str) -> DataProto:
        if self._is_lora:
            if self._is_svd_init:
                # SVD-init (PiSSA/MiLoRA): base = W_pretrained = W_res + A_init @ B_init.
                with self._svd_init_adapter_ctx():
                    data.meta_info["micro_batch_size"] = self.config.rollout.log_prob_micro_batch_size_per_gpu
                    data.meta_info["max_token_len"] = self.config.rollout.log_prob_max_token_len_per_gpu
                    data.meta_info["use_dynamic_bsz"] = self.config.rollout.log_prob_use_dynamic_bsz
                    data.meta_info["temperature"] = self.config.rollout.temperature
                    if self._is_offload_param:
                        load_fsdp_model_to_gpu(self.actor_module_fsdp)
                    with self.ulysses_sharding_manager:
                        output, _ = self.actor.compute_log_prob(data=data, calculate_entropy=False)
                    if self._is_offload_param:
                        offload_fsdp_model_to_cpu(self.actor_module_fsdp)
                return DataProto.from_dict(tensors={output_key: output.to("cpu")})

            # Standard LoRA/plus/b0: disable_adapter() exposes the underlying base model.
            data.meta_info["is_lora"] = True
            data = self.compute_log_prob(data)
            return DataProto.from_dict(tensors={output_key: data.batch["old_log_probs"]})

        data.meta_info["micro_batch_size"] = self.config.rollout.log_prob_micro_batch_size_per_gpu
        data.meta_info["max_token_len"] = self.config.rollout.log_prob_max_token_len_per_gpu
        data.meta_info["use_dynamic_bsz"] = self.config.rollout.log_prob_use_dynamic_bsz
        data.meta_info["temperature"] = self.config.rollout.temperature
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)
        with self.ulysses_sharding_manager:
            output, _ = self.actor.compute_log_prob(data=data, calculate_entropy=False)
        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)
        return DataProto.from_dict(tensors={output_key: output.to("cpu")})

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    @DistProfiler.annotate(color="olive", role="ref_compute_log_prob")
    def compute_ref_log_prob(self, data: DataProto):
        if self._is_lora:
            return self._compute_base_log_prob_from_actor(data, output_key="ref_log_prob")
        assert self._is_ref
        # else:
        # otherwise, the class have a standalone ref model

        micro_batch_size = self.config.ref.log_prob_micro_batch_size_per_gpu
        data.meta_info["micro_batch_size"] = micro_batch_size
        data.meta_info["temperature"] = self.config.rollout.temperature
        data.meta_info["max_token_len"] = self.config.ref.log_prob_max_token_len_per_gpu
        data.meta_info["use_dynamic_bsz"] = self.config.ref.log_prob_use_dynamic_bsz
        with self.ulysses_sharding_manager:
            data = data.to("cpu")  # data will to device with each micro batch on ref.compute_log_prob
            output, _ = self.ref_policy.compute_log_prob(data=data, calculate_entropy=False)
            output = DataProto.from_dict(tensors={"ref_log_prob": output})

        output = output.to("cpu")

        # https://pytorch.org/docs/stable/notes/fsdp.html#fsdp-notes
        # unshard the root FSDP module
        if self.world_size > 1:
            if fsdp_version(self.ref_policy.actor_module) == 1:
                self.ref_policy.actor_module._handle.reshard(True)
            elif fsdp_version(self.ref_policy.actor_module) == 2:
                self.ref_policy.actor_module.reshard()

        return output

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    @DistProfiler.annotate(color="teal", role="base_ref_compute_log_prob")
    def compute_base_ref_log_prob(self, data: DataProto):
        assert self._is_actor, "base-model log prob can only be computed on actor workers"
        return self._compute_base_log_prob_from_actor(data, output_key="base_ref_log_prob")

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    @DistProfiler.annotate(color="teal", role="expected_svd_base_ref_compute_log_prob")
    def compute_expected_svd_base_ref_log_prob(self, data: DataProto):
        """Compute the independently correct pretrained reference for validation health checks."""
        assert self._is_actor, "expected base-model log prob can only be computed on actor workers"
        if not (self._is_lora and self._is_svd_init):
            return self._compute_base_log_prob_from_actor(data, output_key="expected_base_ref_log_prob")

        data.meta_info["micro_batch_size"] = self.config.rollout.log_prob_micro_batch_size_per_gpu
        data.meta_info["max_token_len"] = self.config.rollout.log_prob_max_token_len_per_gpu
        data.meta_info["use_dynamic_bsz"] = self.config.rollout.log_prob_use_dynamic_bsz
        data.meta_info["temperature"] = self.config.rollout.temperature
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)
        with self._svd_init_adapter_ctx():
            with self.ulysses_sharding_manager:
                output, _ = self.actor.compute_log_prob(data=data, calculate_entropy=False)
        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)
        return DataProto.from_dict(tensors={"expected_base_ref_log_prob": output.to("cpu")})

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(self, local_path, hdfs_path=None, global_step=0, max_ckpt_to_keep=None):
        from verl.utils.logger import log_with_rank

        # only support save and load ckpt for actor
        assert self._is_actor

        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)

        self.checkpoint_manager.save_checkpoint(
            local_path=local_path, hdfs_path=hdfs_path, global_step=global_step, max_ckpt_to_keep=max_ckpt_to_keep
        )
        dist.barrier()

        if self._is_lora and hasattr(getattr(self, "actor_module", self.actor_module_fsdp), "peft_config"):
            lora_save_path = os.path.join(local_path, "lora_adapter")
            peft_model = getattr(self, "actor_module", self.actor_module_fsdp)
            peft_config = {}
            if dist.get_rank() == 0:
                os.makedirs(lora_save_path, exist_ok=True)
                peft_config = asdict(peft_model.peft_config.get("default", {}))
                peft_config["task_type"] = peft_config["task_type"].value
                peft_config["peft_type"] = peft_config["peft_type"].value
                peft_config["target_modules"] = list(peft_config["target_modules"])
            try:
                if fsdp_version(self.actor_module_fsdp) > 0:
                    self.actor_module_fsdp = self.actor_module_fsdp.to(get_device_name())
                    lora_params = layered_summon_lora_params(self.actor_module_fsdp)
                    if dist.get_rank() == 0:
                        save_file(lora_params, os.path.join(lora_save_path, "adapter_model.safetensors"))
                        with open(os.path.join(lora_save_path, "adapter_config.json"), "w", encoding="utf-8") as f:
                            json.dump(peft_config, f, ensure_ascii=False, indent=4)
            except Exception as e:
                log_with_rank(
                    f"Save LoRA Adapter Error ({e})", rank=dist.get_rank(), logger=logger, log_only_rank_0=True
                )

            dist.barrier()
            log_with_rank(
                f"[rank-{self.rank}]: Saved LoRA adapter to: {lora_save_path}",
                rank=dist.get_rank(),
                logger=logger,
                log_only_rank_0=True,
            )

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_checkpoint(self, local_path, hdfs_path=None, del_local_after_load=False):
        assert self._is_actor or (not self._is_actor and self._is_rollout), (
            f"Checkpoint loading is only supported for Actor or standalone Rollout Workers, but got "
            f"{self._is_actor} and {self._is_rollout}"
        )

        # No checkpoint to load, just offload the model and optimizer to CPU
        if local_path is None:
            if self._is_offload_param:
                offload_fsdp_model_to_cpu(self.actor_module_fsdp)
            if self._is_offload_optimizer:
                offload_fsdp_optimizer(self.actor_optimizer)
            return

        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)

        self.checkpoint_manager.load_checkpoint(
            local_path=local_path, hdfs_path=hdfs_path, del_local_after_load=del_local_after_load
        )

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)

        if self._is_offload_optimizer:
            offload_fsdp_optimizer(self.actor_optimizer)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def start_profile(self, **kwargs) -> None:
        """Start profiling for the current rank in the current training step."""
        self.profiler.start(**kwargs)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def stop_profile(self) -> None:
        """Stop profiling for the current rank in the current training step."""
        self.profiler.stop()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def dump_memory_snapshot(self, tag: str = "manual", sub_dir: str = None) -> None:
        """Manually trigger a CUDA memory snapshot dump on all ranks."""
        # Memory snapshot is now handled by the profiler system
        # This method is kept for backward compatibility but delegates to profiler
        if hasattr(self, "profiler") and hasattr(self.profiler, "_impl"):
            try:
                # Try to use the profiler's memory snapshot functionality
                if hasattr(self.profiler._impl, "sampler"):
                    out_dir = OmegaConf.select(self.config, "actor.profiler.save_path") or "."
                    self.profiler._impl.sampler.dump_memory_snapshot(out_dir=out_dir, tag=tag, sub_dir=sub_dir)
            except Exception:
                # silently ignore if profiler doesn't support memory snapshots
                pass


class CriticWorker(Worker, DistProfilerExtension):
    def __init__(self, config: FSDPCriticConfig):
        Worker.__init__(self)
        omega_profiler_config = config.get("profiler", {})
        profiler_config = omega_conf_to_dataclass(omega_profiler_config, dataclass_type=ProfilerConfig)
        if omega_profiler_config.get("tool", None) in ["npu", "nsys", "torch", "torch_memory"]:
            tool_config = omega_conf_to_dataclass(
                omega_profiler_config.get("tool_config", {}).get(omega_profiler_config.get("tool"))
            )
        else:
            tool_config = None
        DistProfilerExtension.__init__(
            self, DistProfiler(rank=self.rank, config=profiler_config, tool_config=tool_config)
        )
        import torch.distributed

        self.config = config
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(
                backend=get_nccl_backend(),
                timeout=datetime.timedelta(seconds=self.config.get("nccl_timeout", 600)),
                init_method=os.environ.get("DIST_INIT_METHOD", None),
            )
        self.config: FSDPCriticConfig = config

        # build device mesh for Ulysses Sequence Parallel
        world_size = torch.distributed.get_world_size()
        from torch.distributed.device_mesh import init_device_mesh

        fsdp_size = self.config.model.fsdp_config.fsdp_size
        self.device_mesh = create_device_mesh(world_size=world_size, fsdp_size=fsdp_size)

        self.ulysses_device_mesh = None
        self.ulysses_sequence_parallel_size = self.config.get("ulysses_sequence_parallel_size", 1)
        dp = world_size // self.ulysses_sequence_parallel_size
        if self.ulysses_sequence_parallel_size > 1:
            self.ulysses_device_mesh = init_device_mesh(
                device_name, mesh_shape=(dp, self.ulysses_sequence_parallel_size), mesh_dim_names=["dp", "sp"]
            )

        # create training dispatch
        if self.ulysses_device_mesh is not None:
            is_collect = self.ulysses_device_mesh["sp"].get_local_rank() == 0
            self._register_dispatch_collect_info(
                "critic", dp_rank=self.ulysses_device_mesh["dp"].get_local_rank(), is_collect=is_collect
            )
        else:
            self._register_dispatch_collect_info("critic", dp_rank=self.rank, is_collect=True)

        self.ulysses_sharding_manager = FSDPUlyssesShardingManager(self.ulysses_device_mesh)

        # set FSDP offload params
        self._is_offload_param = self.config.model.fsdp_config.param_offload
        self._is_offload_optimizer = self.config.model.fsdp_config.optimizer_offload

        # normalize config
        self.config.ppo_mini_batch_size *= self.config.rollout_n
        self.config.ppo_mini_batch_size //= torch.distributed.get_world_size() // self.ulysses_sequence_parallel_size
        if self.config.ppo_micro_batch_size is not None:
            self.config.ppo_micro_batch_size //= (
                torch.distributed.get_world_size() // self.ulysses_sequence_parallel_size
            )
            self.config.forward_micro_batch_size //= (
                torch.distributed.get_world_size() // self.ulysses_sequence_parallel_size
            )
            self.config.ppo_micro_batch_size_per_gpu = self.config.ppo_micro_batch_size
            self.config.forward_micro_batch_size_per_gpu = self.config.forward_micro_batch_size

        if self.config.ppo_micro_batch_size_per_gpu is not None:
            assert self.config.ppo_mini_batch_size % self.config.ppo_micro_batch_size_per_gpu == 0, (
                f"normalized ppo_mini_batch_size {self.config.ppo_mini_batch_size} should be divisible by "
                f"ppo_micro_batch_size_per_gpu {self.config.ppo_micro_batch_size_per_gpu}"
            )
            assert self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu > 0, (
                f"normalized ppo_mini_batch_size {self.config.ppo_mini_batch_size} should be larger than "
                f"ppo_micro_batch_size_per_gpu {self.config.ppo_micro_batch_size_per_gpu}"
            )
        self._is_lora = (
            self.config.model.get("lora_adapter_path") is not None or self.config.model.get("lora_rank", 0) > 0
        )
        self.use_orig_params = self.config.model.fsdp_config.get("use_orig_params", False)

    def _build_critic_model_optimizer(self, config):
        # the following line is necessary
        from torch.distributed.fsdp import MixedPrecision

        from verl.utils.model import load_valuehead_model, print_model_size
        from verl.utils.torch_dtypes import PrecisionType

        use_shm = config.model.get("use_shm", False)
        local_path = copy_to_local(config.model.path, use_shm=use_shm)
        # note that the tokenizer between actor and critic may be different. So override tokenizer info with actor info
        # using random initialized model from any architecture. May not be the same as Actor.

        tokenizer_path = copy_to_local(config.model.tokenizer_path, use_shm=use_shm)
        self.tokenizer = hf_tokenizer(tokenizer_path, trust_remote_code=config.model.get("trust_remote_code", False))
        self.processor = hf_processor(tokenizer_path, trust_remote_code=config.model.get("trust_remote_code", False))

        if self.config.model.get("custom_chat_template", None) is not None:
            if self.processor is not None:
                self.processor.chat_template = self.config.model.custom_chat_template
            else:
                self.tokenizer.chat_template = self.config.model.custom_chat_template
        override_config = OmegaConf.to_container(OmegaConf.create(self.config.model.get("override_config", {})))
        override_config_kwargs = {
            "bos_token_id": self.tokenizer.bos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        override_config_kwargs.update(override_config)
        if self.rank == 0:
            print(f"Critic overriding config {override_config_kwargs}")

        torch_dtype = self.config.model.fsdp_config.get("model_dtype", "fp32")
        torch_dtype = PrecisionType.to_dtype(torch_dtype)

        from transformers import AutoConfig

        # override model kwargs
        attn_implementation = override_config.get("attn_implementation", "flash_attention_2")
        critic_model_config = AutoConfig.from_pretrained(
            local_path,
            attn_implementation=attn_implementation,
            trust_remote_code=config.model.get("trust_remote_code", False),
        )
        # TODO: VL models use VisionAttention, which directly uses flash_attention in transformers>=4.53
        # which will be patched by _ulysses_flash_attention_forward, but errorly misses position_ids
        # Maybe support Ulysses in VisionAttention in the future and remove this patch
        if self.ulysses_sequence_parallel_size > 1 and hasattr(critic_model_config, "vision_config"):
            critic_model_config.vision_config._attn_implementation = "eager"

        critic_model_config.num_labels = 1
        # patch for kimi-vl
        if getattr(critic_model_config, "model_type", None) == "kimi_vl":
            critic_model_config.text_config.topk_method = "greedy"

        init_context = get_init_weight_context_manager(
            use_meta_tensor=not critic_model_config.tie_word_embeddings, mesh=self.device_mesh
        )

        with init_context(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            critic_model_config.classifier_dropout = 0.0
            critic_model_config.hidden_dropout = "0"
            critic_model_config.summary_dropout_prob = 0.0

            critic_module = load_valuehead_model(
                local_path,
                torch_dtype,
                critic_model_config,
                config.model.get("trust_remote_code", False),
            )

            use_remove_padding = config.model.get("use_remove_padding", False)

            apply_monkey_patch(
                model=critic_module,
                use_remove_padding=use_remove_padding,
                ulysses_sp_size=self.ulysses_sequence_parallel_size,
            )

            # some parameters may not in torch_dtype
            critic_module.to(torch_dtype)

            if config.model.get("enable_gradient_checkpointing", False):
                critic_module.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

        if self._is_lora:
            print("Applying LoRA to critic module")
            critic_module.enable_input_require_grads()

            # Check if we should load a pre-trained LoRA adapter
            lora_adapter_path = self.config.model.get("lora_adapter_path")
            if lora_adapter_path is not None:
                from peft import PeftModel

                print(f"Loading pre-trained LoRA adapter to critic from: {lora_adapter_path}")

                # Copy adapter to local if needed
                local_adapter_path = copy_to_local(lora_adapter_path, use_shm=self.config.model.get("use_shm", False))

                critic_module = PeftModel.from_pretrained(
                    critic_module,
                    local_adapter_path,
                    is_trainable=True,
                    low_cpu_mem_usage=True,
                )
                peft_config = critic_module.peft_config["default"]
                # Ensure task_type is TaskType enum, not string
                if isinstance(peft_config.task_type, str):
                    peft_config.task_type = TaskType.CAUSAL_LM
                if self.config.model.get("use_adalora", False):
                    critic_module.to(torch_dtype)

            else:
                # Convert config to regular Python types before creating PEFT model
                lora_config = {
                    "task_type": TaskType.CAUSAL_LM,
                    "r": self.config.model.lora_rank,
                    "lora_alpha": self.config.model.lora_alpha,
                    "target_modules": convert_to_regular_types(self.config.model.target_modules),
                    "bias": "none",
                }
                if self.config.model.get("use_adalora", False):
                    adalora_config = {
                        "task_type": TaskType.CAUSAL_LM,
                        "init_r": self.config.model.get("adalora_init_r") or self.config.model.lora_rank,
                        "target_r": self.config.model.get("adalora_target_r") or self.config.model.lora_rank,
                        "lora_alpha": self.config.model.lora_alpha,
                        "target_modules": convert_to_regular_types(self.config.model.target_modules),
                        "beta1": self.config.model.get("adalora_beta1", 0.85),
                        "beta2": self.config.model.get("adalora_beta2", 0.85),
                        "tinit": self.config.model.get("adalora_tinit", 0),
                        "tfinal": self.config.model.get("adalora_tfinal", 0),
                        "deltaT": self.config.model.get("adalora_delta_t", 1),
                        "total_step": self.config.critic.optim.get("total_training_steps", 0),
                        "orth_reg_weight": self.config.model.get("adalora_orth_reg_weight", 0.5),
                    }
                    critic_module = get_peft_model(critic_module, AdaLoraConfig(**adalora_config))
                    critic_module.to(torch_dtype)
                else:
                    critic_module = get_peft_model(critic_module, LoraConfig(**lora_config))
                init_count = _init_lora_a_params(
                    critic_module,
                    init_method=self.config.model.get("lora_A_init_method", "default"),
                    wavelet=self.config.model.get("lora_A_wavelet", "haar"),
                )
                if self.rank == 0 and init_count > 0:
                    init_desc = f"method={self.config.model.get('lora_A_init_method')}"
                    if self.config.model.get("lora_A_init_method") == "wavelet_orth":
                        init_desc += f" wavelet={self.config.model.get('lora_A_wavelet', 'haar')}"
                    print(f"Applied LoRA-A init {init_desc} to {init_count} critic matrices.")

        if self.rank == 0:
            print_model_size(critic_module)

        self.critic_model_config = critic_model_config

        fsdp_config = self.config.model.fsdp_config
        mixed_precision_config = fsdp_config.get("mixed_precision", None)
        if mixed_precision_config is not None:
            param_dtype = PrecisionType.to_dtype(mixed_precision_config.get("param_dtype", "bf16"))
            reduce_dtype = PrecisionType.to_dtype(mixed_precision_config.get("reduce_dtype", "fp32"))
            buffer_dtype = PrecisionType.to_dtype(mixed_precision_config.get("buffer_dtype", "fp32"))
        else:
            param_dtype = torch.bfloat16
            reduce_dtype = torch.float32
            buffer_dtype = torch.float32

        mixed_precision = MixedPrecision(param_dtype=param_dtype, reduce_dtype=reduce_dtype, buffer_dtype=buffer_dtype)

        auto_wrap_policy = get_fsdp_wrap_policy(
            module=critic_module,
            config=self.config.model.fsdp_config.wrap_policy,
            is_lora=self._is_lora
            and not self.config.model.get("use_dora", False)
            and not self.config.model.get("use_adalora", False),
        )

        log_gpu_memory_usage("Before critic FSDP", logger=None)

        fsdp_mesh = self.device_mesh
        sharding_strategy = get_sharding_strategy(fsdp_mesh)

        self.use_orig_params = fsdp_config.get("use_orig_params", False)
        if self.config.model.get("freeze_vision_tower", False):
            vision_tower = get_vl_model_vision_tower(critic_module)
            if vision_tower is not None:
                vision_tower.requires_grad_(False)
                self.use_orig_params = True
                if self.rank == 0:
                    print("[critic model] Vision tower is set to not trainable.")
            else:
                if self.rank == 0:
                    print("[critic model] No vision tower found.")

        # Note: We force turn off CPUOffload for critic because it causes incorrect results when using grad accumulation
        if config.strategy == "fsdp":
            critic_module = FSDP(
                critic_module,
                param_init_fn=init_fn,
                use_orig_params=self.use_orig_params,
                auto_wrap_policy=auto_wrap_policy,
                device_id=get_device_id(),
                sharding_strategy=sharding_strategy,
                mixed_precision=mixed_precision,
                sync_module_states=True,
                forward_prefetch=self.config.model.fsdp_config.forward_prefetch,
                device_mesh=self.device_mesh,
                cpu_offload=None,
            )
        elif config.strategy == "fsdp2":
            assert CPUOffloadPolicy is not None, "PyTorch version >= 2.4 is required for using fully_shard API (FSDP2)"
            mp_policy = MixedPrecisionPolicy(
                param_dtype=param_dtype, reduce_dtype=reduce_dtype, cast_forward_inputs=True
            )
            offload_policy = None
            if fsdp_config.offload_policy:
                self._is_offload_param = False
                self._is_offload_optimizer = False
                offload_policy = CPUOffloadPolicy(pin_memory=True)

            fsdp_kwargs = {
                "mesh": fsdp_mesh,
                "mp_policy": mp_policy,
                "offload_policy": offload_policy,
                "reshard_after_forward": fsdp_config.reshard_after_forward,
                "shard_placement_fn": get_shard_placement_fn(fsdp_size=self.device_mesh.shape[-1]),
            }
            full_state = critic_module.state_dict()
            apply_fsdp2(critic_module, fsdp_kwargs, fsdp_config)
            fsdp2_load_full_state_dict(critic_module, full_state, fsdp_mesh, offload_policy)
        else:
            raise NotImplementedError(f"Unknown strategy {config.strategy}")

        if config.model.get("enable_activation_offload", False):
            enable_gradient_checkpointing = config.model.get("enable_gradient_checkpointing", False)
            enable_activation_offloading(critic_module, config.strategy, enable_gradient_checkpointing)

        log_gpu_memory_usage("After critic FSDP", logger=None)

        critic_optimizer = build_optimizer(critic_module.parameters(), config.optim)

        total_steps = config.optim.get("total_training_steps", 0)
        num_warmup_steps = int(config.optim.get("lr_warmup_steps", -1))

        lr_scheduler_type = config.optim.get("lr_scheduler_type", "constant")
        if num_warmup_steps < 0:
            num_warmup_steps_ratio = config.optim.get("lr_warmup_steps_ratio", 0.0)
            num_warmup_steps = int(num_warmup_steps_ratio * total_steps)

        if self.rank == 0:
            print(f"Total steps: {total_steps}, num_warmup_steps: {num_warmup_steps}")

        from verl.utils.torch_functional import get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup

        if lr_scheduler_type == "constant":
            critic_lr_scheduler = get_constant_schedule_with_warmup(
                optimizer=critic_optimizer, num_warmup_steps=num_warmup_steps
            )
        elif lr_scheduler_type == "cosine":
            min_lr_ratio = config.optim.get("min_lr_ratio", 0.0)
            num_cycles = config.optim.get("num_cycles", 0.5)
            critic_lr_scheduler = get_cosine_schedule_with_warmup(
                optimizer=critic_optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=total_steps,
                min_lr_ratio=min_lr_ratio,
                num_cycles=num_cycles,
            )
        else:
            raise NotImplementedError(f"LR scheduler type {lr_scheduler_type} is not supported")

        return critic_module, critic_optimizer, critic_lr_scheduler

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        # This is used to import external_lib into the huggingface systems
        import_external_libs(self.config.model.get("external_lib", None))

        from verl.workers.critic import DataParallelPPOCritic

        self.critic_module, self.critic_optimizer, self.critic_lr_scheduler = self._build_critic_model_optimizer(
            self.config
        )

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.critic_module)
            log_gpu_memory_usage("After offload critic model during init", logger=logger)
        if self._is_offload_optimizer:
            offload_fsdp_optimizer(optimizer=self.critic_optimizer)
            log_gpu_memory_usage("After offload critic optimizer during init", logger=logger)

        self.critic = DataParallelPPOCritic(
            config=self.config, critic_module=self.critic_module, critic_optimizer=self.critic_optimizer
        )

        self.flops_counter = FlopsCounter(self.critic_model_config)
        self.checkpoint_manager = FSDPCheckpointManager(
            model=self.critic_module,
            optimizer=self.critic_optimizer,
            lr_scheduler=self.critic_lr_scheduler,
            processing_class=self.processor if self.processor is not None else self.tokenizer,
            checkpoint_config=self.config.checkpoint,
        )

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="critic"))
    @DistProfiler.annotate(color="cyan")
    def compute_values(self, data: DataProto):
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.critic_module)
        micro_batch_size = self.config.forward_micro_batch_size_per_gpu
        data.meta_info["micro_batch_size"] = micro_batch_size
        data.meta_info["max_token_len"] = self.config.forward_max_token_len_per_gpu
        data.meta_info["use_dynamic_bsz"] = self.config.use_dynamic_bsz
        # perform forward computation
        with self.ulysses_sharding_manager:
            data = data.to("cpu")  # data will to device with each micro batch on critic.compute_values
            values = self.critic.compute_values(data=data)
            output = DataProto.from_dict(tensors={"values": values})

        output = output.to("cpu")
        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.critic_module)
        return output

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="critic"))
    @DistProfiler.annotate(color="pink")
    def update_critic(self, data: DataProto):
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.critic_module)
        if self._is_offload_optimizer:
            load_fsdp_optimizer(optimizer=self.critic_optimizer, device_id=get_device_id())

        # perform forward computation
        with self.ulysses_sharding_manager:
            data = data.to("cpu")  # data will to device with each micro batch on critic.update_critic
            with Timer(name="update_critic", logger=None) as timer:
                metrics = self.critic.update_critic(data=data)
            delta_time = timer.last

            global_num_tokens = data.meta_info["global_token_num"]
            estimated_flops, promised_flops = self.flops_counter.estimate_flops(global_num_tokens, delta_time)
            metrics["perf/mfu/critic"] = estimated_flops * self.config.ppo_epochs / promised_flops / self.world_size

            lr = self.critic_lr_scheduler.get_last_lr()[0]
            metrics["critic/lr"] = lr
            self.critic_lr_scheduler.step()

            output = DataProto(batch=None, meta_info={"metrics": metrics})

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.critic_module)
        if self._is_offload_optimizer:
            offload_fsdp_optimizer(optimizer=self.critic_optimizer)

        output = output.to("cpu")
        return output

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(self, local_path, hdfs_path=None, global_step=0, max_ckpt_to_keep=None):
        import torch

        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.critic_module)

        self.checkpoint_manager.save_checkpoint(
            local_path=local_path, hdfs_path=hdfs_path, global_step=global_step, max_ckpt_to_keep=max_ckpt_to_keep
        )

        torch.distributed.barrier()
        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.critic_module)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_checkpoint(self, local_path, hdfs_path=None, del_local_after_load=True):
        import torch

        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.critic_module)

        self.checkpoint_manager.load_checkpoint(
            local_path=local_path, hdfs_path=hdfs_path, del_local_after_load=del_local_after_load
        )

        torch.distributed.barrier()
        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.critic_module)

        if self._is_offload_optimizer:
            offload_fsdp_optimizer(self.critic_optimizer)


# TODO(sgm): we may need to extract it to dp_reward_model.py
class RewardModelWorker(Worker, DistProfilerExtension):
    """
    Note that we only implement the reward model that is subclass of AutoModelForTokenClassification.
    """

    def __init__(self, config):
        Worker.__init__(self)

        omega_profiler_config = config.get("profiler", {})
        profiler_config = omega_conf_to_dataclass(omega_profiler_config, dataclass_type=ProfilerConfig)
        if omega_profiler_config.get("tool", None) in ["npu", "nsys", "torch", "torch_memory"]:
            tool_config = omega_conf_to_dataclass(
                omega_profiler_config.get("tool_config", {}).get(omega_profiler_config.get("tool"))
            )
        else:
            tool_config = None
        DistProfilerExtension.__init__(
            self,
            DistProfiler(rank=self.rank, config=profiler_config, tool_config=tool_config),
        )

        import torch.distributed

        self.config = config
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(
                backend=get_nccl_backend(),
                timeout=datetime.timedelta(seconds=self.config.get("nccl_timeout", 600)),
                init_method=os.environ.get("DIST_INIT_METHOD", None),
            )

        # build device mesh for Ulysses Sequence Parallel
        world_size = torch.distributed.get_world_size()
        from torch.distributed.device_mesh import init_device_mesh

        fsdp_size = self.config.model.fsdp_config.fsdp_size
        self.device_mesh = create_device_mesh(world_size=world_size, fsdp_size=fsdp_size)

        self.ulysses_device_mesh = None
        self.ulysses_sequence_parallel_size = self.config.get("ulysses_sequence_parallel_size", 1)
        dp = world_size // self.ulysses_sequence_parallel_size
        if self.ulysses_sequence_parallel_size > 1:
            self.ulysses_device_mesh = init_device_mesh(
                device_name, mesh_shape=(dp, self.ulysses_sequence_parallel_size), mesh_dim_names=["dp", "sp"]
            )

        self.ulysses_sharding_manager = FSDPUlyssesShardingManager(self.ulysses_device_mesh)

        # create training dispatch
        if self.ulysses_device_mesh is not None:
            is_collect = self.ulysses_device_mesh["sp"].get_local_rank() == 0
            self._register_dispatch_collect_info(
                "reward", dp_rank=self.ulysses_device_mesh["dp"].get_local_rank(), is_collect=is_collect
            )
        else:
            self._register_dispatch_collect_info("reward", dp_rank=self.rank, is_collect=True)

        self.use_remove_padding = self.config.model.get("use_remove_padding", False)

        # normalize config
        if self.config.micro_batch_size is not None:
            self.config.micro_batch_size //= torch.distributed.get_world_size()
            self.config.micro_batch_size_per_gpu = self.config.micro_batch_size

    def _build_model(self, config):
        # the following line is necessary
        from torch.distributed.fsdp import CPUOffload
        from transformers import AutoConfig, AutoModelForTokenClassification

        use_shm = config.model.get("use_shm", False)
        # download the checkpoint from hdfs
        local_path = copy_to_local(config.model.path, use_shm=use_shm)

        if self.config.model.input_tokenizer is None:
            self._do_switch_chat_template = False
        else:
            self._do_switch_chat_template = True
            input_tokenizer_local_path = copy_to_local(config.model.input_tokenizer, use_shm=use_shm)
            self.input_tokenizer = hf_tokenizer(
                input_tokenizer_local_path, trust_remote_code=config.model.get("trust_remote_code", False)
            )
            self.tokenizer = hf_tokenizer(local_path, trust_remote_code=config.model.get("trust_remote_code", False))

        trust_remote_code = config.model.get("trust_remote_code", False)
        model_config = AutoConfig.from_pretrained(local_path, trust_remote_code=trust_remote_code)
        model_config.num_labels = 1

        # note that we have to create model in fp32. Otherwise, the optimizer is in bf16, which is incorrect
        init_context = get_init_weight_context_manager(
            use_meta_tensor=not model_config.tie_word_embeddings, mesh=self.device_mesh
        )

        with init_context(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model_config.classifier_dropout = 0.0
            reward_module = AutoModelForTokenClassification.from_pretrained(
                pretrained_model_name_or_path=local_path,
                config=model_config,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                trust_remote_code=trust_remote_code,
            )

            apply_monkey_patch(
                model=reward_module,
                use_remove_padding=config.model.get("use_remove_padding", False),
                ulysses_sp_size=self.ulysses_sequence_parallel_size,
            )

            reward_module.to(torch.bfloat16)

        auto_wrap_policy = get_fsdp_wrap_policy(module=reward_module, config=self.config.model.fsdp_config)

        fsdp_mesh = self.device_mesh
        sharding_strategy = get_sharding_strategy(fsdp_mesh)

        if config.strategy == "fsdp":
            reward_module = FSDP(
                reward_module,
                param_init_fn=init_fn,
                use_orig_params=False,
                auto_wrap_policy=auto_wrap_policy,
                device_id=get_device_id(),
                sharding_strategy=sharding_strategy,  # zero3
                sync_module_states=True,
                cpu_offload=CPUOffload(offload_params=True),
                forward_prefetch=self.config.model.fsdp_config.forward_prefetch,
                device_mesh=self.device_mesh,
            )
        elif config.strategy == "fsdp2":
            assert CPUOffloadPolicy is not None, "PyTorch version >= 2.4 is required for using fully_shard API (FSDP2)"
            cpu_offload = CPUOffloadPolicy(pin_memory=True)
            fsdp_kwargs = {
                "mesh": fsdp_mesh,
                "offload_policy": cpu_offload,
                "reshard_after_forward": config.model.fsdp_config.reshard_after_forward,
                "shard_placement_fn": get_shard_placement_fn(fsdp_size=self.device_mesh.shape[-1]),
            }
            full_state = reward_module.state_dict()
            apply_fsdp2(reward_module, fsdp_kwargs, config.model.fsdp_config)
            fsdp2_load_full_state_dict(reward_module, full_state, fsdp_mesh, cpu_offload)
        else:
            raise NotImplementedError(f"Unknown strategy: {config.strategy}")
        return reward_module

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        # This is used to import external_lib into the huggingface systems
        import_external_libs(self.config.model.get("external_lib", None))
        self.reward_module = self._build_model(config=self.config)

    def _forward_micro_batch(self, micro_batch):
        from verl.utils.attention_utils import index_first_axis, pad_input, rearrange, unpad_input
        from verl.utils.ulysses import gather_outputs_and_unpad, ulysses_pad_and_slice_inputs

        with torch.no_grad(), torch.autocast(device_type=device_name, dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 3, seqlen) -> (3, bsz, seqlen)

            if self.use_remove_padding:
                input_ids_rmpad, indices, *_ = unpad_input(
                    input_ids.unsqueeze(-1), attention_mask
                )  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = (
                        index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                        .transpose(0, 1)
                        .unsqueeze(1)
                    )  # (3, bsz, seqlen) -> (3, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(
                        rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                    ).transpose(0, 1)

                # pad and slice the inputs if sp > 1
                if self.ulysses_sequence_parallel_size > 1:
                    input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad, position_ids_rmpad, sp_size=self.ulysses_sequence_parallel_size
                    )

                # only pass input_ids and position_ids to enable flash_attn_varlen
                output = self.reward_module(
                    input_ids=input_ids_rmpad, attention_mask=None, position_ids=position_ids_rmpad, use_cache=False
                )
                reward_rmpad = output.logits
                reward_rmpad = reward_rmpad.squeeze(0)  # (total_nnz)

                # gather output if sp > 1
                if self.ulysses_sequence_parallel_size > 1:
                    reward_rmpad = gather_outputs_and_unpad(
                        reward_rmpad, gather_dim=0, unpad_dim=0, padding_size=pad_size
                    )

                # pad it back
                rm_score = pad_input(reward_rmpad, indices=indices, batch=batch_size, seqlen=seqlen).squeeze(-1)
            else:
                output = self.reward_module(
                    input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, use_cache=False
                )
                rm_score = output.logits  # (batch_size, seq_len, 1)
                rm_score = rm_score.squeeze(-1)

            # extract the result of the last valid token
            eos_mask_idx = torch.argmax(position_ids * attention_mask, dim=-1)  # (bsz,)
            rm_score = rm_score[torch.arange(batch_size), eos_mask_idx]
            return rm_score

    def _expand_to_token_level(self, data: DataProto, scores: torch.Tensor):
        batch_size = data.batch.batch_size[0]
        # expand as token_level_reward
        attention_mask = data.batch["attention_mask"]
        position_ids = data.batch["position_ids"]
        response_length = data.batch["responses"].shape[-1]
        if position_ids.dim() == 3:  # qwen2vl mrope [bs, 3, seq_len]
            position_ids = position_ids[:, 0, :]
        eos_mask_idx = torch.argmax(position_ids * attention_mask, dim=-1)  # (bsz,)
        token_level_scores = torch.zeros_like(attention_mask, dtype=scores.dtype)  # (bsz, seqlen)
        token_level_scores[torch.arange(batch_size), eos_mask_idx] = scores

        # select the response part
        token_level_scores = token_level_scores[:, -response_length:]

        return token_level_scores

    def _switch_chat_template(self, data: DataProto):
        src_max_length = data.batch["attention_mask"].shape[-1]

        src_tokenizer = self.input_tokenizer
        target_tokenizer = self.tokenizer

        rm_input_ids = []
        rm_attention_mask = []

        for i in range(data.batch.batch_size[0]):
            if not isinstance(data.non_tensor_batch["raw_prompt"][i], list | np.ndarray):
                raise TypeError(
                    f"raw_prompt must be a list or numpy array, got {type(data.non_tensor_batch['raw_prompt'][i])}"
                )

            # extract raw prompt
            chat: list = list(data.non_tensor_batch["raw_prompt"][i])

            # extract response
            response_ids = data.batch["responses"][i]
            response_length = response_ids.shape[-1]
            valid_response_length = data.batch["attention_mask"][i][-response_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            response = src_tokenizer.decode(valid_response_ids)
            # remove bos and eos
            response = response.replace(src_tokenizer.eos_token, "")

            chat.append({"role": "assistant", "content": response})

            prompt_with_chat_template = target_tokenizer.apply_chat_template(
                chat, add_generation_prompt=False, tokenize=False
            )
            if self.rank == 0 and i == 0:
                # for debugging purpose
                print(f"Switch template. chat: {prompt_with_chat_template}")

            # the maximum length is actually determined by the reward model itself
            max_length = self.config.get("max_length", src_max_length)
            if max_length is None:
                max_length = src_max_length

            model_inputs = target_tokenizer(prompt_with_chat_template, return_tensors="pt", add_special_tokens=False)
            input_ids, attention_mask = verl_F.postprocess_data(
                input_ids=model_inputs["input_ids"],
                attention_mask=model_inputs["attention_mask"],
                max_length=max_length,
                pad_token_id=target_tokenizer.pad_token_id,
                left_pad=False,  # right padding
                truncation=self.config.get("truncation", "right"),
            )  # truncate from the right

            rm_input_ids.append(input_ids)
            rm_attention_mask.append(attention_mask)

        rm_input_ids = torch.cat(rm_input_ids, dim=0)
        rm_attention_mask = torch.cat(rm_attention_mask, dim=0)

        rm_position_ids = compute_position_id_with_mask(rm_attention_mask)

        rm_inputs = {"input_ids": rm_input_ids, "attention_mask": rm_attention_mask, "position_ids": rm_position_ids}

        return DataProto.from_dict(rm_inputs)

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="reward"))
    @DistProfiler.annotate(color="brown")
    def compute_rm_score(self, data: DataProto):
        import itertools

        from verl.utils.seqlen_balancing import get_reverse_idx, rearrange_micro_batches

        # Support all hardwares
        data = data.to(get_device_id())
        if self._do_switch_chat_template:
            rm_data = self._switch_chat_template(data)
        else:
            rm_input_ids = data.batch["input_ids"]
            rm_attention_mask = data.batch["attention_mask"]
            rm_position_ids = data.batch["position_ids"]
            rm_inputs = {
                "input_ids": rm_input_ids,
                "attention_mask": rm_attention_mask,
                "position_ids": rm_position_ids,
            }
            rm_data = DataProto.from_dict(rm_inputs)

        # Support all hardwares
        rm_data = rm_data.to(get_device_id())

        # perform forward computation
        with self.ulysses_sharding_manager:
            use_dynamic_bsz = self.config.use_dynamic_bsz
            if use_dynamic_bsz:
                max_token_len = self.config.forward_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                micro_batches, indices = rearrange_micro_batches(batch=rm_data.batch, max_token_len=max_token_len)
            else:
                micro_batches = rm_data.batch.split(self.config.micro_batch_size_per_gpu)
            output = []
            for micro_batch in micro_batches:
                rm_score = self._forward_micro_batch(micro_batch)
                output.append(rm_score)
            scores = torch.cat(output, dim=0)  # (batch_size)

            if use_dynamic_bsz:
                indices = list(itertools.chain.from_iterable(indices))
                assert len(indices) == scores.size(0), f"{len(indices)} vs. {scores.size()}"
                revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
                scores = scores[revert_indices]

            token_level_scores = self._expand_to_token_level(data, scores)
            # Note that this is only the scores, may not be the final rewards used to train RL
            output = DataProto.from_dict(tensors={"rm_scores": token_level_scores})

        # https://pytorch.org/docs/stable/notes/fsdp.html#fsdp-notes
        # unshard the root FSDP module
        if self.world_size > 1 and fsdp_version(self.reward_module) == 1:
            self.reward_module._handle.reshard(True)

        output = output.to("cpu")
        return output


# ================================= Async related workers =================================
class AsyncActorRolloutRefWorker(ActorRolloutRefWorker):
    @register(dispatch_mode=Dispatch.DIRECT_ROLLOUT_METHOD)
    async def wake_up(self):
        await self.rollout_mode()
        return True

    @register(dispatch_mode=Dispatch.DIRECT_ROLLOUT_METHOD)
    async def sleep(self):
        await self.trainer_mode()
        return True

    # ============================ vLLM related ============================

    @register(dispatch_mode=Dispatch.DIRECT_ROLLOUT_METHOD)
    def get_zeromq_address(self):
        return self.rollout.get_zeromq_address()

    # ============================ SGLang related ============================

    @register(dispatch_mode=Dispatch.DIRECT_ROLLOUT_METHOD, blocking=False)
    async def chat_completion(self, json_request):
        ret = await self.rollout.chat_completion(json_request)
        return ret

    @register(dispatch_mode=Dispatch.DIRECT_ROLLOUT_METHOD, blocking=False)
    async def generate(
        self,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        request_id: str,
        image_data: Optional[list[Any]] = None,
    ) -> list[int]:
        ret = await self.rollout.generate(prompt_ids, sampling_params, request_id, image_data=image_data)
        return ret
