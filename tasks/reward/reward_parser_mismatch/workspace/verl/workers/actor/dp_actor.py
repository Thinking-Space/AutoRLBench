# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
Single Process Actor
"""

import logging
import math
import os

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.tensor import DTensor

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, get_policy_loss_fn, kl_penalty
from verl.utils.attention_utils import index_first_axis, pad_input, rearrange, unpad_input
from verl.utils.device import get_device_id, get_device_name
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import prepare_dynamic_batch, restore_dynamic_batch
from verl.utils.torch_dtypes import PrecisionType
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import gather_outputs_and_unpad, ulysses_pad, ulysses_pad_and_slice_inputs
from verl.workers.actor import BasePPOActor
from verl.workers.config import ActorConfig

__all__ = ["DataParallelPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

_VIMPO_TERM_COUNT = 0
_VIMPO_TERM_VALUE_SUM = 1
_VIMPO_TERM_TARGET_SUM = 2
_VIMPO_TERM_REWARD_SUM = 3
_VIMPO_TERM_BASELINE_SUM = 4
_VIMPO_TERM_ERROR_SQ_SUM = 5
_VIMPO_TERM_ABS_ERROR_SUM = 6
_VIMPO_TERM_TARGET_SQ_SUM = 7
_VIMPO_TERM_HUBER_SUM = 8
_VIMPO_TERM_HUBER_CLIP_SUM = 9
_VIMPO_TERM_NUM_STATS = 10

_VIMPO_ADV_COUNT = 0
_VIMPO_ADV_SUM = 1
_VIMPO_ADV_SQ_SUM = 2
_VIMPO_ADV_RAW_SUM = 3
_VIMPO_ADV_RAW_SQ_SUM = 4
_VIMPO_ADV_RAW_ABS_SUM = 5
_VIMPO_ADV_RAW_POS_SUM = 6
_VIMPO_ADV_VALUE_SUM = 7
_VIMPO_ADV_NUM_STATS = 8

_VIMPO_KL_COUNT = 0
_VIMPO_KL_SUM = 1
_VIMPO_KL_NUM_STATS = 2


def _optional_float(value, name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, str) and value.lower() in {"none", "null"}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"vimpo_config.{name} must be null/None or a float, got {value!r}.") from exc


class DataParallelPPOActor(BasePPOActor):
    """FSDP DataParallel PPO Actor or Ref worker

    Args:
        config (ActorConfig): Actor config
        actor_module (nn.Module): Actor or ref module
        actor_optimizer (torch.optim.Optimizer, optional): Actor optimizer. Defaults to None.
    """

    def __init__(self, config: ActorConfig, actor_module: nn.Module, actor_optimizer: torch.optim.Optimizer = None):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        self.ref_policy: "DataParallelPPOActor | None" = None
        self._use_lora_reference = False
        self._vimpo_adaptive_value_logit_grad_rms_ema = 0.0
        self._vimpo_adaptive_actor_logit_grad_rms_unscaled_ema = 0.0
        role = "Ref" if actor_optimizer is None else "Actor"

        self.use_remove_padding = self.config.get("use_remove_padding", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_remove_padding={self.use_remove_padding}")
        self.use_fused_kernels = self.config.get("use_fused_kernels", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_fused_kernels={self.use_fused_kernels}")

        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        self.use_dynamic_bsz = self.config.get("use_dynamic_bsz", False)

        self.use_prefix_grouper = self.config.get("use_prefix_grouper", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_prefix_grouper={self.use_prefix_grouper}")

        if self.config.entropy_from_logits_with_chunking:
            entropy_from_logits = verl_F.entropy_from_logits_with_chunking
        else:
            entropy_from_logits = verl_F.entropy_from_logits

        self.compute_entropy_from_logits = (
            torch.compile(entropy_from_logits, dynamic=True)
            if self.config.get("use_torch_compile", True)  # use torch compile by default
            else entropy_from_logits
        )
        self.device_name = get_device_name()
        self.param_dtype = PrecisionType.to_dtype(self.config.fsdp_config.get("dtype", "bfloat16"))
        if self.param_dtype == torch.float16:
            from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler

            self.scaler = ShardedGradScaler(growth_interval=400)
        else:
            self.scaler = None

        # Sum of squared probabilities computation (for optimal_token_baseline)
        # Only initialize if calculate_sum_pi_squared config is enabled
        if self.config.get("calculate_sum_pi_squared", False):
            self.calculate_sum_pi_squared_from_logits = (
                torch.compile(verl_F.calculate_sum_pi_squared_from_logits, dynamic=True)
                if self.config.get("use_torch_compile", True)
                else verl_F.calculate_sum_pi_squared_from_logits
            )
            assert not (self.use_fused_kernels or self.use_prefix_grouper), (
                "calculate_sum_pi_squared is not supported with "
                f"{self.use_fused_kernels=} or {self.use_prefix_grouper=} for now."
            )

    def _forward_micro_batch(
        self,
        micro_batch: dict[str, torch.Tensor],
        temperature: float,
        calculate_entropy: bool = False,
        return_valid_response_logits: bool = False,
    ) -> dict[str, torch.Tensor]:
        """
        Returns:
            dict[str, torch.Tensor]:
                log_probs: (bs, response_len)
                if calculate_entropy is True:
                    entropys: (bs, response_len)
                if calculate_sum_pi_squared is False:
                    sum_pi_squared: (bs, response_len)
                if return_valid_response_logits is True:
                    valid_response_logits: (num_valid_response_tokens, vocab)
        """
        calculate_sum_pi_squared = self.config.get("calculate_sum_pi_squared", False)
        sum_pi_squared_checkpointing = self.config.get("sum_pi_squared_checkpointing", False)
        # PrefixGrouper path for shared-prefix optimization
        if self.use_prefix_grouper:
            can_use_pg = (
                not self.use_remove_padding
                and not self.use_ulysses_sp
                and not self.use_fused_kernels
                and not self.use_dynamic_bsz
            )
            if can_use_pg and "response_mask" in micro_batch and "uid" in micro_batch:
                from verl.trainer.ppo.prefix_grouper_utils import forward_micro_batch_with_prefix_grouper

                return forward_micro_batch_with_prefix_grouper(
                    micro_batch=micro_batch,
                    model=self.actor_module,
                    temperature=temperature,
                    calculate_entropy=calculate_entropy,
                    device_name=self.device_name,
                    param_dtype=self.param_dtype,
                    use_chunking_entropy=self.config.get("entropy_from_logits_with_chunking", False),
                )

        response_length = micro_batch["responses"].size(-1)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch.keys():
            from verl.utils.model import extract_multi_modal_inputs

            multi_modal_inputs = extract_multi_modal_inputs(micro_batch["multi_modal_inputs"])

        with torch.autocast(device_type=self.device_name, dtype=self.param_dtype):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            entropy = None
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 4, seqlen) -> (4, bsz, seqlen)

            if self.use_remove_padding:
                input_ids_rmpad, indices, cu_seqlens, *_ = unpad_input(
                    input_ids.unsqueeze(-1), attention_mask
                )  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = (
                        index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                        .transpose(0, 1)
                        .unsqueeze(1)
                    )  # (4, bsz, seqlen) -> (4, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(
                        rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                    ).transpose(0, 1)

                is_mask_all_zero = attention_mask.sum() == 0
                if is_mask_all_zero:
                    input_ids_rmpad = torch.zeros(
                        (1, self.ulysses_sequence_parallel_size),
                        device=input_ids.device,
                        dtype=input_ids.dtype,
                    )
                    if position_ids.dim() == 3:
                        position_ids_rmpad = torch.zeros(
                            (position_ids.shape[0], 1, self.ulysses_sequence_parallel_size),
                            device=position_ids.device,
                            dtype=position_ids.dtype,
                        )
                    else:
                        position_ids_rmpad = torch.zeros(
                            (1, self.ulysses_sequence_parallel_size),
                            device=position_ids.device,
                            dtype=position_ids.dtype,
                        )

                if "image_bound" in multi_modal_inputs:
                    from verl.utils.dataset.vision_utils import process_multi_modal_inputs_for_minicpmo

                    multi_modal_inputs = process_multi_modal_inputs_for_minicpmo(
                        input_ids, attention_mask, position_ids, cu_seqlens, multi_modal_inputs
                    )

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    is_vlm_model = hasattr(
                        getattr(self.actor_module, "module", self.actor_module).config, "vision_config"
                    )
                    if is_vlm_model:
                        # vlm model's inputs will be sliced after embedding
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    else:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled,
                        position_ids_rmpad=None,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = self.actor_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs.squeeze(0)  # (total_nnz,)
                    entropy_rmpad = output.entropy.squeeze(0)  # (total_nnz,)
                    if return_valid_response_logits:
                        raise NotImplementedError("VIMPO exact KL does not support fused actor kernels.")

                else:
                    logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
                    logits_rmpad.div_(temperature)

                    # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                    inplace_backward = True
                    if calculate_entropy:
                        inplace_backward = False
                    log_probs = logprobs_from_logits(
                        logits=logits_rmpad,
                        labels=input_ids_rmpad_rolled,
                        inplace_backward=inplace_backward,
                    )

                    # compute entropy
                    if calculate_entropy:
                        # ((total_nnz / sp) + pad)
                        entropy_rmpad = (
                            self.compute_entropy_from_logits(logits_rmpad)
                            if not self.config.entropy_checkpointing
                            else torch.utils.checkpoint.checkpoint(self.compute_entropy_from_logits, logits_rmpad)
                        )

                    # Compute sum_pi_squared if requested (for optimal_token_baseline)
                    if calculate_sum_pi_squared:
                        sum_pi_squared_rmpad = (
                            self.calculate_sum_pi_squared_from_logits(logits_rmpad)
                            if not sum_pi_squared_checkpointing
                            else torch.utils.checkpoint.checkpoint(
                                self.calculate_sum_pi_squared_from_logits, logits_rmpad
                            )
                        )

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outputs_and_unpad(
                        log_probs,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size,
                    )
                    if calculate_entropy:
                        entropy_rmpad = gather_outputs_and_unpad(
                            entropy_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                    if calculate_sum_pi_squared:
                        sum_pi_squared_rmpad = gather_outputs_and_unpad(
                            sum_pi_squared_rmpad, gather_dim=0, unpad_dim=0, padding_size=pad_size
                        )
                    if return_valid_response_logits:
                        logits_rmpad = gather_outputs_and_unpad(
                            logits_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )

                if is_mask_all_zero:
                    log_probs = log_probs[:0]
                    if calculate_entropy:
                        entropy_rmpad = entropy_rmpad[:0]

                # pad back to (bsz, seqlen)
                if calculate_entropy:
                    full_entropy = pad_input(
                        hidden_states=entropy_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                if calculate_sum_pi_squared:
                    full_sum_pi_squared = pad_input(
                        hidden_states=sum_pi_squared_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                full_log_probs = pad_input(
                    hidden_states=log_probs.unsqueeze(-1),
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )

                # only return response part:
                if calculate_entropy:
                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                if calculate_sum_pi_squared:
                    # (bsz, response_length)
                    sum_pi_squared = full_sum_pi_squared.squeeze(-1)[:, -response_length - 1 : -1]
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                if return_valid_response_logits:
                    response_position_mask = torch.zeros(
                        (batch_size, seqlen),
                        dtype=torch.bool,
                        device=attention_mask.device,
                    )
                    response_position_mask[:, -response_length - 1 : -1] = micro_batch["response_mask"].to(torch.bool)
                    valid_response_logits = logits_rmpad[response_position_mask.reshape(-1)[indices]]

            else:  # not using rmpad and no ulysses sp
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs[:, -response_length - 1 : -1]
                    entropy = output.entropy[:, -response_length - 1 : -1]  # (bsz, response_length)
                    if return_valid_response_logits:
                        raise NotImplementedError("VIMPO exact KL does not support fused actor kernels.")

                else:
                    logits = output.logits

                    logits.div_(temperature)
                    logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
                    log_probs = logprobs_from_logits(logits, micro_batch["responses"])
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)
                        else:
                            entropy = torch.utils.checkpoint.checkpoint(verl_F.entropy_from_logits, logits)
                    # Compute sum_pi_squared if requested (for optimal_token_baseline)
                    if calculate_sum_pi_squared:
                        sum_pi_squared = (
                            self.calculate_sum_pi_squared_from_logits(logits)
                            if not sum_pi_squared_checkpointing
                            else torch.utils.checkpoint.checkpoint(self.calculate_sum_pi_squared_from_logits, logits)
                        )
                    if return_valid_response_logits:
                        valid_response_logits = logits[micro_batch["response_mask"].to(torch.bool)]

            outputs = {"log_probs": log_probs}
            if calculate_entropy:
                outputs["entropys"] = entropy
            if calculate_sum_pi_squared:
                outputs["sum_pi_squared"] = sum_pi_squared
            if return_valid_response_logits:
                outputs["valid_response_logits"] = valid_response_logits
            return outputs

    def _parse_vimpo_mode(self, data: DataProto) -> dict[str, object]:
        vimpo_config = data.meta_info.get("vimpo_config", {}) or {}
        use_vimpo_loss = bool(vimpo_config.get("use_vimpo_loss", False))
        value_type = str(vimpo_config.get("value_type", "raw"))
        use_grpo_actor = bool(vimpo_config.get("use_grpo_actor", False))
        use_ppo_actor = bool(vimpo_config.get("use_ppo_actor", False))
        vimpo_kl_estimator = vimpo_config.get("vimpo_kl_estimator", None)
        vimpo_beta = float(vimpo_config.get("vimpo_beta", 0.0))
        vimpo_actor_coeff = float(vimpo_config.get("vimpo_actor_coeff", 1.0))
        vimpo_adaptive_actor_coeff = bool(vimpo_config.get("vimpo_adaptive_actor_coeff", False))
        vimpo_adaptive_actor_alpha = float(vimpo_config.get("vimpo_adaptive_actor_alpha", 0.15))
        vimpo_adaptive_actor_ema_decay = float(vimpo_config.get("vimpo_adaptive_actor_ema_decay", 0.9))
        vimpo_adaptive_actor_eps = float(vimpo_config.get("vimpo_adaptive_actor_eps", 1e-12))
        vimpo_cap_v0 = _optional_float(vimpo_config.get("vimpo_cap_v0", None), "vimpo_cap_v0")
        vimpo_value_loss_type = str(vimpo_config.get("vimpo_value_loss_type", "squared_error"))
        vimpo_value_huber_delta = float(vimpo_config.get("vimpo_value_huber_delta", 1.0))
        vimpo_value_loss_normalized = bool(vimpo_config.get("vimpo_value_loss_normalized", False))
        vimpo_value_loss_coeff = float(vimpo_config.get("vimpo_value_loss_coeff", 1.0))
        vimpo_ppo_whiten_adv = bool(vimpo_config.get("vimpo_ppo_whiten_adv", True))
        vimpo_detach_kl = bool(vimpo_config.get("vimpo_detach_kl", True))
        vocab_chunk_size = int(vimpo_config.get("vocab_chunk_size", 0))
        candidate_kl_topk = int(vimpo_config.get("candidate_kl_topk", 0))
        candidate_kl_M = int(vimpo_config.get("candidate_kl_M", 0))
        lam = float(vimpo_config.get("lam", 1.0))
        gamma = float(vimpo_config.get("gamma", 1.0))

        if use_vimpo_loss and value_type not in {"raw", "soft"}:
            raise ValueError(f"vimpo_config.value_type must be 'raw' or 'soft', got {value_type!r}.")
        if use_grpo_actor and use_ppo_actor:
            raise ValueError("vimpo_config.use_grpo_actor and vimpo_config.use_ppo_actor cannot both be True.")
        if use_vimpo_loss and value_type == "soft" and (use_grpo_actor or use_ppo_actor):
            raise NotImplementedError(
                "vimpo_config.value_type='soft' does not support use_grpo_actor or use_ppo_actor in v1."
            )
        if use_vimpo_loss and vimpo_kl_estimator not in {"exact", "candidate_set", "k1", "k2", "k3"}:
            raise NotImplementedError(
                f"VIMPO KL estimator {vimpo_kl_estimator!r} is not supported yet. "
                "Only 'exact', 'candidate_set', 'k1', 'k2', and 'k3' are implemented."
            )
        if use_vimpo_loss and value_type == "raw" and vimpo_kl_estimator == "k1":
            raise ValueError(
                "vimpo_config.vimpo_kl_estimator='k1' is degenerate for raw VIMPO: "
                "the sampled KL equals log(pi/ref), so the token term cancels to zero. "
                "Use 'exact', 'candidate_set', 'k2', or 'k3'."
            )
        if use_vimpo_loss and vimpo_kl_estimator == "candidate_set" and candidate_kl_topk <= 0:
            raise ValueError(
                f"VIMPO candidate_set KL requires candidate_kl_topk > 0, got {candidate_kl_topk}."
            )
        if use_vimpo_loss and vimpo_kl_estimator == "candidate_set" and candidate_kl_M != 0:
            raise NotImplementedError("VIMPO candidate_set KL currently supports policy top-k only; candidate_kl_M must be 0.")
        if use_vimpo_loss and self.config.use_kl_loss:
            raise NotImplementedError("VIMPO exact-KL mode does not support actor.use_kl_loss at the same time.")
        if use_vimpo_loss and gamma != 1.0:
            raise ValueError("VIMPO currently enforces gamma=1.0; remove vimpo_config.gamma or set it to 1.0.")
        if use_vimpo_loss and not (0 <= lam <= 1):
            raise ValueError(f"VIMPO requires lam in [0, 1], got {lam}.")
        if use_vimpo_loss and vimpo_value_loss_type not in {"squared_error", "huber"}:
            raise ValueError(
                f"vimpo_config.vimpo_value_loss_type must be 'squared_error' or 'huber', got {vimpo_value_loss_type!r}."
            )
        if use_vimpo_loss and vimpo_value_huber_delta <= 0:
            raise ValueError(f"vimpo_config.vimpo_value_huber_delta must be positive, got {vimpo_value_huber_delta}.")
        if use_vimpo_loss and (not math.isfinite(vimpo_value_loss_coeff) or vimpo_value_loss_coeff < 0):
            raise ValueError(f"vimpo_config.vimpo_value_loss_coeff must be non-negative, got {vimpo_value_loss_coeff}.")
        if use_vimpo_loss and vimpo_adaptive_actor_alpha < 0:
            raise ValueError(f"vimpo_config.vimpo_adaptive_actor_alpha must be non-negative, got {vimpo_adaptive_actor_alpha}.")
        if use_vimpo_loss and not 0 <= vimpo_adaptive_actor_ema_decay < 1:
            raise ValueError(
                "vimpo_config.vimpo_adaptive_actor_ema_decay must be in [0, 1), "
                f"got {vimpo_adaptive_actor_ema_decay}."
            )
        if use_vimpo_loss and vimpo_adaptive_actor_eps <= 0:
            raise ValueError(f"vimpo_config.vimpo_adaptive_actor_eps must be positive, got {vimpo_adaptive_actor_eps}.")
        if use_vimpo_loss and vimpo_adaptive_actor_coeff:
            if value_type != "raw":
                raise NotImplementedError("vimpo_adaptive_actor_coeff currently requires vimpo_config.value_type='raw'.")
            if not use_ppo_actor:
                raise NotImplementedError("vimpo_adaptive_actor_coeff currently requires vimpo_config.use_ppo_actor=True.")
            if use_grpo_actor:
                raise NotImplementedError("vimpo_adaptive_actor_coeff does not support use_grpo_actor=True.")

        return {
            "use_vimpo_loss": use_vimpo_loss,
            "value_type": value_type,
            "use_grpo_actor": use_grpo_actor,
            "use_ppo_actor": use_ppo_actor,
            "vimpo_beta": vimpo_beta,
            "vimpo_actor_coeff": vimpo_actor_coeff,
            "vimpo_adaptive_actor_coeff": vimpo_adaptive_actor_coeff,
            "vimpo_adaptive_actor_alpha": vimpo_adaptive_actor_alpha,
            "vimpo_adaptive_actor_ema_decay": vimpo_adaptive_actor_ema_decay,
            "vimpo_adaptive_actor_eps": vimpo_adaptive_actor_eps,
            "vimpo_cap_v0": vimpo_cap_v0,
            "vimpo_value_loss_type": vimpo_value_loss_type,
            "vimpo_value_huber_delta": vimpo_value_huber_delta,
            "vimpo_value_loss_normalized": vimpo_value_loss_normalized,
            "vimpo_value_loss_coeff": vimpo_value_loss_coeff,
            "vimpo_ppo_whiten_adv": vimpo_ppo_whiten_adv,
            "vimpo_detach_kl": vimpo_detach_kl,
            "vimpo_kl_estimator": vimpo_kl_estimator,
            "vocab_chunk_size": vocab_chunk_size,
            "candidate_kl_topk": candidate_kl_topk,
            "candidate_kl_M": candidate_kl_M,
            "lam": lam,
        }

    def _optimizer_step(self):
        assert self.config.grad_clip is not None
        if self.scaler is not None:
            self.scaler.unscale_(self.actor_optimizer)
        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        elif isinstance(self.actor_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)

        if isinstance(grad_norm, DTensor):
            grad_norm = grad_norm.full_tensor()

        # if grad_norm is not finite, skip the update
        if self.scaler is not None:
            self.scaler.step(self.actor_optimizer)
            self.scaler.update()
        else:
            if not torch.isfinite(grad_norm):
                print(f"WARN: rank {torch.distributed.get_rank()} grad_norm is not finite: {grad_norm}")
                self.actor_optimizer.zero_grad()
            else:
                self.actor_optimizer.step()

        # Clear cached weight scales for QAT (weights changed)
        if getattr(self.actor_module, "_qat_fuse_enabled", False):
            from verl.utils.qat import invalidate_all_scales

            invalidate_all_scales(self.actor_module)

        return grad_norm

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_prob(self, data: DataProto, calculate_entropy: bool = False) -> dict[str, torch.Tensor]:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            dict[str, torch.Tensor]: a dict containing keys
                - ``log_probs``: tensor of shape [batch_size, response_length]. torch.float32.
                - ``entropys``: tensor of shape [batch_size, response_length]. torch.float32.
                - ``sum_pi_squared``: tensor of shape [batch_size, response_length]. torch.float32.
                - ``ref_log_prob``: optional tensor used by VIMPO soft value. torch.float32.
                - ``vimpo_old_token_kl``: optional detached old-policy KL tensor used by VIMPO soft value.
        """
        calculate_sum_pi_squared = self.config.get("calculate_sum_pi_squared", False)

        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
        pad_token_id = data.meta_info.get("pad_token_id", 0)
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        vimpo_mode = self._parse_vimpo_mode(data)
        compute_soft_old_kl = bool(vimpo_mode["use_vimpo_loss"] and vimpo_mode["value_type"] == "soft")
        if compute_soft_old_kl and "response_mask" not in data.batch:
            raise RuntimeError("VIMPO soft value old-policy KL computation requires response_mask in the batch.")

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []
        if compute_soft_old_kl and "response_mask" in data.batch:
            select_keys.append("response_mask")
        if self.use_prefix_grouper:
            for key in ["prompts", "response_mask"]:
                if key in data.batch and key not in select_keys:
                    select_keys.append(key)
            if "uid" in data.non_tensor_batch:
                non_tensor_select_keys.append("uid")

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        if use_dynamic_bsz:
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            dp_group = torch.distributed.group.WORLD if torch.distributed.is_initialized() else None
            micro_batches, batch_idx_list = prepare_dynamic_batch(
                data,
                max_token_len=max_token_len,
                dp_group=dp_group,
            )
        else:
            micro_batches = data.split(micro_batch_size)

        log_probs_lst = []
        entropy_lst = []
        sum_pi_squared_lst = []
        ref_log_probs_lst = []
        vimpo_old_token_kl_lst = []
        for micro_batch in micro_batches:
            micro_batch = micro_batch.to(get_device_id())
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch, "pad_token_id": pad_token_id}
            with torch.no_grad():
                return_valid_response_logits = bool(
                    compute_soft_old_kl and vimpo_mode["vimpo_kl_estimator"] not in {"k1", "k2", "k3"}
                )
                outputs = self._forward_micro_batch(
                    model_inputs,
                    temperature=temperature,
                    calculate_entropy=calculate_entropy,
                    return_valid_response_logits=return_valid_response_logits,
                )
            log_probs_lst.append(outputs["log_probs"])
            if calculate_entropy:
                entropy_lst.append(outputs["entropys"])
            if calculate_sum_pi_squared:
                sum_pi_squared_lst.append(outputs["sum_pi_squared"])
            if compute_soft_old_kl:
                from recipe.vimpo.vimpo_core_algos import (
                    compute_sampled_token_kl_from_log_probs,
                    compute_vimpo_flat_kl_from_logits,
                )

                response_mask = model_inputs["response_mask"]
                valid_response_mask = response_mask.to(torch.bool)
                with torch.no_grad():
                    ref_outputs = self._forward_reference_micro_batch(
                        model_inputs,
                        temperature=temperature,
                    )
                    ref_log_prob = torch.zeros_like(outputs["log_probs"])
                    if valid_response_mask.any():
                        ref_valid_log_prob = logprobs_from_logits(
                            ref_outputs["valid_response_logits"],
                            model_inputs["responses"][valid_response_mask],
                            inplace_backward=False,
                        )
                        ref_log_prob[valid_response_mask] = ref_valid_log_prob.to(ref_log_prob.dtype)
                    if vimpo_mode["vimpo_kl_estimator"] in {"k1", "k2", "k3"}:
                        vimpo_old_token_kl = compute_sampled_token_kl_from_log_probs(
                            log_prob=outputs["log_probs"],
                            ref_log_prob=ref_log_prob,
                            estimator=vimpo_mode["vimpo_kl_estimator"],
                            detach_kl=True,
                            output_dtype=outputs["log_probs"].dtype,
                        )
                    else:
                        vimpo_old_token_kl = torch.zeros_like(outputs["log_probs"])
                        if valid_response_mask.any():
                            flat_vimpo_old_token_kl = compute_vimpo_flat_kl_from_logits(
                                policy_logits=outputs["valid_response_logits"],
                                ref_logits=ref_outputs["valid_response_logits"],
                                estimator=vimpo_mode["vimpo_kl_estimator"],
                                detach_kl=True,
                                vocab_chunk_size=vimpo_mode["vocab_chunk_size"],
                                candidate_kl_topk=vimpo_mode["candidate_kl_topk"],
                                candidate_kl_M=vimpo_mode["candidate_kl_M"],
                                compute_dtype=outputs["valid_response_logits"].dtype,
                                output_dtype=vimpo_old_token_kl.dtype,
                            )
                            vimpo_old_token_kl[valid_response_mask] = flat_vimpo_old_token_kl
                ref_log_probs_lst.append(ref_log_prob)
                vimpo_old_token_kl_lst.append(vimpo_old_token_kl)
                outputs.pop("valid_response_logits", None)
                del ref_outputs

        log_probs = torch.concat(log_probs_lst, dim=0)
        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)
        if calculate_sum_pi_squared:
            sum_pi_squared = torch.concat(sum_pi_squared_lst, dim=0)
        if compute_soft_old_kl:
            ref_log_prob = torch.concat(ref_log_probs_lst, dim=0)
            vimpo_old_token_kl = torch.concat(vimpo_old_token_kl_lst, dim=0)

        if use_dynamic_bsz:
            log_probs = restore_dynamic_batch(log_probs, batch_idx_list)
            if calculate_entropy:
                entropys = restore_dynamic_batch(entropys, batch_idx_list)
            if calculate_sum_pi_squared:
                sum_pi_squared = restore_dynamic_batch(sum_pi_squared, batch_idx_list)
            if compute_soft_old_kl:
                ref_log_prob = restore_dynamic_batch(ref_log_prob, batch_idx_list)
                vimpo_old_token_kl = restore_dynamic_batch(vimpo_old_token_kl, batch_idx_list)

        outputs = {"log_probs": log_probs}
        if calculate_entropy:
            outputs["entropys"] = entropys
        if calculate_sum_pi_squared:
            outputs["sum_pi_squared"] = sum_pi_squared
        if compute_soft_old_kl:
            outputs["ref_log_prob"] = ref_log_prob
            outputs["vimpo_old_token_kl"] = vimpo_old_token_kl
        return outputs

    def _forward_reference_micro_batch(
        self,
        model_inputs: dict[str, torch.Tensor],
        temperature: float,
    ) -> dict[str, torch.Tensor]:
        if self.ref_policy is not None:
            return self.ref_policy._forward_micro_batch(
                model_inputs,
                temperature=temperature,
                calculate_entropy=False,
                return_valid_response_logits=True,
            )
        if self._use_lora_reference:
            with self.actor_module.disable_adapter():
                return self._forward_micro_batch(
                    model_inputs,
                    temperature=temperature,
                    calculate_entropy=False,
                    return_valid_response_logits=True,
                )
        raise RuntimeError("VIMPO requires a reference-policy path for ref log-prob or KL computation.")

    def _select_update_policy_data(self, data: DataProto, vimpo_mode: dict[str, object]) -> DataProto:
        select_keys = [
            "responses",
            "response_mask",
            "input_ids",
            "attention_mask",
            "position_ids",
        ]
        if vimpo_mode["use_vimpo_loss"]:
            select_keys.append("token_level_rewards")
        if (
            (not vimpo_mode["use_vimpo_loss"])
            or vimpo_mode["use_grpo_actor"]
            or vimpo_mode["use_ppo_actor"]
            or vimpo_mode["value_type"] == "soft"
        ):
            select_keys.append("old_log_probs")
        if (not vimpo_mode["use_vimpo_loss"]) or vimpo_mode["use_grpo_actor"]:
            select_keys.append("advantages")
        if self.use_prefix_grouper and "prompts" in data.batch.keys():
            select_keys.append("prompts")
        if self.config.use_kl_loss or (vimpo_mode["use_vimpo_loss"] and vimpo_mode["value_type"] == "soft"):
            select_keys.append("ref_log_prob")
        if vimpo_mode["use_vimpo_loss"] and vimpo_mode["value_type"] == "soft":
            select_keys.extend(["vimpo_old_token_kl", "vimpo_soft_v0"])
        if vimpo_mode["use_vimpo_loss"] and vimpo_mode["value_type"] == "raw":
            select_keys.append("vimpo_reward_baseline")
        if "rollout_is_weights" in data.batch.keys():
            select_keys.append("rollout_is_weights")
        if "rollout_log_probs" in data.batch.keys() and ((not vimpo_mode["use_vimpo_loss"]) or vimpo_mode["use_grpo_actor"]):
            select_keys.append("rollout_log_probs")

        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        non_tensor_select_keys = []
        if has_multi_modal_inputs:
            non_tensor_select_keys.append("multi_modal_inputs")
        if (self.use_prefix_grouper or vimpo_mode["use_vimpo_loss"]) and "uid" in data.non_tensor_batch.keys():
            non_tensor_select_keys.append("uid")

        return data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

    @torch.no_grad()
    def _compute_vimpo_value_loss_norm_denom(
        self,
        mini_batch: DataProto,
        vimpo_mode: dict[str, object],
    ) -> torch.Tensor | None:
        if not vimpo_mode["use_vimpo_loss"] or not vimpo_mode["vimpo_value_loss_normalized"]:
            return None

        if "vimpo_value_loss_norm_denom" not in mini_batch.meta_info:
            raise RuntimeError(
                "vimpo_value_loss_normalized=True requires trainer-side full-batch "
                "vimpo_value_loss_norm_denom in DataProto.meta_info."
            )
        denom = torch.as_tensor(mini_batch.meta_info["vimpo_value_loss_norm_denom"], dtype=torch.float32)
        if denom.item() < 1e-4:
            raise ValueError(
                "vimpo_value_loss_normalized=True requires full actor-update batch target MSE denominator >= 1e-4, "
                f"got {denom.item()}."
            )
        return denom.detach()

    def _vimpo_forward_needs_valid_logits(self, vimpo_mode: dict[str, object]) -> bool:
        return bool(
            vimpo_mode["use_vimpo_loss"]
            and vimpo_mode["value_type"] == "raw"
            and vimpo_mode["vimpo_kl_estimator"] not in {"k1", "k2", "k3"}
        )

    def _vimpo_value_loss_metrics(
        self,
        vimpo_loss_metrics: dict[str, torch.Tensor],
        vimpo_mode: dict[str, object],
    ) -> dict[str, float]:
        metrics = {
            "actor/vimpo_value_loss": vimpo_loss_metrics["vimpo_value_loss_for_backward"].item(),
        }
        if vimpo_mode["vimpo_value_loss_normalized"] or float(vimpo_mode["vimpo_value_loss_coeff"]) != 1.0:
            metrics["actor/vimpo_value_loss_raw"] = vimpo_loss_metrics["vimpo_value_loss_raw"].item()
        if "vimpo_value_loss_norm_denom" in vimpo_loss_metrics:
            metrics["actor/vimpo_value_loss_norm_denom"] = vimpo_loss_metrics["vimpo_value_loss_norm_denom"].item()
        return metrics

    def _compute_raw_vimpo_loss_for_micro_batch(
        self,
        model_inputs: dict[str, torch.Tensor],
        outputs: dict[str, torch.Tensor],
        log_prob: torch.Tensor,
        response_mask: torch.Tensor,
        temperature: float,
        vimpo_mode: dict[str, object],
        loss_agg_mode: str,
        value_loss_norm_denom: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float]]:
        from recipe.vimpo.vimpo_core_algos import (
            compute_sampled_token_kl_from_log_probs,
            compute_vimpo_flat_kl_from_logits,
            compute_vimpo_loss,
        )

        valid_response_mask = response_mask.to(torch.bool)
        with torch.no_grad():
            ref_outputs = self._forward_reference_micro_batch(
                model_inputs,
                temperature=temperature,
            )
            ref_log_prob = torch.zeros_like(log_prob)
            if valid_response_mask.any():
                ref_valid_log_prob = logprobs_from_logits(
                    ref_outputs["valid_response_logits"],
                    model_inputs["responses"][valid_response_mask],
                    inplace_backward=False,
                )
                ref_log_prob[valid_response_mask] = ref_valid_log_prob.to(ref_log_prob.dtype)

        if vimpo_mode["vimpo_kl_estimator"] in {"k1", "k2", "k3"}:
            vimpo_token_kl = compute_sampled_token_kl_from_log_probs(
                log_prob=log_prob,
                ref_log_prob=ref_log_prob,
                estimator=vimpo_mode["vimpo_kl_estimator"],
                detach_kl=vimpo_mode["vimpo_detach_kl"],
                output_dtype=log_prob.dtype,
            )
        else:
            policy_logits = outputs["valid_response_logits"]
            vimpo_token_kl = torch.zeros_like(log_prob)
            if valid_response_mask.any():
                flat_vimpo_token_kl = compute_vimpo_flat_kl_from_logits(
                    policy_logits=policy_logits,
                    ref_logits=ref_outputs["valid_response_logits"],
                    estimator=vimpo_mode["vimpo_kl_estimator"],
                    detach_kl=vimpo_mode["vimpo_detach_kl"],
                    vocab_chunk_size=vimpo_mode["vocab_chunk_size"],
                    candidate_kl_topk=vimpo_mode["candidate_kl_topk"],
                    candidate_kl_M=vimpo_mode["candidate_kl_M"],
                    compute_dtype=policy_logits.dtype,
                    output_dtype=vimpo_token_kl.dtype,
                )
                vimpo_token_kl[valid_response_mask] = flat_vimpo_token_kl

        vimpo_terminal_mse, vimpo_kl_agg, vimpo_loss_metrics = compute_vimpo_loss(
            log_prob=log_prob,
            ref_log_prob=ref_log_prob,
            token_kl=vimpo_token_kl,
            token_level_rewards=model_inputs["token_level_rewards"],
            response_mask=response_mask,
            beta=vimpo_mode["vimpo_beta"],
            gamma=1.0,
            reward_baseline=model_inputs["vimpo_reward_baseline"],
            loss_agg_mode=loss_agg_mode,
            global_batch_info=self.config.global_batch_info,
            value_loss_type=vimpo_mode["vimpo_value_loss_type"],
            huber_delta=vimpo_mode["vimpo_value_huber_delta"],
            value_loss_normalized=vimpo_mode["vimpo_value_loss_normalized"],
            value_loss_coeff=vimpo_mode["vimpo_value_loss_coeff"],
            value_loss_norm_denom=value_loss_norm_denom,
        )
        metrics = self._vimpo_value_loss_metrics(vimpo_loss_metrics, vimpo_mode)
        return vimpo_terminal_mse, vimpo_kl_agg, ref_log_prob, vimpo_token_kl, metrics

    @torch.no_grad()
    def _token_mean_coeff_scale(self, response_mask: torch.Tensor) -> torch.Tensor:
        global_batch_info = getattr(self.config, "global_batch_info", {}) or {}
        dp_size = float(global_batch_info.get("dp_size", 1.0))
        batch_num_tokens = global_batch_info.get("batch_num_tokens", None)
        if batch_num_tokens is None:
            denominator = response_mask.to(torch.float32).sum()
        else:
            denominator = torch.as_tensor(batch_num_tokens, dtype=torch.float32, device=response_mask.device)
        return torch.as_tensor(dp_size, dtype=torch.float32, device=response_mask.device) / denominator.clamp_min(1.0)

    @torch.no_grad()
    def _global_logit_grad_rms(self, stats: tuple[torch.Tensor, torch.Tensor] | None) -> torch.Tensor | None:
        if stats is None:
            return None
        sum_sq, count = stats
        reduced = torch.stack(
            [
                sum_sq.detach().to(device=sum_sq.device, dtype=torch.float32),
                count.detach().to(device=sum_sq.device, dtype=torch.float32),
            ]
        )
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(reduced, op=torch.distributed.ReduceOp.SUM)
        if reduced[1].item() <= 0:
            return None
        return torch.sqrt(reduced[0] / reduced[1].clamp_min(1.0))

    @torch.no_grad()
    def _init_vimpo_terminal_stats(self, device: torch.device) -> torch.Tensor:
        return torch.zeros(_VIMPO_TERM_NUM_STATS, dtype=torch.float64, device=device)

    @torch.no_grad()
    def _init_vimpo_adv_stats(self, device: torch.device) -> torch.Tensor:
        return torch.zeros(_VIMPO_ADV_NUM_STATS, dtype=torch.float64, device=device)

    @torch.no_grad()
    def _init_vimpo_kl_stats(self, device: torch.device) -> torch.Tensor:
        return torch.zeros(_VIMPO_KL_NUM_STATS, dtype=torch.float64, device=device)

    @torch.no_grad()
    def _accumulate_vimpo_kl_stats(
        self,
        stats: torch.Tensor | None,
        token_kl: torch.Tensor,
        response_mask: torch.Tensor,
    ) -> torch.Tensor:
        if stats is None:
            stats = self._init_vimpo_kl_stats(token_kl.device)
        valid_mask = response_mask.to(torch.bool)
        if not valid_mask.any():
            return stats
        valid_kl = torch.masked_select(token_kl.detach().to(torch.float64), valid_mask)
        stats[_VIMPO_KL_COUNT] += valid_kl.numel()
        stats[_VIMPO_KL_SUM] += valid_kl.sum()
        return stats

    @torch.no_grad()
    def _accumulate_vimpo_terminal_stats(
        self,
        stats: torch.Tensor | None,
        terminal_value: torch.Tensor,
        terminal_target: torch.Tensor,
        final_reward: torch.Tensor,
        baseline: torch.Tensor,
        vimpo_mode: dict[str, object],
    ) -> torch.Tensor:
        if stats is None:
            stats = self._init_vimpo_terminal_stats(terminal_value.device)
        value = terminal_value.detach().to(torch.float64)
        target = terminal_target.detach().to(torch.float64)
        reward = final_reward.detach().to(torch.float64)
        baseline = baseline.detach().to(torch.float64)
        residual = value - target
        abs_residual = residual.abs()
        huber_delta = torch.as_tensor(vimpo_mode["vimpo_value_huber_delta"], dtype=torch.float64, device=value.device)
        huber_loss = torch.where(
            abs_residual <= huber_delta,
            0.5 * residual.square(),
            huber_delta * (abs_residual - 0.5 * huber_delta),
        )
        huber_clipped = (abs_residual > huber_delta).to(torch.float64)

        stats[_VIMPO_TERM_COUNT] += value.numel()
        stats[_VIMPO_TERM_VALUE_SUM] += value.sum()
        stats[_VIMPO_TERM_TARGET_SUM] += target.sum()
        stats[_VIMPO_TERM_REWARD_SUM] += reward.sum()
        stats[_VIMPO_TERM_BASELINE_SUM] += baseline.sum()
        stats[_VIMPO_TERM_ERROR_SQ_SUM] += residual.square().sum()
        stats[_VIMPO_TERM_ABS_ERROR_SUM] += abs_residual.sum()
        stats[_VIMPO_TERM_TARGET_SQ_SUM] += target.square().sum()
        stats[_VIMPO_TERM_HUBER_SUM] += huber_loss.sum()
        stats[_VIMPO_TERM_HUBER_CLIP_SUM] += huber_clipped.sum()
        return stats

    @torch.no_grad()
    def _accumulate_raw_vimpo_terminal_stats(
        self,
        stats: torch.Tensor | None,
        log_prob: torch.Tensor,
        ref_log_prob: torch.Tensor,
        token_kl: torch.Tensor,
        model_inputs: dict[str, torch.Tensor],
        response_mask: torch.Tensor,
        vimpo_mode: dict[str, object],
    ) -> torch.Tensor:
        dtype = log_prob.dtype
        response_mask_f = response_mask.to(dtype=dtype)
        beta = torch.as_tensor(vimpo_mode["vimpo_beta"], dtype=dtype, device=log_prob.device)
        terminal_value = torch.sum(beta * (log_prob - ref_log_prob - token_kl) * response_mask_f, dim=-1)
        final_reward = torch.sum(model_inputs["token_level_rewards"].to(dtype) * response_mask_f, dim=-1)
        baseline = model_inputs["vimpo_reward_baseline"].to(dtype=dtype, device=log_prob.device)
        terminal_target = final_reward - baseline
        return self._accumulate_vimpo_terminal_stats(
            stats=stats,
            terminal_value=terminal_value,
            terminal_target=terminal_target,
            final_reward=final_reward,
            baseline=baseline,
            vimpo_mode=vimpo_mode,
        )

    @torch.no_grad()
    def _accumulate_soft_vimpo_terminal_stats(
        self,
        stats: torch.Tensor | None,
        log_prob: torch.Tensor,
        model_inputs: dict[str, torch.Tensor],
        response_mask: torch.Tensor,
        vimpo_mode: dict[str, object],
    ) -> torch.Tensor:
        dtype = log_prob.dtype
        response_mask_f = response_mask.to(dtype=dtype)
        beta = torch.as_tensor(vimpo_mode["vimpo_beta"], dtype=dtype, device=log_prob.device)
        terminal_value = beta * torch.sum((log_prob - model_inputs["ref_log_prob"].to(dtype)) * response_mask_f, dim=-1)
        final_reward = torch.sum(model_inputs["token_level_rewards"].to(dtype) * response_mask_f, dim=-1)
        baseline = model_inputs["vimpo_soft_v0"].to(dtype=dtype, device=log_prob.device)
        terminal_target = final_reward - baseline
        return self._accumulate_vimpo_terminal_stats(
            stats=stats,
            terminal_value=terminal_value,
            terminal_target=terminal_target,
            final_reward=final_reward,
            baseline=baseline,
            vimpo_mode=vimpo_mode,
        )

    @torch.no_grad()
    def _accumulate_vimpo_adv_stats(
        self,
        stats: torch.Tensor | None,
        advantages: torch.Tensor,
        raw_advantages: torch.Tensor,
        values: torch.Tensor,
        response_mask: torch.Tensor,
    ) -> torch.Tensor:
        if stats is None:
            stats = self._init_vimpo_adv_stats(advantages.device)
        valid_mask = response_mask.to(torch.bool)
        if not valid_mask.any():
            return stats
        valid_adv = torch.masked_select(advantages.detach().to(torch.float64), valid_mask)
        valid_raw_adv = torch.masked_select(raw_advantages.detach().to(torch.float64), valid_mask)
        valid_values = torch.masked_select(values.detach().to(torch.float64), valid_mask)
        stats[_VIMPO_ADV_COUNT] += valid_adv.numel()
        stats[_VIMPO_ADV_SUM] += valid_adv.sum()
        stats[_VIMPO_ADV_SQ_SUM] += valid_adv.square().sum()
        stats[_VIMPO_ADV_RAW_SUM] += valid_raw_adv.sum()
        stats[_VIMPO_ADV_RAW_SQ_SUM] += valid_raw_adv.square().sum()
        stats[_VIMPO_ADV_RAW_ABS_SUM] += valid_raw_adv.abs().sum()
        stats[_VIMPO_ADV_RAW_POS_SUM] += (valid_raw_adv > 0).to(torch.float64).sum()
        stats[_VIMPO_ADV_VALUE_SUM] += valid_values.sum()
        return stats

    @torch.no_grad()
    def _reduce_vimpo_stats(self, stats: torch.Tensor | None) -> torch.Tensor | None:
        if stats is None:
            return None
        reduced = stats.detach().clone()
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(reduced, op=torch.distributed.ReduceOp.SUM)
        return reduced

    @torch.no_grad()
    def _append_global_vimpo_metrics(
        self,
        metrics: dict,
        terminal_stats: torch.Tensor | None,
        adv_stats: torch.Tensor | None,
        kl_stats: torch.Tensor | None,
        vimpo_mode: dict[str, object],
    ) -> None:
        terminal_stats = self._reduce_vimpo_stats(terminal_stats)
        if terminal_stats is not None and terminal_stats[_VIMPO_TERM_COUNT].item() > 0:
            count = terminal_stats[_VIMPO_TERM_COUNT].clamp_min(1.0)
            mse = terminal_stats[_VIMPO_TERM_ERROR_SQ_SUM] / count
            rmse = torch.sqrt(mse)
            target_rms = torch.sqrt(terminal_stats[_VIMPO_TERM_TARGET_SQ_SUM] / count)
            fixed_terminal_metrics = {
                "actor/vimpo_terminal_value_global": (terminal_stats[_VIMPO_TERM_VALUE_SUM] / count).item(),
                "actor/vimpo_terminal_target_global": (terminal_stats[_VIMPO_TERM_TARGET_SUM] / count).item(),
                "actor/vimpo_final_reward_global": (terminal_stats[_VIMPO_TERM_REWARD_SUM] / count).item(),
                "actor/vimpo_reward_baseline_global": (terminal_stats[_VIMPO_TERM_BASELINE_SUM] / count).item(),
                "actor/vimpo_terminal_squared_error_mean_global": mse.item(),
                "actor/vimpo_terminal_abs_error_mean_global": (terminal_stats[_VIMPO_TERM_ABS_ERROR_SUM] / count).item(),
                "actor/vimpo_terminal_rmse_global": rmse.item(),
                "actor/vimpo_terminal_target_rms_global": target_rms.item(),
                "actor/vimpo_terminal_rmse_over_target_rms_global": (rmse / (target_rms + 1e-8)).item(),
            }
            if vimpo_mode["vimpo_value_loss_type"] == "huber":
                fixed_terminal_metrics["actor/vimpo_terminal_huber_loss_mean_global"] = (
                    terminal_stats[_VIMPO_TERM_HUBER_SUM] / count
                ).item()
                fixed_terminal_metrics["actor/vimpo_terminal_huber_clipped_frac_global"] = (
                    terminal_stats[_VIMPO_TERM_HUBER_CLIP_SUM] / count
                ).item()
            metrics.update({key: [value] for key, value in fixed_terminal_metrics.items()})

        adv_stats = self._reduce_vimpo_stats(adv_stats)
        if adv_stats is not None and adv_stats[_VIMPO_ADV_COUNT].item() > 0:
            count = adv_stats[_VIMPO_ADV_COUNT].clamp_min(1.0)
            adv_mean = adv_stats[_VIMPO_ADV_SUM] / count
            raw_adv_mean = adv_stats[_VIMPO_ADV_RAW_SUM] / count
            fixed_adv_metrics = {
                "actor/vimpo_adv_mean_global": adv_mean.item(),
                "actor/vimpo_adv_std_global": torch.sqrt(
                    (adv_stats[_VIMPO_ADV_SQ_SUM] / count - adv_mean.square()).clamp_min(0.0)
                ).item(),
                "actor/vimpo_adv_raw_mean_global": raw_adv_mean.item(),
                "actor/vimpo_adv_raw_std_global": torch.sqrt(
                    (adv_stats[_VIMPO_ADV_RAW_SQ_SUM] / count - raw_adv_mean.square()).clamp_min(0.0)
                ).item(),
                "actor/vimpo_adv_raw_abs_mean_global": (adv_stats[_VIMPO_ADV_RAW_ABS_SUM] / count).item(),
                "actor/vimpo_adv_raw_pos_frac_global": (adv_stats[_VIMPO_ADV_RAW_POS_SUM] / count).item(),
                "actor/vimpo_value_mean_global": (adv_stats[_VIMPO_ADV_VALUE_SUM] / count).item(),
            }
            metrics.update({key: [value] for key, value in fixed_adv_metrics.items()})

        kl_stats = self._reduce_vimpo_stats(kl_stats)
        if kl_stats is not None and kl_stats[_VIMPO_KL_COUNT].item() > 0:
            metrics["actor/vimpo_kl_mean_global"] = [
                (kl_stats[_VIMPO_KL_SUM] / kl_stats[_VIMPO_KL_COUNT].clamp_min(1.0)).item()
            ]

    def _compute_soft_vimpo_loss_for_micro_batch(
        self,
        model_inputs: dict[str, torch.Tensor],
        log_prob: torch.Tensor,
        response_mask: torch.Tensor,
        vimpo_mode: dict[str, object],
        loss_agg_mode: str,
        value_loss_norm_denom: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
        from recipe.vimpo.vimpo_core_algos import compute_soft_trajectory_vimpo_loss

        vimpo_terminal_mse, vimpo_kl_agg, vimpo_loss_metrics = compute_soft_trajectory_vimpo_loss(
            log_prob=log_prob,
            ref_log_prob=model_inputs["ref_log_prob"],
            old_token_kl=model_inputs["vimpo_old_token_kl"],
            token_level_rewards=model_inputs["token_level_rewards"],
            response_mask=response_mask,
            soft_value_baseline=model_inputs["vimpo_soft_v0"],
            beta=vimpo_mode["vimpo_beta"],
            loss_agg_mode=loss_agg_mode,
            global_batch_info=self.config.global_batch_info,
            value_loss_type=vimpo_mode["vimpo_value_loss_type"],
            huber_delta=vimpo_mode["vimpo_value_huber_delta"],
            value_loss_normalized=vimpo_mode["vimpo_value_loss_normalized"],
            value_loss_coeff=vimpo_mode["vimpo_value_loss_coeff"],
            value_loss_norm_denom=value_loss_norm_denom,
        )
        metrics = self._vimpo_value_loss_metrics(vimpo_loss_metrics, vimpo_mode)
        return vimpo_terminal_mse, vimpo_kl_agg, metrics

    def _compute_raw_vimpo_ppo_targets_for_micro_batch(
        self,
        log_prob: torch.Tensor,
        ref_log_prob: torch.Tensor,
        vimpo_token_kl: torch.Tensor,
        model_inputs: dict[str, torch.Tensor],
        response_mask: torch.Tensor,
        vimpo_mode: dict[str, object],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float]]:
        from recipe.vimpo.vimpo_core_algos import compute_detached_vimpo_ppo_targets

        detached_advantages, detached_values, _, raw_advantages = compute_detached_vimpo_ppo_targets(
            log_prob=log_prob,
            ref_log_prob=ref_log_prob,
            token_kl=vimpo_token_kl,
            response_mask=response_mask,
            beta=vimpo_mode["vimpo_beta"],
            gamma=1.0,
            lam=vimpo_mode["lam"],
            whiten_adv=vimpo_mode["vimpo_ppo_whiten_adv"],
        )
        return detached_advantages, detached_values, raw_advantages, {}

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()
        vimpo_mode = self._parse_vimpo_mode(data)

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        pad_token_id = data.meta_info.get("pad_token_id", 0)

        data = self._select_update_policy_data(data, vimpo_mode)

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        mini_batches = data.split(self.config.ppo_mini_batch_size)

        on_policy = len(mini_batches) == 1 and self.config.ppo_epochs == 1

        metrics = {}
        if (not vimpo_mode["use_vimpo_loss"]) or vimpo_mode["use_grpo_actor"] or vimpo_mode["use_ppo_actor"]:
            metrics["actor/pg_loss"] = 0.0
        if self.config.use_kl_loss:
            metrics["actor/kl_loss"] = 0.0
        if vimpo_mode["use_vimpo_loss"]:
            metrics["actor/vimpo_kl_agg"] = 0.0
        vimpo_terminal_stats = None
        vimpo_adv_stats = None
        vimpo_kl_stats = None
        for _ in range(self.config.ppo_epochs):
            for batch_idx, mini_batch in enumerate(mini_batches):
                value_loss_norm_denom = self._compute_vimpo_value_loss_norm_denom(mini_batch, vimpo_mode)
                if self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    dp_group = torch.distributed.group.WORLD if torch.distributed.is_initialized() else None
                    micro_batches, _ = prepare_dynamic_batch(
                        mini_batch,
                        max_token_len=max_token_len,
                        dp_group=dp_group,
                    )
                else:
                    self.gradient_accumulation = (
                        self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    )
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()
                append_to_dict(
                    metrics,
                    {
                        "actor/num_micro_batches": float(len(micro_batches)),
                        "actor/mini_batch_items": float(len(mini_batch)),
                    },
                )

                for micro_batch in micro_batches:
                    micro_batch = micro_batch.to(get_device_id())
                    micro_batch_metrics = {}
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch, "pad_token_id": pad_token_id}
                    response_mask = model_inputs["response_mask"]

                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode

                    calculate_entropy = self.config.calculate_entropy or (entropy_coeff != 0)

                    if self.config.use_dynamic_bsz:
                        loss_scale_factor = response_mask.shape[0] / self.config.ppo_mini_batch_size
                    else:
                        loss_scale_factor = 1 / self.gradient_accumulation

                    # all return: (bsz, response_length)
                    outputs = self._forward_micro_batch(
                        model_inputs,
                        temperature=temperature,
                        calculate_entropy=calculate_entropy,
                        return_valid_response_logits=self._vimpo_forward_needs_valid_logits(vimpo_mode),
                    )
                    log_prob = outputs["log_probs"]
                    entropy = outputs["entropys"] if calculate_entropy else None
                    pg_loss = None
                    vimpo_kl_agg = None
                    vimpo_terminal_mse = None
                    detached_advantages = None
                    actor_coeff_for_loss = vimpo_mode["vimpo_actor_coeff"]
                    old_log_prob = None
                    rollout_is_weights = None
                    loss_mode = None
                    ref_log_prob = None
                    vimpo_token_kl = None
                    if vimpo_mode["use_vimpo_loss"]:
                        if vimpo_mode["value_type"] == "soft":
                            vimpo_terminal_mse, vimpo_kl_agg, vimpo_loss_metrics = self._compute_soft_vimpo_loss_for_micro_batch(
                                model_inputs=model_inputs,
                                log_prob=log_prob,
                                response_mask=response_mask,
                                vimpo_mode=vimpo_mode,
                                loss_agg_mode=loss_agg_mode,
                                value_loss_norm_denom=value_loss_norm_denom,
                            )
                            vimpo_terminal_stats = self._accumulate_soft_vimpo_terminal_stats(
                                stats=vimpo_terminal_stats,
                                log_prob=log_prob,
                                model_inputs=model_inputs,
                                response_mask=response_mask,
                                vimpo_mode=vimpo_mode,
                            )
                            vimpo_kl_stats = self._accumulate_vimpo_kl_stats(
                                stats=vimpo_kl_stats,
                                token_kl=model_inputs["vimpo_old_token_kl"],
                                response_mask=response_mask,
                            )
                        else:
                            (
                                vimpo_terminal_mse,
                                vimpo_kl_agg,
                                ref_log_prob,
                                vimpo_token_kl,
                                vimpo_loss_metrics,
                            ) = self._compute_raw_vimpo_loss_for_micro_batch(
                                model_inputs=model_inputs,
                                outputs=outputs,
                                log_prob=log_prob,
                                response_mask=response_mask,
                                temperature=temperature,
                                vimpo_mode=vimpo_mode,
                                loss_agg_mode=loss_agg_mode,
                                value_loss_norm_denom=value_loss_norm_denom,
                            )
                            vimpo_terminal_stats = self._accumulate_raw_vimpo_terminal_stats(
                                stats=vimpo_terminal_stats,
                                log_prob=log_prob,
                                ref_log_prob=ref_log_prob,
                                token_kl=vimpo_token_kl,
                                model_inputs=model_inputs,
                                response_mask=response_mask,
                                vimpo_mode=vimpo_mode,
                            )
                            vimpo_kl_stats = self._accumulate_vimpo_kl_stats(
                                stats=vimpo_kl_stats,
                                token_kl=vimpo_token_kl,
                                response_mask=response_mask,
                            )
                            if vimpo_mode["use_ppo_actor"]:
                                (
                                    detached_advantages,
                                    detached_values,
                                    raw_advantages,
                                    vimpo_ppo_metrics,
                                ) = self._compute_raw_vimpo_ppo_targets_for_micro_batch(
                                    log_prob=log_prob,
                                    ref_log_prob=ref_log_prob,
                                    vimpo_token_kl=vimpo_token_kl,
                                    model_inputs=model_inputs,
                                    response_mask=response_mask,
                                    vimpo_mode=vimpo_mode,
                                )
                                vimpo_adv_stats = self._accumulate_vimpo_adv_stats(
                                    stats=vimpo_adv_stats,
                                    advantages=detached_advantages,
                                    raw_advantages=raw_advantages,
                                    values=detached_values,
                                    response_mask=response_mask,
                                )
                                micro_batch_metrics.update(vimpo_ppo_metrics)
                        micro_batch_metrics.update(vimpo_loss_metrics)
                        metrics["actor/vimpo_kl_agg"] += vimpo_kl_agg.detach().item() * loss_scale_factor

                    if (not vimpo_mode["use_vimpo_loss"]) or vimpo_mode["use_grpo_actor"] or vimpo_mode["use_ppo_actor"]:
                        advantages = (
                            detached_advantages
                            if vimpo_mode["use_vimpo_loss"] and vimpo_mode["use_ppo_actor"]
                            else model_inputs["advantages"]
                        )
                        # for fully_async_policy
                        if hasattr(self.config, "use_rollout_log_probs") and self.config.use_rollout_log_probs:
                            old_log_prob = model_inputs["old_log_probs"]
                        else:
                            if on_policy:
                                old_log_prob = log_prob.detach()
                            else:
                                old_log_prob = model_inputs["old_log_probs"]

                        loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")

                        rollout_is_weights = model_inputs.get("rollout_is_weights", None)
                        policy_loss_fn = get_policy_loss_fn(loss_mode)
                        pg_loss, pg_metrics = policy_loss_fn(
                            old_log_prob=old_log_prob,
                            log_prob=log_prob,
                            advantages=advantages,
                            response_mask=response_mask,
                            loss_agg_mode=loss_agg_mode,
                            config=self.config,
                            rollout_is_weights=rollout_is_weights,
                        )
                        micro_batch_metrics.update(pg_metrics)

                        rollout_log_prob = model_inputs.get("rollout_log_probs", None)
                        if loss_mode != "bypass_mode" and rollout_log_prob is not None:
                            from verl.trainer.ppo.rollout_corr_helper import compute_rollout_corr_metrics_from_logprobs

                            rollout_corr_metrics = compute_rollout_corr_metrics_from_logprobs(
                                log_prob=log_prob,
                                rollout_log_prob=rollout_log_prob,
                                response_mask=response_mask,
                            )
                            micro_batch_metrics.update(rollout_corr_metrics)

                    if (
                        vimpo_mode["use_vimpo_loss"]
                        and vimpo_mode["value_type"] == "raw"
                        and vimpo_mode["vimpo_adaptive_actor_coeff"]
                        and loss_agg_mode == "token-mean"
                        and "valid_response_logits" in outputs
                    ):
                        from recipe.vimpo.vimpo_core_algos import (
                            compute_vimpo_adaptive_actor_coeff,
                            compute_vimpo_ppo_actor_logit_grad_stats,
                            compute_vimpo_value_logit_grad_stats,
                        )

                        scale_metrics = {}
                        value_rms = None
                        if vimpo_mode["vimpo_detach_kl"]:
                            value_stats = compute_vimpo_value_logit_grad_stats(
                                valid_response_logits=outputs["valid_response_logits"],
                                log_prob=log_prob,
                                ref_log_prob=ref_log_prob,
                                token_kl=vimpo_token_kl,
                                token_level_rewards=model_inputs["token_level_rewards"],
                                response_mask=response_mask,
                                reward_baseline=model_inputs["vimpo_reward_baseline"],
                                beta=vimpo_mode["vimpo_beta"],
                                value_loss_type=vimpo_mode["vimpo_value_loss_type"],
                                huber_delta=vimpo_mode["vimpo_value_huber_delta"],
                                vocab_chunk_size=vimpo_mode["vocab_chunk_size"],
                            )
                            value_rms = self._global_logit_grad_rms(value_stats)
                            if value_rms is not None:
                                scale_metrics["actor/vimpo_value_logit_grad_rms"] = value_rms.detach().item()
                        if (
                            pg_loss is not None
                            and old_log_prob is not None
                            and detached_advantages is not None
                            and loss_mode == "vanilla"
                        ):
                            clip_ratio = self.config.clip_ratio
                            actor_stats_unscaled = compute_vimpo_ppo_actor_logit_grad_stats(
                                valid_response_logits=outputs["valid_response_logits"],
                                log_prob=log_prob,
                                old_log_prob=old_log_prob,
                                advantages=detached_advantages,
                                response_mask=response_mask,
                                vimpo_actor_coeff=1.0,
                                token_mean_coeff_scale=self._token_mean_coeff_scale(response_mask),
                                clip_ratio_low=self.config.clip_ratio_low
                                if self.config.clip_ratio_low is not None
                                else clip_ratio,
                                clip_ratio_high=self.config.clip_ratio_high
                                if self.config.clip_ratio_high is not None
                                else clip_ratio,
                                clip_ratio_c=self.config.get("clip_ratio_c", 3.0),
                                rollout_is_weights=rollout_is_weights,
                                vocab_chunk_size=vimpo_mode["vocab_chunk_size"],
                            )
                            actor_rms_unscaled = self._global_logit_grad_rms(actor_stats_unscaled)
                            if actor_rms_unscaled is not None:
                                if vimpo_mode["vimpo_adaptive_actor_coeff"]:
                                    if value_rms is None:
                                        raise RuntimeError(
                                            "vimpo_adaptive_actor_coeff requires value logit-gradient RMS, but it was "
                                            "unavailable. Ensure vimpo_detach_kl=True and valid response logits exist."
                                        )
                                    if actor_rms_unscaled.detach().item() <= 0.0:
                                        actor_coeff_for_loss = 0.0
                                    else:
                                        (
                                            actor_coeff_for_loss,
                                            self._vimpo_adaptive_value_logit_grad_rms_ema,
                                            self._vimpo_adaptive_actor_logit_grad_rms_unscaled_ema,
                                        ) = compute_vimpo_adaptive_actor_coeff(
                                            value_logit_grad_rms=value_rms,
                                            actor_logit_grad_rms_unscaled=actor_rms_unscaled,
                                            value_logit_grad_rms_ema=self._vimpo_adaptive_value_logit_grad_rms_ema,
                                            actor_logit_grad_rms_unscaled_ema=(
                                                self._vimpo_adaptive_actor_logit_grad_rms_unscaled_ema
                                            ),
                                            alpha=vimpo_mode["vimpo_adaptive_actor_alpha"],
                                            ema_decay=vimpo_mode["vimpo_adaptive_actor_ema_decay"],
                                            eps=vimpo_mode["vimpo_adaptive_actor_eps"],
                                        )
                                actor_rms = actor_rms_unscaled * float(actor_coeff_for_loss)
                                scale_metrics["actor/vimpo_actor_logit_grad_rms_unscaled"] = (
                                    actor_rms_unscaled.detach().item()
                                )
                                scale_metrics["actor/vimpo_actor_logit_grad_rms"] = actor_rms.detach().item()
                            elif vimpo_mode["vimpo_adaptive_actor_coeff"]:
                                raise RuntimeError(
                                    "vimpo_adaptive_actor_coeff requires actor logit-gradient RMS, but it was unavailable."
                                )
                        elif vimpo_mode["vimpo_adaptive_actor_coeff"]:
                            raise NotImplementedError(
                                "vimpo_adaptive_actor_coeff requires PPO actor loss_mode='vanilla' with available "
                                "pg_loss, old_log_prob, and detached VIMPO PPO advantages."
                            )
                        scale_metrics["actor/vimpo_adaptive_actor_ema_value_logit_grad_rms"] = (
                            self._vimpo_adaptive_value_logit_grad_rms_ema
                        )
                        scale_metrics["actor/vimpo_adaptive_actor_ema_actor_logit_grad_rms_unscaled"] = (
                            self._vimpo_adaptive_actor_logit_grad_rms_unscaled_ema
                        )
                        micro_batch_metrics.update(scale_metrics)

                    if vimpo_mode["use_vimpo_loss"]:
                        from recipe.vimpo.vimpo_core_algos import compose_vimpo_policy_loss

                        policy_loss, vimpo_policy_metrics = compose_vimpo_policy_loss(
                            pg_loss=pg_loss,
                            vimpo_terminal_mse=vimpo_terminal_mse,
                            use_grpo_actor=vimpo_mode["use_grpo_actor"],
                            use_ppo_actor=vimpo_mode["use_ppo_actor"],
                            vimpo_actor_coeff=actor_coeff_for_loss,
                            device=log_prob.device,
                        )
                        if vimpo_mode["vimpo_adaptive_actor_coeff"] and "vimpo_actor_coeff" in vimpo_policy_metrics:
                            micro_batch_metrics["actor/vimpo_actor_coeff"] = vimpo_policy_metrics["vimpo_actor_coeff"]
                    else:
                        policy_loss = pg_loss if pg_loss is not None else torch.zeros([], device=log_prob.device)
                    if calculate_entropy and entropy is not None:
                        entropy_agg = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
                        micro_batch_metrics["actor/entropy"] = entropy_agg.detach().item()
                        if entropy_coeff != 0:
                            policy_loss -= entropy_agg * entropy_coeff

                    if self.config.use_kl_loss:
                        ref_log_prob = model_inputs["ref_log_prob"]
                        # compute kl loss
                        kld = kl_penalty(
                            logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type
                        )
                        kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        metrics["actor/kl_loss"] += kl_loss.detach().item() * loss_scale_factor
                        micro_batch_metrics["actor/kl_coef"] = self.config.kl_loss_coef

                    if self.config.use_dynamic_bsz:
                        # relative to the dynamic bsz
                        loss = policy_loss * loss_scale_factor
                    else:
                        loss = policy_loss * loss_scale_factor
                    if self.scaler is not None:
                        self.scaler.scale(loss).backward()
                    else:
                        loss.backward()

                    if pg_loss is not None:
                        metrics["actor/pg_loss"] += pg_loss.detach().item() * loss_scale_factor
                    append_to_dict(metrics, micro_batch_metrics)

                grad_norm = self._optimizer_step()
                mini_batch_metrics = {"actor/grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, mini_batch_metrics)
        if "actor/vimpo_value_logit_grad_rms" in metrics and "actor/vimpo_actor_logit_grad_rms" in metrics:
            value_rms_values = metrics["actor/vimpo_value_logit_grad_rms"]
            actor_rms_values = metrics["actor/vimpo_actor_logit_grad_rms"]
            if value_rms_values and actor_rms_values:
                value_rms_mean = sum(value_rms_values) / len(value_rms_values)
                actor_rms_mean = sum(actor_rms_values) / len(actor_rms_values)
                metrics["actor/vimpo_actor_to_value_logit_grad_rms_ratio"] = [
                    actor_rms_mean / (value_rms_mean + 1e-12)
                ]
        self._append_global_vimpo_metrics(
            metrics=metrics,
            terminal_stats=vimpo_terminal_stats,
            adv_stats=vimpo_adv_stats,
            kl_stats=vimpo_kl_stats,
            vimpo_mode=vimpo_mode,
        )
        self.actor_optimizer.zero_grad()
        return metrics
