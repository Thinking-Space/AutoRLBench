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

import json
import logging
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


def _write_actor_policy_debug(stage: str, payload: dict) -> str:
    out_dir = os.getenv(
        "VERL_ACTOR_POLICY_DEBUG_DIR",
        "./outputs/actor_policy_debug",
    )
    os.makedirs(out_dir, exist_ok=True)
    exp_name = str(payload.get("experiment_name", "unknown_exp"))
    path = os.path.join(out_dir, f"{exp_name}.{stage}.rank{payload.get('rank', 'x')}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, sort_keys=True, indent=2)
    return path


def _actor_adapter_state_payload(peft_model, extra: dict | None = None) -> dict:
    active_adapters = []
    if peft_model is not None:
        active_adapters_attr = getattr(peft_model, "active_adapters", [])
        if callable(active_adapters_attr):
            try:
                active_adapters_attr = active_adapters_attr()
            except (TypeError, ValueError):
                active_adapters_attr = []
        if active_adapters_attr is None:
            active_adapters = []
        elif isinstance(active_adapters_attr, str):
            active_adapters = [active_adapters_attr]
        else:
            try:
                active_adapters = list(active_adapters_attr)
            except TypeError:
                active_adapters = []

    payload = {
        "has_peft_config": hasattr(peft_model, "peft_config"),
        "active_adapters": active_adapters,
        "disable_adapter_layers": bool(getattr(getattr(peft_model, "base_model", None), "_disable_adapter", False)),
    }
    if extra:
        payload.update(extra)
    return payload


def _base_ref_probe_payload(stage: str, tensor: torch.Tensor, old_log_prob: torch.Tensor | None = None) -> dict:
    flat = tensor.reshape(-1)
    payload = {
        "stage": stage,
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "mean": float(tensor.mean().item()),
        "abs_mean": float(tensor.abs().mean().item()),
        "min": float(tensor.min().item()),
        "max": float(tensor.max().item()),
        "head": flat[:8].detach().cpu().tolist(),
    }
    if old_log_prob is not None:
        diff = old_log_prob - tensor
        payload.update(
            {
                "old_mean": float(old_log_prob.mean().item()),
                "diff_mean": float(diff.mean().item()),
                "diff_abs_mean": float(diff.abs().mean().item()),
            }
        )
    return payload


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
        self._grad_dump_done = False
        role = "Ref" if actor_optimizer is None else "Actor"

        self.use_remove_padding = self.config.get("use_remove_padding", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_remove_padding={self.use_remove_padding}")
        self.use_fused_kernels = self.config.get("use_fused_kernels", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_fused_kernels={self.use_fused_kernels}")

        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

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

        self._adalora_step = 0
        self._debug_base_kl_once = False
        self._debug_base_ref_flow_once = False

    def _forward_micro_batch(
        self, micro_batch, temperature, calculate_entropy=False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
        """
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
                        if not self.config.entropy_checkpointing:
                            entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)
                        else:
                            entropy_rmpad = torch.utils.checkpoint.checkpoint(
                                self.compute_entropy_from_logits, logits_rmpad
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
                full_log_probs = pad_input(
                    hidden_states=log_probs.unsqueeze(-1),
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )

                # only return response part:
                if calculate_entropy:
                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)

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

            return entropy, log_probs

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

        peft_model = getattr(self.actor_module, "_fsdp_wrapped_module", self.actor_module)
        peft_config = getattr(getattr(peft_model, "peft_config", {}), "get", lambda *_: None)("default")
        if peft_config is not None and str(getattr(peft_config, "peft_type", "")).endswith("ADALORA"):
            init_r = getattr(peft_config, "init_r", None)
            target_r = getattr(peft_config, "target_r", None)
            should_allocate = init_r is None or target_r is None or target_r < init_r
            if should_allocate:
                missing_grad_count = 0
                for name, param in peft_model.named_parameters():
                    if "lora_" in name and ".default" in name and param.grad is None:
                        # AdaLoRA rank allocation reads gradients even after optimizer.step().
                        param.grad = torch.zeros_like(param)
                        missing_grad_count += 1
                if missing_grad_count and torch.distributed.get_rank() == 0:
                    print(f"Filled {missing_grad_count} missing AdaLoRA grads with zeros before rank allocation.")
                peft_model.base_model.update_and_allocate(self._adalora_step)
                self._adalora_step += 1
            elif self._adalora_step == 0 and torch.distributed.get_rank() == 0:
                print("Skipping AdaLoRA rank allocation because target_r >= init_r; rank schedule is a no-op.")
                self._adalora_step += 1
        return grad_norm

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_prob(self, data: DataProto, calculate_entropy=False) -> torch.Tensor:
        if not hasattr(self, "_debug_actor_compute_log_prob_once"):
            self._debug_actor_compute_log_prob_once = False
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        # set to eval
        self.actor_module.eval()
        if not self._debug_actor_compute_log_prob_once and torch.distributed.get_rank() == 0:
            _write_actor_policy_debug(
                "actor_compute_log_prob",
                {
                    "experiment_name": str(data.meta_info.get("experiment_name", "unknown_exp")),
                    "rank": int(torch.distributed.get_rank()),
                    **_actor_adapter_state_payload(self.actor_module),
                },
            )
            self._debug_actor_compute_log_prob_once = True

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        if use_dynamic_bsz:
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
        else:
            micro_batches = data.split(micro_batch_size)

        log_probs_lst = []
        entropy_lst = []
        for micro_batch in micro_batches:
            micro_batch = micro_batch.to(get_device_id())
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            with torch.no_grad():
                entropy, log_probs = self._forward_micro_batch(
                    model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
                )
            log_probs_lst.append(log_probs)
            if calculate_entropy:
                entropy_lst.append(entropy)

        log_probs = torch.concat(log_probs_lst, dim=0)
        entropys = None
        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)

        if use_dynamic_bsz:
            log_probs = restore_dynamic_batch(log_probs, batch_idx_list)
            if calculate_entropy:
                entropys = restore_dynamic_batch(entropys, batch_idx_list)

        return log_probs, entropys

    def _maybe_dump_full_grads(self) -> None:
        dump_path = os.getenv("VERL_FULL_GRAD_DUMP_PATH")
        if not dump_path or self._grad_dump_done or not isinstance(self.actor_module, FSDP):
            return

        requested = os.getenv("VERL_FULL_GRAD_LAYERS", "").strip()
        selected = {name.strip() for name in requested.split(",") if name.strip()}
        grad_tensors = {}

        with FSDP.summon_full_params(self.actor_module, writeback=False, with_grads=True):
            for name, param in self.actor_module.named_parameters():
                if not name.endswith(".weight"):
                    continue
                if selected and name not in selected:
                    continue
                if param.grad is None:
                    continue
                grad_tensors[name] = param.grad.detach().cpu().float().clone()

        if torch.distributed.get_rank() == 0:
            parent = os.path.dirname(dump_path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            torch.save(grad_tensors, dump_path)
            print(f"Saved {len(grad_tensors)} full gradients to {dump_path}")

        torch.distributed.barrier()
        self._grad_dump_done = True

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()
        if not hasattr(self, "_debug_actor_update_policy_once"):
            self._debug_actor_update_policy_once = False
        if not self._debug_actor_update_policy_once and torch.distributed.get_rank() == 0:
            _write_actor_policy_debug(
                "actor_update_policy",
                {
                    "experiment_name": str(data.meta_info.get("experiment_name", "unknown_exp")),
                    "rank": int(torch.distributed.get_rank()),
                    **_actor_adapter_state_payload(self.actor_module),
                },
            )
            self._debug_actor_update_policy_once = True

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error

        select_keys = [
            "responses",
            "response_mask",
            "input_ids",
            "attention_mask",
            "position_ids",
            "old_log_probs",
            "advantages",
        ]
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")
        safe_base_ref_log_prob = None
        if "base_ref_log_prob" in data.batch.keys():
            # TensorDict.select() corrupts this tensor in the current training path,
            # so carry a materialized copy across the selection boundary.
            safe_base_ref_log_prob = data.batch["base_ref_log_prob"].clone().contiguous()
        # Include pre-computed IS weights if present in batch
        # Weights are computed centrally in trainer and added to batch when algorithm.rollout_is=True
        if "rollout_is_weights" in data.batch.keys():
            select_keys.append("rollout_is_weights")
        # Include rollout_log_probs for computing rollout_corr metrics in bypass mode
        if "rollout_log_probs" in data.batch.keys():
            select_keys.append("rollout_log_probs")

        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)
        if safe_base_ref_log_prob is not None:
            data.batch["base_ref_log_prob"] = safe_base_ref_log_prob
        if (
            not self._debug_base_ref_flow_once
            and torch.distributed.get_rank() == 0
            and "base_ref_log_prob" in data.batch.keys()
        ):
            _write_base_kl_debug(
                "actor_update_probe_after_select",
                {
                    "experiment_name": str(data.meta_info.get("experiment_name", "unknown_exp")),
                    "rank": int(torch.distributed.get_rank()),
                    **_base_ref_probe_payload(
                        "after_select",
                        data.batch["base_ref_log_prob"],
                        data.batch.get("old_log_probs"),
                    ),
                },
            )

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        mini_batches = data.split(self.config.ppo_mini_batch_size)

        on_policy = len(mini_batches) == 1 and self.config.ppo_epochs == 1

        metrics = {}
        for _ in range(self.config.ppo_epochs):
            for batch_idx, mini_batch in enumerate(mini_batches):
                if (
                    not self._debug_base_ref_flow_once
                    and torch.distributed.get_rank() == 0
                    and "base_ref_log_prob" in mini_batch.batch.keys()
                ):
                    _write_base_kl_debug(
                        "actor_update_probe_mini_batch",
                        {
                            "experiment_name": str(data.meta_info.get("experiment_name", "unknown_exp")),
                            "rank": int(torch.distributed.get_rank()),
                            "mini_batch_idx": int(batch_idx),
                            **_base_ref_probe_payload(
                                "mini_batch_before_dynamic",
                                mini_batch.batch["base_ref_log_prob"],
                                mini_batch.batch.get("old_log_probs"),
                            ),
                        },
                    )
                if self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = (
                        self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    )
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()

                for micro_idx, micro_batch in enumerate(micro_batches):
                    if (
                        not self._debug_base_ref_flow_once
                        and torch.distributed.get_rank() == 0
                        and "base_ref_log_prob" in micro_batch.batch.keys()
                    ):
                        _write_base_kl_debug(
                            "actor_update_probe_micro_before_to",
                            {
                                "experiment_name": str(data.meta_info.get("experiment_name", "unknown_exp")),
                                "rank": int(torch.distributed.get_rank()),
                                "mini_batch_idx": int(batch_idx),
                                "micro_batch_idx": int(micro_idx),
                                **_base_ref_probe_payload(
                                    "micro_batch_before_to",
                                    micro_batch.batch["base_ref_log_prob"],
                                    micro_batch.batch.get("old_log_probs"),
                                ),
                            },
                        )
                    micro_batch = micro_batch.to(get_device_id())
                    micro_batch_metrics = {}
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                    if (
                        not self._debug_base_ref_flow_once
                        and torch.distributed.get_rank() == 0
                        and "base_ref_log_prob" in model_inputs
                    ):
                        _write_base_kl_debug(
                            "actor_update_probe_micro_after_to",
                            {
                                "experiment_name": str(data.meta_info.get("experiment_name", "unknown_exp")),
                                "rank": int(torch.distributed.get_rank()),
                                "mini_batch_idx": int(batch_idx),
                                "micro_batch_idx": int(micro_idx),
                                **_base_ref_probe_payload(
                                    "micro_batch_after_to",
                                    model_inputs["base_ref_log_prob"],
                                    model_inputs.get("old_log_probs"),
                                ),
                            },
                        )
                        self._debug_base_ref_flow_once = True
                    response_mask = model_inputs["response_mask"]
                    old_log_prob = model_inputs["old_log_probs"]
                    advantages = model_inputs["advantages"]

                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode

                    calculate_entropy = self.config.calculate_entropy or (entropy_coeff != 0)

                    if self.config.use_dynamic_bsz:
                        loss_scale_factor = response_mask.shape[0] / self.config.ppo_mini_batch_size
                    else:
                        loss_scale_factor = 1 / self.gradient_accumulation

                    # all return: (bsz, response_length)
                    entropy, log_prob = self._forward_micro_batch(
                        model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
                    )

                    # Keep rollout/eval-mode log probs for diagnostics such as base-model KL.
                    rollout_old_log_prob = model_inputs["old_log_probs"]
                    # for fully_async_policy recipe
                    if hasattr(self.config, "use_rollout_log_probs") and self.config.use_rollout_log_probs:
                        old_log_prob = rollout_old_log_prob
                    else:
                        if on_policy:
                            old_log_prob = log_prob.detach()
                        else:
                            old_log_prob = rollout_old_log_prob

                    loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")
                    # vanilla -> verl.trainer.ppo.core_algos.compute_policy_loss_vanilla

                    # Extract pre-computed rollout correction weights if present
                    # Weights are computed centrally in trainer and added when algorithm.rollout_is=True
                    rollout_is_weights = model_inputs.get("rollout_is_weights", None)

                    # gpg -> verl.trainer.ppo.core_algos.compute_policy_loss_gpg
                    # clip_cov -> verl.trainer.ppo.core_algos.compute_policy_loss_clip_cov
                    policy_loss_fn = get_policy_loss_fn(loss_mode)

                    # Compute policy loss (any function is expected to return 2 values)
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

                    # Skip if using pure rollout correction mode (metrics already in pg_metrics)
                    rollout_log_prob = model_inputs.get("rollout_log_probs", None)
                    if loss_mode != "rollout_correction" and rollout_log_prob is not None:
                        # Compute metrics using CURRENT policy π_θ vs π_rollout
                        # Tracks evolving off-policy gap as π_θ updates during mini-batch training
                        from verl.trainer.ppo.rollout_corr_helper import compute_rollout_corr_metrics_from_logprobs

                        rollout_corr_metrics = compute_rollout_corr_metrics_from_logprobs(
                            log_prob=log_prob,
                            rollout_log_prob=rollout_log_prob,
                            response_mask=response_mask,
                        )
                        micro_batch_metrics.update(rollout_corr_metrics)

                    policy_loss = pg_loss
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
                        micro_batch_metrics["actor/kl_loss"] = kl_loss.detach().item() * loss_scale_factor
                        micro_batch_metrics["actor/kl_coef"] = self.config.kl_loss_coef

                    if "base_ref_log_prob" in model_inputs:
                        base_ref_log_prob = model_inputs["base_ref_log_prob"]
                        # Compare rollout/eval-mode policy log probs against the frozen reference.
                        # This keeps base-model KL on the same inference-style footing as standard LoRA evals.
                        negative_approx_base_kl = torch.clamp(
                            rollout_old_log_prob - base_ref_log_prob, min=-20.0, max=20.0
                        )
                        base_model_kl = verl_F.masked_mean(-negative_approx_base_kl, response_mask)
                        micro_batch_metrics["actor/base_model_kl"] = base_model_kl.detach().item()
                        global_step = int(data.meta_info.get("global_steps", -1))
                        if not self._debug_base_kl_once and torch.distributed.get_rank() == 0:
                            self._debug_base_kl_once = True
                            rollout_vs_current = rollout_old_log_prob - log_prob.detach()
                            current_vs_base = log_prob.detach() - base_ref_log_prob
                            debug_payload = {
                                "stage": "actor_update",
                                "experiment_name": str(data.meta_info.get("experiment_name", "unknown_exp")),
                                "global_step": global_step,
                                "rank": int(torch.distributed.get_rank()),
                                "mini_batch_idx": int(batch_idx),
                                "micro_batch_idx": int(micro_idx),
                                **_debug_base_kl_payload(rollout_old_log_prob, base_ref_log_prob, response_mask),
                                "current_log_prob_mean": float(log_prob.detach().mean().item()),
                                "rollout_vs_current_diff_mean": float(rollout_vs_current.mean().item()),
                                "rollout_vs_current_diff_abs_mean": float(rollout_vs_current.abs().mean().item()),
                                "current_vs_base_diff_mean": float(current_vs_base.mean().item()),
                                "current_vs_base_diff_abs_mean": float(current_vs_base.abs().mean().item()),
                            }
                            debug_path = _write_base_kl_debug("actor_update", debug_payload)
                            print(f"[DEBUG actor-update base KL] wrote {debug_path}")

                    if self.config.use_dynamic_bsz:
                        # relative to the dynamic bsz
                        loss = policy_loss * loss_scale_factor
                    else:
                        loss = policy_loss * loss_scale_factor
                    if self.scaler is not None:
                        self.scaler.scale(loss).backward()
                    else:
                        loss.backward()

                    micro_batch_metrics["actor/pg_loss"] = pg_loss.detach().item() * loss_scale_factor
                    append_to_dict(metrics, micro_batch_metrics)

                self._maybe_dump_full_grads()
                grad_norm = self._optimizer_step()
                mini_batch_metrics = {"actor/grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, mini_batch_metrics)
        self.actor_optimizer.zero_grad()
        return metrics
