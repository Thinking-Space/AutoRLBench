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
from verl.utils.torch_functional import logprobs_for_ids_fp32
from verl.utils.ulysses import (
    gather_outputs_and_unpad,
    get_ulysses_sequence_parallel_rank,
    ulysses_pad,
    ulysses_pad_and_slice_inputs,
)
from verl.workers.actor import BasePPOActor
from verl.workers.actor.mt_opd import refresh_opd_advantage
from verl.workers.config import ActorConfig

__all__ = ["DataParallelPPOActor", "_compute_delta_opd_rm_scores"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _positive_temperature(value, *, label="temperature"):
    """Validate temperatures before passing them to eager/fused kernels."""

    if isinstance(value, bool):
        raise TypeError(f"{label} must be a positive finite number")
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a positive finite number, got {value!r}") from exc
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{label} must be a positive finite number, got {value!r}")
    return value


def _align_rmpad_topk_ids_for_ulysses(
    topk_ids: torch.Tensor,
    *,
    local_token_count: int,
    padding_size: int,
    sp_size: int,
    sp_rank: int,
) -> torch.Tensor:
    """Align externally supplied remove-padding top-k ids with local SP logits.

    ``student_top_k_ids`` is unpadded before the actor forward and therefore
    initially covers the full packed sequence. Ulysses pads that packed
    sequence at the end and gives each SP rank one contiguous slice. The token
    ids used to gather local student logits must receive the identical
    transformation before the gather; gathering them across ranks only after
    the local log-prob computation is too late.

    Already-local ids are accepted so this helper remains safe if a caller
    starts pre-sharding them upstream. Any other shape is rejected instead of
    relying on a later, opaque ``torch.gather`` size error.
    """
    if topk_ids.ndim != 2:
        raise ValueError(f"remove-padding top-k ids must be [tokens, K], got {tuple(topk_ids.shape)}")
    if sp_size <= 1:
        if topk_ids.size(0) != local_token_count:
            raise ValueError(
                "top-k ids do not align with actor logits: "
                f"ids={tuple(topk_ids.shape)}, local_token_count={local_token_count}"
            )
        return topk_ids
    if not 0 <= sp_rank < sp_size:
        raise ValueError(f"invalid Ulysses SP rank {sp_rank} for size {sp_size}")
    if topk_ids.size(0) == local_token_count:
        return topk_ids

    global_token_count = local_token_count * sp_size - padding_size
    if topk_ids.size(0) != global_token_count:
        raise ValueError(
            "global top-k ids do not align with Ulysses-partitioned actor logits: "
            f"ids={tuple(topk_ids.shape)}, local_token_count={local_token_count}, "
            f"padding_size={padding_size}, sp_size={sp_size}"
        )

    if padding_size:
        padding = torch.zeros(
            (padding_size, topk_ids.size(1)),
            dtype=topk_ids.dtype,
            device=topk_ids.device,
        )
        topk_ids = torch.cat((topk_ids, padding), dim=0)
    start = sp_rank * local_token_count
    return topk_ids.narrow(0, start, local_token_count).contiguous()


def _compute_delta_opd_rm_scores(
    student_logp: torch.Tensor,
    teacher_rl_logp: torch.Tensor,
    teacher_ref_logp: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute Direct-OPD dense top-k rewards.

    Direct-OPD uses the teacher RL/reference log-prob gap on student top-k ids,
    weighted by detached normalized student top-k probabilities.
    """
    # Direct/ExOPD are also reduction-heavy OPD paths.  Keep their deltas and
    # normalized student weights in FP32 even when the worker's model emits
    # BF16 log-probabilities; casting only the selected [B,T,K] tensors is
    # inexpensive and avoids silently rounding a small teacher/reference gap.
    student_logp = student_logp.to(dtype=torch.float32)
    teacher_rl_logp = teacher_rl_logp.to(dtype=torch.float32)
    teacher_ref_logp = teacher_ref_logp.to(dtype=torch.float32)

    if student_logp.shape != teacher_rl_logp.shape or student_logp.shape != teacher_ref_logp.shape:
        raise ValueError(
            "Direct-OPD tensors must have the same shape: "
            f"student_logp={tuple(student_logp.shape)}, "
            f"teacher_rl_logp={tuple(teacher_rl_logp.shape)}, "
            f"teacher_ref_logp={tuple(teacher_ref_logp.shape)}"
        )
    if student_logp.dim() != 3:
        raise ValueError(f"Direct-OPD expects [B, T, K] tensors, got {tuple(student_logp.shape)}")
    if valid_mask is not None and valid_mask.shape != student_logp.shape:
        raise ValueError(
            "Direct-OPD valid_mask must have the same shape as log-prob tensors: "
            f"valid_mask={tuple(valid_mask.shape)}, student_logp={tuple(student_logp.shape)}"
        )

    if valid_mask is not None:
        valid_mask = valid_mask.bool()
        masked_student_logp = student_logp.detach().masked_fill(~valid_mask, -torch.inf)
    else:
        masked_student_logp = student_logp.detach()

    delta = teacher_rl_logp.detach() - teacher_ref_logp.detach()
    if valid_mask is not None:
        delta = delta.masked_fill(~valid_mask, 0.0)

    student_weights = torch.softmax(masked_student_logp, dim=-1)
    student_weights = torch.nan_to_num(student_weights, nan=0.0, posinf=0.0, neginf=0.0)
    rm_scores = (student_weights * delta).detach()
    return rm_scores, delta.detach()


def _compute_exopd_rm_scores(
    opd_rm_scores: torch.Tensor,
    delta_rm_scores: torch.Tensor,
    exopd_lambda: float,
) -> torch.Tensor:
    """Combine standard OPD with Direct-OPD to get reward extrapolation.

    G-OPD reframes on-policy distillation as dense KL-constrained RL, where the
    optimum is

        log pi_theta = log pi* + (lambda - 1) (log pi* - log pi_ref)

    Standard OPD is the lambda = 1 special case. The extra term is exactly what
    Direct-OPD already computes -- ``student_weights * (log pi* - log pi_ref)``
    -- so extrapolation is a linear combination of two rewards this codebase
    already produces, not a new model path.

    lambda > 1 pushes the student past the teacher along the direction RL moved
    the teacher away from its own pre-RL base. lambda < 1 interpolates back
    toward the base.

    At lambda == 1 this returns ``opd_rm_scores`` unchanged, bit for bit, so the
    existing ``opd_kl`` recipe is recoverable exactly rather than approximately.
    """
    if opd_rm_scores.shape != delta_rm_scores.shape:
        raise ValueError(
            "ExOPD tensors must have the same shape: "
            f"opd_rm_scores={tuple(opd_rm_scores.shape)}, "
            f"delta_rm_scores={tuple(delta_rm_scores.shape)}"
        )
    if not math.isfinite(exopd_lambda):
        raise ValueError(f"exopd_lambda must be finite, got {exopd_lambda}")
    if exopd_lambda < 0:
        raise ValueError(f"exopd_lambda must be non-negative, got {exopd_lambda}")

    if exopd_lambda == 1.0:
        # identity by construction; skip the arithmetic so the tensor is
        # returned untouched and opd_kl stays exactly reproducible
        return opd_rm_scores.detach()

    return (opd_rm_scores + (exopd_lambda - 1.0) * delta_rm_scores).detach()


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
        # QAT: install the post-RoPE TurboQuant fake quantizer when a scheme is
        # configured.  Done here rather than via qat_kv.verl_plugin because
        # nothing in this deployment actually invokes that plugin's worker
        # patching (the mopd-pylibs sitecustomize only touches Ray tracing).
        # No-op when no turboquant_* scheme is set.
        try:
            from qat_kv.tq_scheme import ensure_actor_quantizer

            ensure_actor_quantizer(actor_module, config)
        except ImportError:
            pass
        self.actor_optimizer = actor_optimizer
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

        # MT-OPD M4: refresh the OPD reward against the current student on every inner
        # update. Off-policy only; a no-op when strictly on-policy.
        self._opd_refresh_advantage = bool(self.config.get("opd_refresh_advantage", False))
        self._opd_reward_weight_mode = str(
            self.config.get("opd_reward_weight_mode", "student_p")
        )
        if self._opd_refresh_advantage and torch.distributed.get_rank() == 0:
            print(f"{role} opd_refresh_advantage=True mode={self._opd_reward_weight_mode}")

    def _forward_micro_batch(
        self,
        micro_batch,
        temperature,
        calculate_entropy=False,
        top_k=0,
        student_top_k_ids=None,
        output_hidden_states=False,
        output_kv=False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
            topk_ids: # (bs, response_len, k)
            topk_log_probs: # (bs, response_len, k)

        When ``output_hidden_states`` is set, the per-block outputs are also
        published on ``self._micro_batch_hidden_states`` (they do not fit the
        4-tuple contract that five call sites depend on).  Block alignment uses
        this to score the forward it already ran instead of running a second
        one; see ``update_policy``.

        When ``output_kv`` is set and a TurboQuant fake quantizer is installed,
        the dequantized K/V actually fed to SDPA (the differentiable quantizer
        output) are published on ``self._micro_batch_kv`` under
        ``{layer_idx: (key, value)}``.  KV alignment uses this to score the
        quantizer's output against the frozen teacher's full-precision K/V
        without a second forward.
        """
        # Cleared unconditionally so a later caller can never pick up a stale
        # tensor from a previous micro-batch.
        self._micro_batch_hidden_states = None
        self._micro_batch_kv = None
        if output_hidden_states and self.use_remove_padding:
            raise ValueError(
                "output_hidden_states is not supported with use_remove_padding: "
                "the hidden states would be packed and could not be aligned "
                "against an unpacked response mask"
            )
        if self.use_remove_padding:
            # The unpadded path concatenates the batch into one sequence and
            # passes no attention_mask, relying on flash-attn varlen to split on
            # cu_seqlens.  The QAT hook is SDPA-based and never receives
            # cu_seqlens, so every token would attend across sequence
            # boundaries.  Measured on an 8-layer Qwen2 with two sequences: the
            # second sequence's final hidden state moves by 3.13 when packed,
            # while the quantization effect it is supposed to learn from is only
            # 0.19.  The corruption is 16x the signal, and nothing raises.
            from qat_kv.turboquant_attn_hook import get_installed_state

            if get_installed_state(self.actor_module) is not None:
                raise ValueError(
                    "use_remove_padding=True with the TurboQuant fake quantizer "
                    "installed: the packed forward has no attention mask, so "
                    "tokens attend across sequence boundaries and training is "
                    "silently wrong.  Disable one of the two."
                )
        # ``temperature`` is forwarded to custom fused model kernels when
        # enabled.  Guard it before model execution so invalid values cannot
        # trigger an opaque divide-by-zero/NaN inside a CUDA extension.
        temperature = _positive_temperature(temperature)
        response_length = micro_batch["responses"].size(-1)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch.keys():
            from verl.utils.model import extract_multi_modal_inputs

            multi_modal_inputs = extract_multi_modal_inputs(micro_batch["multi_modal_inputs"])

        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            entropy = None
            topk_ids = None
            topk_log_probs = None
            
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
                
                need_logits = top_k > 0

                if self.use_fused_kernels and not need_logits:
                    log_probs = output.log_probs.squeeze(0)  # (total_nnz,)
                    entropy_rmpad = output.entropy.squeeze(0)  # (total_nnz,)

                else:
                    logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
                    # Keep model logits in their native dtype.  The selected
                    # log-prob helper applies temperature after casting each
                    # row to FP32; mutating BF16 logits here would round the
                    # denominator before the numerically sensitive OPD delta.

                    # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                    inplace_backward = True
                    if calculate_entropy:
                        inplace_backward = False
                    
                    # Optimization: when top_k > 0, compute log_softmax once and gather both
                    # log_probs and topk_log_probs to avoid duplicate computation and gradient
                    # issues from inplace operations
                    need_topk = top_k > 0
                    if need_topk:
                        # Compute log_softmax once for both target and topk tokens
                        # Note: we don't use inplace_backward here to ensure correct gradients
                        # when both log_probs and topk_log_probs are needed
                        # Resolve candidate IDs below, then normalize only
                        # the selected labels/candidates in bounded FP32
                        # chunks.  A full FP32 [tokens, vocab] tensor is too
                        # expensive at the 16K context budget.
                        log_probs = None
                    else:
                        log_probs = logprobs_for_ids_fp32(
                            logits_rmpad, input_ids_rmpad_rolled, temperature=temperature
                        )

                    # compute entropy
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy_logits = (
                                logits_rmpad
                                if float(temperature) == 1.0
                                else logits_rmpad / float(temperature)
                            )
                            entropy_rmpad = self.compute_entropy_from_logits(entropy_logits)  # ((total_nnz / sp) + pad)
                        else:
                            entropy_logits = (
                                logits_rmpad
                                if float(temperature) == 1.0
                                else logits_rmpad / float(temperature)
                            )
                            entropy_rmpad = torch.utils.checkpoint.checkpoint(
                                self.compute_entropy_from_logits, entropy_logits
                            )
                    
                    if need_topk:
                        if student_top_k_ids is not None:
                            # Use specific IDs from the rollout. They are padded
                            # [batch, sequence/response, K] in the DataProto and
                            # must follow the same remove-padding transform as
                            # the actor input before indexing packed logits.
                            topk_ids = student_top_k_ids
                            if student_top_k_ids.ndim == 3:
                                if student_top_k_ids.size(-1) != top_k:
                                    raise ValueError(
                                        f"student top-k width {student_top_k_ids.size(-1)} does not match K={top_k}"
                                    )
                                if student_top_k_ids.shape[1] != seqlen:
                                    full_student_top_k_ids = torch.zeros(
                                        (batch_size, seqlen, top_k),
                                        dtype=student_top_k_ids.dtype,
                                        device=student_top_k_ids.device,
                                    )
                                    full_student_top_k_ids[:, -response_length - 1 : -1, :] = student_top_k_ids
                                    student_top_k_ids = full_student_top_k_ids

                                flat_ids = student_top_k_ids.reshape(-1, top_k)
                                topk_ids = flat_ids[indices]

                            topk_ids = _align_rmpad_topk_ids_for_ulysses(
                                topk_ids,
                                local_token_count=logits_rmpad.size(0),
                                padding_size=pad_size if self.use_ulysses_sp else 0,
                                sp_size=self.ulysses_sequence_parallel_size,
                                sp_rank=get_ulysses_sequence_parallel_rank() if self.use_ulysses_sp else 0,
                            )

                        else:
                            # Legacy/resample behavior already produces ids for
                            # the local SP logits and needs no extra slicing.
                            _, topk_ids = torch.topk(logits_rmpad, k=top_k, dim=-1)

                        selected_ids = torch.cat(
                            (input_ids_rmpad_rolled.unsqueeze(-1), topk_ids), dim=-1
                        )
                        selected_log_probs = logprobs_for_ids_fp32(
                            logits_rmpad, selected_ids, temperature=temperature
                        )
                        log_probs = selected_log_probs[..., 0]
                        topk_log_probs = selected_log_probs[..., 1:]

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
                    if top_k > 0:
                        topk_ids = gather_outputs_and_unpad(
                            topk_ids,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                        topk_log_probs = gather_outputs_and_unpad(
                            topk_log_probs,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
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
                
                if top_k > 0:
                    full_topk_ids = pad_input(
                        hidden_states=topk_ids,
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                    full_topk_log_probs = pad_input(
                        hidden_states=topk_log_probs,
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )

                # only return response part:
                if calculate_entropy:
                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                
                if top_k > 0:
                    topk_ids = full_topk_ids[:, -response_length - 1 : -1, :]
                    topk_log_probs = full_topk_log_probs[:, -response_length - 1 : -1, :]

            else:  # not using rmpad and no ulysses sp
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                if output_hidden_states:
                    extra_args["output_hidden_states"] = True

                if output_kv:
                    from qat_kv.turboquant_attn_hook import get_installed_state

                    _tq = get_installed_state(self.actor_module)
                    if _tq is None:
                        raise RuntimeError(
                            "output_kv requested but no TurboQuant fake "
                            "quantizer is installed; the student's quantized "
                            "K/V cannot be captured"
                        )
                    # Run the forward inside the capture context so the
                    # dequantized K/V fed to SDPA are stashed per layer.  The
                    # capture context clears state.kv on entry and restores
                    # record_kv on exit.
                    with _tq.capturing_kv() as _cap:
                        output = self.actor_module(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            position_ids=position_ids,
                            **multi_modal_inputs,
                            use_cache=False,
                            **extra_args,
                        )  # prevent model thinks we are generating
                    self._micro_batch_kv = _cap
                else:
                    output = self.actor_module(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        **multi_modal_inputs,
                        use_cache=False,
                        **extra_args,
                    )  # prevent model thinks we are generating

                if output_hidden_states:
                    # index 0 is the embedding output, 1..L are the block outputs
                    self._micro_batch_hidden_states = output.hidden_states[1:]
                
                need_logits = top_k > 0
                if self.use_fused_kernels and not need_logits:
                    log_probs = output.log_probs[:, -response_length - 1 : -1]
                    entropy = output.entropy[:, -response_length - 1 : -1]  # (bsz, response_length)

                else:
                    logits = output.logits

                    logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
                    
                    # Optimization: when top_k > 0, compute log_softmax once and gather both
                    # log_probs and topk_log_probs to avoid duplicate computation
                    need_topk = top_k > 0
                    if need_topk:
                        # Resolve candidate IDs below, then normalize only
                        # selected rows in bounded FP32 chunks.  Keeping a
                        # complete FP32 [B,T,V] log-softmax would add several
                        # GiB at the 16K context budget.
                        log_probs = None
                    else:
                        log_probs = logprobs_for_ids_fp32(
                            logits, micro_batch["responses"], temperature=temperature
                        )
                    
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy_logits = (
                                logits
                                if float(temperature) == 1.0
                                else logits / float(temperature)
                            )
                            entropy = verl_F.entropy_from_logits(entropy_logits)  # (bsz, response_length)
                        else:
                            entropy_logits = (
                                logits
                                if float(temperature) == 1.0
                                else logits / float(temperature)
                            )
                            entropy = torch.utils.checkpoint.checkpoint(verl_F.entropy_from_logits, entropy_logits)
                    
                    if need_topk:
                        if student_top_k_ids is not None:
                             topk_ids = student_top_k_ids
                             # Ensure shape alignment if needed, but for non-rmpad (bsz, seq, k) should match logits (bsz, seq, vocab) dim 0,1
                        else:
                             _, topk_ids = torch.topk(logits, k=top_k, dim=-1)
                        
                        selected_ids = torch.cat(
                            (micro_batch["responses"].unsqueeze(-1), topk_ids), dim=-1
                        )
                        selected_log_probs = logprobs_for_ids_fp32(
                            logits, selected_ids, temperature=temperature
                        )
                        log_probs = selected_log_probs[..., 0]
                        topk_log_probs = selected_log_probs[..., 1:]

            return entropy, log_probs, topk_ids, topk_log_probs

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_probs_for_ids(self, data: DataProto) -> torch.Tensor:
        """Compute the log probability for specific token ids
        Args:
            data (DataProto): a DataProto containing input_ids, attention_mask, position_ids, responses, 
                             and target_ids (batch, response_len, k) in batch
        Returns:
            torch.Tensor: (batch, response_len, k) log probs for target_ids
        """
        # set to eval
        self.actor_module.eval()

        target_ids = data.batch["target_ids"]
        
        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids", "target_ids"]
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)
        
        if use_dynamic_bsz:
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
        else:
            micro_batches = data.split(micro_batch_size)

        topk_log_probs_lst = []
        top_k = target_ids.shape[-1]

        for micro_batch in micro_batches:
            micro_batch = micro_batch.to(get_device_id())
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            mb_target_ids = model_inputs["target_ids"]
            with torch.no_grad():
                # We reuse _forward_micro_batch. It returns (entropy, log_probs, topk_ids, topk_log_probs)
                _, _, _, topk_log_probs = self._forward_micro_batch(
                    model_inputs, temperature=temperature, calculate_entropy=False, 
                    top_k=top_k, student_top_k_ids=mb_target_ids
                )
            # Keep on GPU to avoid expensive CPU-GPU transfer for large top-k
            # topk_log_probs = topk_log_probs.to("cpu")
            topk_log_probs_lst.append(topk_log_probs)

        topk_log_probs_tensor = torch.concat(topk_log_probs_lst, dim=0)

        if use_dynamic_bsz:
            topk_log_probs_tensor = restore_dynamic_batch(topk_log_probs_tensor, batch_idx_list)

        return topk_log_probs_tensor

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_distillation_reward(self, data: DataProto) -> DataProto:
        """Compute the distillation reward (rm_scores) on GPU
        Args:
            data (DataProto): containing all necessary tensors for distillation reward calculation
        Returns:
            DataProto: containing rm_scores and other updated tensors (e.g., union_ids)
        """
        # Sub-phase marks.  compute_distillation_reward ran 9.4x above its
        # forward-pass floor on j22947 (143.5 s against 15.2 s for the same 8
        # micro-batches that compute_log_prob does in 37.6 s), so the cost is
        # after the forward, in the OPD reward math on the [B,T,K] candidate
        # tensors.  One timer per section, same cumulative style as `adv`.
        import time as _drtime
        _dr_t0 = _drtime.time()
        _dr_marks: dict[str, float] = {}

        # One-shot inventory of what Ray had to marshal to get here.  The
        # driver measures this call at 79.2 s while the worker does 17.3 s of
        # work, so the difference is transport -- but which tensors, and in
        # what dtype, was never measured.
        if not getattr(self, "_dr_inventory_logged", False):
            self._dr_inventory_logged = True
            rows, total = [], 0
            for _k, _v in data.batch.items():
                if not torch.is_tensor(_v):
                    continue
                _b = _v.numel() * _v.element_size()
                total += _b
                rows.append((_b, _k, tuple(_v.shape), str(_v.dtype)))
            rows.sort(reverse=True)
            print("[dr-inventory] DataProto tensors reaching compute_distillation_reward:", flush=True)
            for _b, _k, _sh, _dt in rows[:12]:
                print(f"[dr-inventory]   {_b/2**30:7.2f} GiB  {_k:<32} {_sh} {_dt}", flush=True)
            print(f"[dr-inventory]   TOTAL {total/2**30:.2f} GiB across {len(rows)} tensors", flush=True)

        # Set to eval mode for forward passes
        self.actor_module.eval()

        # 1. Extract parameters from meta_info
        top_k = data.meta_info.get("log_prob_top_k", 0)
        strategy = data.meta_info.get("top_k_strategy", "only_stu")
        kl_estimator = data.meta_info.get("kl_estimator", "k1")
        reward_weight_mode = data.meta_info.get("reward_weight_mode", "student_p")  # "student_p", "teacher_p", or "none"
        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]

        # 2. Compute Student Log Probs on Teacher IDs if needed
        # (This replaces the previous call to compute_log_probs_for_ids in ray_trainer)
        S_on_T = None
        if strategy in ["only_tch", "intersection", "union", "union-intersection"]:
            target_ids = data.batch["teacher_top_k_ids"]
            
            # Select keys for micro-batching
            has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
            select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
            non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []
            
            # We need to pass target_ids to _forward_micro_batch, but since we are micro-batching, 
            # we should split target_ids as well.
            mb_data = data.select(batch_keys=select_keys + ["teacher_top_k_ids"], 
                                 non_tensor_batch_keys=non_tensor_select_keys)
            
            if use_dynamic_bsz:
                max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
                micro_batches, batch_idx_list = prepare_dynamic_batch(mb_data, max_token_len=max_token_len)
            else:
                micro_batches = mb_data.split(micro_batch_size)

            S_on_T_lst = []
            for micro_batch in micro_batches:
                micro_batch = micro_batch.to(get_device_id())
                model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                mb_target_ids = model_inputs["teacher_top_k_ids"]
                with torch.no_grad():
                    _, _, _, topk_log_probs = self._forward_micro_batch(
                        model_inputs, temperature=temperature, calculate_entropy=False, 
                        top_k=top_k, student_top_k_ids=mb_target_ids
                    )
                S_on_T_lst.append(topk_log_probs)

            S_on_T = torch.concat(S_on_T_lst, dim=0)
            if use_dynamic_bsz:
                S_on_T = restore_dynamic_batch(S_on_T, batch_idx_list)

        _dr_marks["dr_m1_forward"] = _drtime.time() - _dr_t0

        # 3. Compute rm_scores on GPU
        # Move all necessary tensors to GPU (they should already be there if passed from fsdp_workers)
        device = get_device_id()
        S_ids = data.batch["student_top_k_ids"].to(device)
        # The teacher/student models may execute in BF16, but the dense OPD
        # delta, candidate softmax and reward accumulation are numerical
        # reductions and must remain FP32.  Keeping the selected [B,T,K]
        # tensors in FP32 is cheap compared with materializing full-vocab
        # logits and makes the reward dtype explicit to the policy loss.
        S_logp = data.batch["student_top_k_log_probs"].to(device=device, dtype=torch.float32)
        T_on_S = data.batch["teacher_on_student_log_probs"].to(device=device, dtype=torch.float32)
        
        T_ids = data.batch.get("teacher_top_k_ids", None)
        if T_ids is not None: T_ids = T_ids.to(device)
        T_logp = data.batch.get("teacher_top_k_log_probs", None)
        if T_logp is not None: T_logp = T_logp.to(device=device, dtype=torch.float32)
        overlap_mask = data.batch.get("overlap_mask", None)
        if overlap_mask is not None: overlap_mask = overlap_mask.to(device)

        # Padded response positions do not carry a valid student distribution.
        # Their top-k log-probs can be -inf; including them in the candidate
        # softmax makes the all-padding rows undefined and can turn a finite
        # OPD reward into NaN during backward.
        response_mask = data.batch.get("response_mask", None)
        if response_mask is None:
            raise KeyError("compute_distillation_reward requires response_mask")
        response_mask = response_mask.to(device=device, dtype=torch.bool)
        _dr_marks["dr_m2_tohost"] = _drtime.time() - _dr_t0
        if response_mask.ndim != S_logp.ndim - 1:
            raise ValueError(
                f"response_mask shape {tuple(response_mask.shape)} must align with "
                f"top-k tensors {tuple(S_logp.shape)}"
            )
        response_valid_mask = response_mask.unsqueeze(-1).expand_as(S_logp)

        def compute_reward_weights(S_logp, T_logp, valid_mask, weight_mode, normalize=True):
            """Compute weights for reward calculation.
            
            Args:
                S_logp: Student log probabilities (batch, seq, K)
                T_logp: Teacher log probabilities (batch, seq, K)
                valid_mask: Boolean mask for valid tokens (batch, seq, K)
                weight_mode: "student_p", "teacher_p", or "none"
                normalize: If True, apply softmax normalization across K dim.
                          If False, use raw probabilities (masked by valid_mask).
            
            Returns:
                Weights (batch, seq, K)
            """
            if weight_mode == "student_p":
                log_probs = S_logp
            elif weight_mode == "teacher_p":
                log_probs = T_logp
            elif weight_mode == "none":
                # 对于"none"模式，使用均匀分布
                log_probs = torch.zeros_like(S_logp)
            else:
                raise ValueError(f"Unknown reward_weight_mode: {weight_mode}")
            
            log_probs = torch.where(valid_mask, log_probs.float(), torch.full_like(log_probs.float(), -float('inf')))
            
            if normalize:
                norm_log_weights = log_probs - torch.logsumexp(log_probs, dim=-1, keepdim=True)
                weights = torch.exp(norm_log_weights)
            else:
                weights = torch.exp(log_probs)
            
            weights = torch.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
            
            return weights

        res_tensors = {}
        
        if strategy == "only_stu":
            valid_mask = response_valid_mask & torch.isfinite(S_logp) & torch.isfinite(T_on_S)
            kl_val = torch.where(valid_mask, S_logp - T_on_S, torch.zeros_like(S_logp))
            norm_weights = compute_reward_weights(S_logp, T_on_S, valid_mask, reward_weight_mode)
            rm_scores = torch.nan_to_num(-kl_val * norm_weights, nan=0.0, posinf=0.0, neginf=0.0)
            
        elif strategy == "only_tch":
            kl_val = torch.nan_to_num(S_on_T - T_logp, nan=0.0, posinf=0.0, neginf=0.0)
            valid_mask = torch.ones_like(S_on_T, dtype=torch.bool)
            norm_weights = compute_reward_weights(S_on_T, T_logp, valid_mask, reward_weight_mode)
            rm_scores = -kl_val * norm_weights
            res_tensors["union_top_k_ids"] = T_ids
            
        elif strategy == "intersection":
            valid_mask = overlap_mask.bool()
            kl_val = torch.nan_to_num(S_logp - T_on_S, nan=0.0, posinf=0.0, neginf=0.0)
            kl_val = torch.where(valid_mask, kl_val, torch.zeros_like(kl_val))
            norm_weights = compute_reward_weights(S_logp, T_on_S, valid_mask, reward_weight_mode)
            rm_scores = -kl_val * norm_weights
            
        elif strategy == "union":
            union_ids = torch.cat([S_ids, T_ids], dim=-1)
            S_logp_union = torch.cat([S_logp, S_on_T], dim=-1)
            T_logp_union = torch.cat([T_on_S, T_logp], dim=-1)
            
            T_in_S = data.batch["teacher_in_student_mask"].bool().to(device)
            valid_mask = torch.cat([
                torch.ones_like(S_ids, dtype=torch.bool),
                ~T_in_S
            ], dim=-1)
            
            kl_val = torch.nan_to_num(S_logp_union - T_logp_union, nan=0.0, posinf=0.0, neginf=0.0)
            kl_val = torch.where(valid_mask, kl_val, torch.zeros_like(kl_val))
            norm_weights = compute_reward_weights(S_logp_union, T_logp_union, valid_mask, reward_weight_mode)
            rm_scores = -kl_val * norm_weights
            
            # Use different keys to avoid conflict with batch's student_top_k_ids
            res_tensors["union_top_k_ids"] = union_ids
            res_tensors["union_top_k_log_probs"] = S_logp_union
            res_tensors["student_log_probs_on_teacher_ids"] = S_on_T
        
        elif strategy == "union-intersection":
            union_ids = torch.cat([S_ids, T_ids], dim=-1)
            S_logp_union = torch.cat([S_logp, S_on_T], dim=-1)
            T_logp_union = torch.cat([T_on_S, T_logp], dim=-1)

            S_in_T = overlap_mask.bool().to(device)
            T_in_S = data.batch["teacher_in_student_mask"].bool().to(device)
            valid_mask = torch.cat([
                ~S_in_T,    # S_ids is valid if not in T
                ~T_in_S     # T_ids is valid if not in S
            ], dim=-1)
                
            kl_val = torch.nan_to_num(S_logp_union - T_logp_union, nan=0.0, posinf=0.0, neginf=0.0)
            kl_val = torch.where(valid_mask, kl_val, torch.zeros_like(kl_val))
            norm_weights = compute_reward_weights(S_logp_union, T_logp_union, valid_mask, reward_weight_mode, normalize=False)
            rm_scores = -kl_val * norm_weights
            
            # Use different keys to avoid conflict with batch's student_top_k_ids
            res_tensors["union_top_k_ids"] = union_ids
            res_tensors["union_top_k_log_probs"] = S_logp_union
            res_tensors["student_log_probs_on_teacher_ids"] = S_on_T
            
        # Position ramp on the OPD reward.
        #
        # The quantization damage is not uniform along the response: measured
        # between the quantized and full-precision arms, the gap in
        # val-topk/student_p_sum_intersection grows 60x from the first 1k
        # response tokens (-0.00007) to the last 1k (-0.00417), because every
        # decode step reads the quantized KV of everything before it.  A flat
        # reward spends most of its weight where there is almost nothing wrong.
        #
        # QAT_OPD_POSITION_RAMP="lo,hi" scales the reward linearly from lo at
        # the first response token to hi at the last.  Note this also multiplies
        # the *mean* reward by (lo+hi)/2, which is an effective learning-rate
        # change on top of the reweighting; QAT_OPD_POSITION_RAMP_NORMALIZE=1
        # divides it out so only the shape changes.
        _ramp = os.environ.get("QAT_OPD_POSITION_RAMP", "").strip()
        if _ramp:
            try:
                _lo, _hi = (float(x) for x in _ramp.split(","))
            except Exception as exc:
                raise ValueError(
                    f"QAT_OPD_POSITION_RAMP must be 'lo,hi', got {_ramp!r}"
                ) from exc
            _T = rm_scores.shape[1]
            _pos = torch.arange(_T, device=rm_scores.device, dtype=torch.float32)
            _pos = _pos / max(_T - 1, 1)
            _w = _lo + (_hi - _lo) * _pos
            if os.environ.get("QAT_OPD_POSITION_RAMP_NORMALIZE", "0") == "1":
                _w = _w / _w.mean()
            rm_scores = rm_scores * _w.view(1, _T, *([1] * (rm_scores.dim() - 2)))

        res_tensors["rm_scores"] = torch.nan_to_num(rm_scores, nan=0.0, posinf=0.0, neginf=0.0)
        _dr_marks["dr_m3_rmscore"] = _drtime.time() - _dr_t0
        # Cumulative marks, so the per-section cost is the successive
        # difference.  Printed rather than returned: the DataProto this builds
        # is consumed by a reward path that does not carry meta_info through.
        _m1 = _dr_marks.get("dr_m1_forward", 0.0)
        _m2 = _dr_marks.get("dr_m2_tohost", 0.0)
        _m3 = _dr_marks.get("dr_m3_rmscore", 0.0)
        print(
            f"[dr-profile] forward={_m1:.1f}s to_device={_m2 - _m1:.1f}s "
            f"rm_scores={_m3 - _m2:.1f}s total={_m3:.1f}s",
            flush=True,
        )
        return DataProto.from_dict(tensors=res_tensors)

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_delta_opd_reward(self, data: DataProto) -> DataProto:
        """Compute Direct-OPD dense top-k rewards on GPU."""
        device = get_device_id()
        student_logp = data.batch["student_top_k_log_probs"].to(device=device, dtype=torch.float32)
        teacher_rl_logp = data.batch["teacher_on_student_log_probs"].to(device=device, dtype=torch.float32)
        teacher_ref_logp = data.batch["teacher_ref_on_student_log_probs"].to(device=device, dtype=torch.float32)
        valid_mask = None
        strategy = data.meta_info.get("top_k_strategy", "only_stu")
        if strategy in {"intersection", "top_p_intersec", "top-p-intersec"} and "overlap_mask" in data.batch.keys():
            valid_mask = data.batch["overlap_mask"].to(device).bool()

        rm_scores, delta_log_ratio = _compute_delta_opd_rm_scores(
            student_logp=student_logp,
            teacher_rl_logp=teacher_rl_logp,
            teacher_ref_logp=teacher_ref_logp,
            valid_mask=valid_mask,
        )
        return DataProto.from_dict(tensors={"rm_scores": rm_scores, "delta_log_ratio": delta_log_ratio})

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_exopd_reward(self, data: DataProto) -> DataProto:
        """Compute reward-extrapolated OPD (ExOPD) dense top-k rewards.

        Consumes the standard OPD reward already in ``opd_rm_scores`` plus the
        Direct-OPD teacher/reference gap, and returns their lambda-weighted
        combination. ``exopd_lambda == 1`` reproduces standard OPD exactly.
        """
        device = get_device_id()
        student_logp = data.batch["student_top_k_log_probs"].to(device=device, dtype=torch.float32)
        teacher_rl_logp = data.batch["teacher_on_student_log_probs"].to(device=device, dtype=torch.float32)
        teacher_ref_logp = data.batch["teacher_ref_on_student_log_probs"].to(device=device, dtype=torch.float32)

        if "opd_rm_scores" not in data.batch.keys():
            raise ValueError(
                "reward_mode=exopd needs the standard OPD reward preserved as "
                "'opd_rm_scores' before the delta pass overwrites 'rm_scores'"
            )
        opd_rm_scores = data.batch["opd_rm_scores"].to(device=device, dtype=torch.float32)

        valid_mask = None
        strategy = data.meta_info.get("top_k_strategy", "only_stu")
        if strategy in {"intersection", "top_p_intersec", "top-p-intersec"} and "overlap_mask" in data.batch.keys():
            valid_mask = data.batch["overlap_mask"].to(device).bool()

        delta_rm_scores, delta_log_ratio = _compute_delta_opd_rm_scores(
            student_logp=student_logp,
            teacher_rl_logp=teacher_rl_logp,
            teacher_ref_logp=teacher_ref_logp,
            valid_mask=valid_mask,
        )
        exopd_lambda = float(data.meta_info.get("exopd_lambda", 1.0))
        rm_scores = _compute_exopd_rm_scores(
            opd_rm_scores=opd_rm_scores,
            delta_rm_scores=delta_rm_scores,
            exopd_lambda=exopd_lambda,
        )
        return DataProto.from_dict(tensors={
            "rm_scores": rm_scores,
            "delta_log_ratio": delta_log_ratio,
            "exopd_delta_rm_scores": delta_rm_scores,
        })

    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        elif isinstance(self.actor_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)

        if isinstance(grad_norm, DTensor):
            grad_norm = grad_norm.full_tensor()

        # if grad_norm is not finite, skip the update
        if not torch.isfinite(grad_norm):
            print(f"WARN: rank {torch.distributed.get_rank()} grad_norm is not finite: {grad_norm}")
            self.actor_optimizer.zero_grad()
        else:
            self.actor_optimizer.step()
        return grad_norm

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_prob(self, data: DataProto, calculate_entropy=False) -> torch.Tensor:
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

        top_k = data.meta_info.get("top_k", 0)
        print(f"In compute_log_prob, top_k: {top_k}")
        log_probs_lst = []
        entropy_lst = []
        topk_ids_lst = []
        topk_log_probs_lst = []

        for micro_batch in micro_batches:
            micro_batch = micro_batch.to(get_device_id())
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            with torch.no_grad():
                entropy, log_probs, topk_ids, topk_log_probs = self._forward_micro_batch(
                    model_inputs, temperature=temperature, calculate_entropy=calculate_entropy, top_k=top_k
                )
            # Keep on GPU to avoid expensive CPU-GPU transfer for large top-k
            # log_probs = log_probs.to("cpu")
            log_probs_lst.append(log_probs)
            if calculate_entropy:
                # entropy = entropy.to("cpu")
                entropy_lst.append(entropy)
            if top_k > 0:
                # topk_ids = topk_ids.to("cpu")
                # topk_log_probs = topk_log_probs.to("cpu")
                topk_ids_lst.append(topk_ids)
                topk_log_probs_lst.append(topk_log_probs)

        log_probs = torch.concat(log_probs_lst, dim=0)
        entropys = None
        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)
        
        topk_ids_tensor = None
        topk_log_probs_tensor = None
        if top_k > 0:
            topk_ids_tensor = torch.concat(topk_ids_lst, dim=0)
            topk_log_probs_tensor = torch.concat(topk_log_probs_lst, dim=0)

        if use_dynamic_bsz:
            log_probs = restore_dynamic_batch(log_probs, batch_idx_list)
            if calculate_entropy:
                entropys = restore_dynamic_batch(entropys, batch_idx_list)
            if top_k > 0:
                topk_ids_tensor = restore_dynamic_batch(topk_ids_tensor, batch_idx_list)
                topk_log_probs_tensor = restore_dynamic_batch(topk_log_probs_tensor, batch_idx_list)

        return log_probs, entropys, topk_ids_tensor, topk_log_probs_tensor

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        effective_kl_loss_coef = float(data.meta_info.get("actor_kl_loss_coef", self.config.kl_loss_coef))

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error

        # Which phase owns the step's memory peak?  verl only reports one
        # max-allocated figure for the whole step, which cannot distinguish
        # "the vLLM KV pool during rollout" from "actor activations during the
        # update" -- and the two have opposite fixes (lower
        # rollout.gpu_memory_utilization vs. shrink the micro-batch).
        # Reading the peak *on entry* attributes it: deliberately no
        # reset_peak_memory_stats() here, because that would clobber the
        # step-level perf/max_memory_allocated_gb this is meant to explain.
        _mem_metrics = {}
        if torch.cuda.is_available():
            _gib = 1024**3
            # Driver-level numbers are the ground truth.  vLLM allocates its
            # weights and KV pool through its own CuMemAllocator, which
            # torch.cuda.memory_allocated() cannot see at all -- so the
            # torch-reported peak understates real occupancy by whatever vLLM
            # is still holding, and headroom computed from it is not safe.
            _free, _total = torch.cuda.mem_get_info()
            _mem_metrics = {
                "qat/mem_peak_before_update_gb": torch.cuda.max_memory_allocated() / _gib,
                "qat/mem_resident_at_update_entry_gb": torch.cuda.memory_allocated() / _gib,
                "qat/mem_driver_used_at_update_entry_gb": (_total - _free) / _gib,
                "qat/mem_driver_total_gb": _total / _gib,
            }

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
        # Include pre-computed IS weights if present in batch
        # Weights are computed centrally in trainer and added to batch when algorithm.rollout_is=True
        if "rollout_is_weights" in data.batch.keys():
            select_keys.append("rollout_is_weights")

        if "format_mask" in data.batch.keys():
            select_keys.append("format_mask") # (bsz, 1)

        # MT-OPD M1: per-sequence loss weight that rebalances gradient share across
        # domains. A tensor rather than the domain labels themselves, so it rides the
        # same dynamic-batch reordering as advantages.
        if "domain_loss_weight" in data.batch.keys():
            select_keys.append("domain_loss_weight")
        
        # Include student_top_k_log_probs if present (for top-k distillation)
        if "student_top_k_log_probs" in data.batch.keys():
            select_keys.append("student_top_k_log_probs")

        # The rollout engine's own log-probs.  Under
        # ``bypass_old_logprob_for_rollout`` these are what ``old_log_probs``
        # aliases, but on-policy the loss overwrites ``old_log_prob`` with
        # ``log_prob_for_loss.detach()`` -- the same tensor it just computed --
        # which is why actor/ppo_kl and every rollout_corr/* metric come out
        # identically 0.  Carrying the rollout tensor separately gives a real
        # comparison: for a quantized rollout it is the running measurement of
        # whether the actor's fake quantizer reproduces the engine's real one.
        if "rollout_log_probs" in data.batch.keys():
            select_keys.append("rollout_log_probs")

        # MT-OPD M4: the frozen teacher's log-probs on the student's sampled ids. Needed
        # in the inner loop to rebuild the reward against the current student; it is the
        # cacheable half of the reward, so carrying it costs one tensor and no compute.
        if self._opd_refresh_advantage and "teacher_on_student_log_probs" in data.batch.keys():
            select_keys.append("teacher_on_student_log_probs")

        # Include student_top_k_ids if present (for fixing "apples-to-oranges" bug)
        if "student_top_k_ids" in data.batch.keys():
            select_keys.append("student_top_k_ids")

        # Include union_top_k_ids/log_probs for union strategy
        if "union_top_k_ids" in data.batch.keys():
            print("Now we are using union strategy, get union_top_k_ids")
            select_keys.append("union_top_k_ids")
            # now we don't need to store student_top_k_ids and student_top_k_log_probs for union strategy
            if "student_top_k_ids" in select_keys:
                select_keys.remove("student_top_k_ids")

        if "union_top_k_log_probs" in data.batch.keys():
            print("Now we are using union strategy, get union_top_k_log_probs")
            select_keys.append("union_top_k_log_probs")
            # now we don't need to store student_top_k_log_probs for union strategy
            if "student_top_k_log_probs" in select_keys:
                select_keys.remove("student_top_k_log_probs")   

        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        mini_batches = data.split(self.config.ppo_mini_batch_size)

        on_policy = len(mini_batches) == 1 and self.config.ppo_epochs == 1

        metrics = {}
        for _ in range(self.config.ppo_epochs):
            for batch_idx, mini_batch in enumerate(mini_batches):
                if self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = (
                        self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    )
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()
                grad_probe_enabled = os.getenv("QAT_GRAD_PROBE", "0") == "1"
                probe_pg_accum = None
                probe_ba_accum = None

                for micro_batch in micro_batches:
                    micro_batch = micro_batch.to(get_device_id())
                    micro_batch_metrics = {}
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                    response_mask = model_inputs["response_mask"]
                    old_log_prob = model_inputs["old_log_probs"]
                    advantages = model_inputs["advantages"]

                    # MT-OPD M1 is applied *after* the forward pass, because M4 may
                    # replace `advantages` wholesale and would otherwise discard the
                    # domain weighting. See the M1 block further down.
                    #
                    # MT-OPD M1: rebalance each domain's share of the gradient.
                    #
                    # Applied to the advantage rather than to response_mask. Scaling the
                    # mask would work -- token-mean is sum(loss*mask)/sum(mask), and the
                    # weights are normalised to token-weighted mean 1 so the denominator
                    # is unchanged, making the two numerically equivalent here -- but
                    # response_mask is also the averaging mask for ppo_kl and
                    # pg_clipfrac. Scaling it would turn those into domain-weighted
                    # means, and with IF's weight around 37x they would report almost
                    # only IF. The advantage carries the loss and nothing else.
                    # (moved below the forward pass -- see the M1 application site.)

                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode

                    if self.config.use_dynamic_bsz:
                        loss_scale_factor = response_mask.shape[0] / self.config.ppo_mini_batch_size
                    else:
                        loss_scale_factor = 1 / self.gradient_accumulation

                    # all return: (bsz, response_length)
                    calculate_entropy = False
                    if entropy_coeff != 0:
                        calculate_entropy = True

                    # Block alignment needs the quantized forward's block
                    # outputs.  Asking the policy forward for them is far
                    # cheaper than letting the alignment helper run its own
                    # second forward: that second forward builds a full
                    # autograd graph and is backwarded through, so reusing this
                    # one removes an entire forward *and* backward per
                    # micro-batch.  Gradient checkpointing already retains the
                    # block boundaries, so the extra retention is close to free.
                    block_align_coef = float(getattr(self.config, "block_align_coef", 0.0) or 0.0)
                    kv_align_coef = float(getattr(self.config, "kv_align_coef", 0.0) or 0.0)
                    # Packed (remove-padding) hidden states cannot be aligned
                    # against an unpacked response mask, so that path keeps the
                    # separate unpacked forward.
                    reuse_hidden_for_block_align = (
                        block_align_coef > 0 and not self.use_remove_padding
                    )
                    # KV capture follows the same rule: the captured per-layer
                    # K/V are packed tensors and cannot be aligned against the
                    # unpacked response mask.
                    capture_kv_for_align = kv_align_coef > 0 and not self.use_remove_padding
                    
                    # Check if we have 3D advantages (top-k sampling case)
                    # If so, we need to recompute top-k log probs for correct gradient
                    if advantages.dim() == 3:
                        top_k = advantages.shape[-1]
                        # For union strategy, use union_top_k_ids; otherwise use student_top_k_ids
                        student_top_k_ids = None
                        if "union_top_k_ids" in model_inputs:
                            student_top_k_ids = model_inputs["union_top_k_ids"]
                        elif "student_top_k_ids" in model_inputs:
                            student_top_k_ids = model_inputs["student_top_k_ids"]

                        entropy, log_prob, _, topk_log_probs = self._forward_micro_batch(
                            model_inputs, temperature=temperature, calculate_entropy=calculate_entropy,
                            top_k=top_k, student_top_k_ids=student_top_k_ids,
                            output_hidden_states=reuse_hidden_for_block_align,
                            output_kv=capture_kv_for_align,
                        )
                        log_prob_for_loss = topk_log_probs

                        # MT-OPD M4: refresh the reward against the *current* student.
                        #
                        # Only off-policy needs this. When on_policy, `advantages` was
                        # built from this same set of weights, so a refresh would be a
                        # no-op (and `old_log_prob` is this forward's own output).
                        # Off-policy, the stored reward describes the student as it was
                        # before the earlier inner updates -- and reward staleness is the
                        # one staleness term that is free to fix, since `T_on_S` is frozen
                        # and `topk_log_probs` was already computed for the gradient.
                        if (
                            self._opd_refresh_advantage
                            and not on_policy
                            and "teacher_on_student_log_probs" in model_inputs
                        ):
                            advantages = refresh_opd_advantage(
                                student_top_k_log_probs=topk_log_probs,
                                teacher_on_student_log_probs=model_inputs[
                                    "teacher_on_student_log_probs"
                                ],
                                response_mask=response_mask,
                                reward_weight_mode=self._opd_reward_weight_mode,
                            )
                            micro_batch_metrics["mt_opd/m4_advantage_refreshed"] = 1.0

                    else:
                        _, log_prob, *_ = self._forward_micro_batch(
                            model_inputs, temperature=temperature, calculate_entropy=calculate_entropy,
                            output_hidden_states=reuse_hidden_for_block_align,
                            output_kv=capture_kv_for_align,
                        )
                        log_prob_for_loss = log_prob

                    if "rollout_log_probs" in model_inputs:
                        from qat_kv.rollout_agreement import rollout_logprob_agreement

                        micro_batch_metrics.update(
                            rollout_logprob_agreement(
                                log_prob.detach(),
                                model_inputs["rollout_log_probs"],
                                response_mask,
                            )
                        )

                    # MT-OPD M1: rebalance each domain's share of the gradient. Applied
                    # here, after M4, so a refreshed advantage still carries the domain
                    # weighting.
                    #
                    # Applied to the advantage rather than to response_mask. Scaling the
                    # mask would work -- token-mean is sum(loss*mask)/sum(mask), and the
                    # weights are normalised to token-weighted mean 1 so the denominator
                    # is unchanged, making the two numerically equivalent here -- but
                    # response_mask is also the averaging mask for ppo_kl and
                    # pg_clipfrac. Scaling it would turn those into domain-weighted
                    # means, and with IF's weight around 37x they would report almost
                    # only IF. The advantage carries the loss and nothing else.
                    if "domain_loss_weight" in model_inputs:
                        _dw = model_inputs["domain_loss_weight"].to(advantages.dtype)
                        advantages = advantages * _dw.view(
                            -1, *([1] * (advantages.dim() - 1))
                        )

                    format_mask = None
                    if "format_mask" in model_inputs.keys():
                        format_mask = model_inputs["format_mask"]
            

                    # for fully_async_policy recipe
                    if hasattr(self.config, "use_rollout_log_probs") and self.config.use_rollout_log_probs:
                        old_log_prob = model_inputs["old_log_probs"]
                    else:
                        if on_policy:
                            print("on_policy")
                            # For on-policy (ppo_epochs=1), use current policy as "old"
                            # log_prob_for_loss is already 3D for top-k case
                            old_log_prob = log_prob_for_loss.detach()
                        else:
                            print("off_policy")
                            # For off-policy, use stored log probs
                            # For 3D top-k case, use stored log probs (union or student)
                            if advantages.dim() == 3:
                                if "union_top_k_log_probs" in model_inputs:
                                    old_log_prob = model_inputs["union_top_k_log_probs"]
                                elif "student_top_k_log_probs" in model_inputs:
                                    old_log_prob = model_inputs["student_top_k_log_probs"]
                                else:
                                    old_log_prob = model_inputs["old_log_probs"]
                            else:
                                old_log_prob = model_inputs["old_log_probs"]

                    loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")
                    # vanilla -> verl.trainer.ppo.core_algos.compute_policy_loss_vanilla

                    # Extract pre-computed rollout correction weights if present
                    # Weights are computed centrally in trainer and added when algorithm.rollout_is=True
                    rollout_is_weights = model_inputs.get("rollout_is_weights", None)

                    # NOTE: Both mismatch diagnostic metrics (PPL, KL, etc.) and IS weight metrics
                    # are computed centrally in ray_trainer.py for consistency and efficiency.
                    # This ensures metrics are computed uniformly across all batches at the trainer level
                    # and avoids redundant computation across workers and micro-batches.

                    # gpg -> verl.trainer.ppo.core_algos.compute_policy_loss_gpg
                    # clip_cov -> verl.trainer.ppo.core_algos.compute_policy_loss_clip_cov
                    policy_loss_fn = get_policy_loss_fn(loss_mode)

                    # Compute policy loss (any function is expected to return 2 values)
                    pg_loss, pg_metrics = policy_loss_fn(
                        old_log_prob=old_log_prob,
                        log_prob=log_prob_for_loss,  # 3D for top-k, 2D otherwise
                        advantages=advantages,
                        response_mask=response_mask,
                        loss_agg_mode=loss_agg_mode,
                        config=self.config,
                        rollout_is_weights=rollout_is_weights,
                        format_mask=format_mask,
                    )
                    micro_batch_metrics.update(pg_metrics)

                    if entropy_coeff != 0:
                        entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        # compute policy loss
                        policy_loss = pg_loss - entropy_loss * entropy_coeff
                    else:
                        policy_loss = pg_loss

                    if self.config.use_kl_loss:
                        ref_log_prob = model_inputs["ref_log_prob"]
                        # compute kl loss
                        kld = kl_penalty(
                            logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type
                        )
                        kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        policy_loss = policy_loss + kl_loss * effective_kl_loss_coef
                        micro_batch_metrics["actor/kl_loss"] = kl_loss.detach().item() * loss_scale_factor
                        micro_batch_metrics["actor/kl_coef"] = effective_kl_loss_coef

                    # Optional controlled objective split for short QAT probes.
                    # The normal training path is unchanged ("both").  "kl"
                    # keeps only the OPD/reverse-KL policy term, while "mse"
                    # keeps only the differentiable hidden-state alignment
                    # term.  This is intentionally environment-controlled so
                    # it cannot silently alter ordinary production runs.
                    probe_objective = os.getenv("QAT_PROBE_OBJECTIVE", "both").lower()
                    if probe_objective not in ("both", "kl", "mse"):
                        raise ValueError(
                            f"QAT_PROBE_OBJECTIVE must be both, kl, or mse; got {probe_objective!r}"
                        )
                    if probe_objective == "mse":
                        policy_loss = pg_loss * 0.0

                    block_loss = None

                    # --- QAT: align the quantized activations to this model's
                    # own full-precision forward.  Opt-in twice over: a
                    # positive coefficient AND an installed fake quantizer.
                    # Unlike the top-k OPD reward this is a direct
                    # differentiable loss, so it carries no detached-reward
                    # estimator bias.
                    if block_align_coef > 0:
                        from qat_kv.block_align_qat import (
                            BlockAlignConfig,
                            compute_quantization_block_alignment,
                        )
                        from qat_kv.turboquant_attn_hook import get_installed_state
                        from qat_kv.wa_scheme import get_installed_wa_state

                        # Block alignment scores per-block hidden states against
                        # a full-precision reference.  It does not care which
                        # axis is quantized -- NVFP4 weight/activation fake-quant
                        # perturbs those hidden states the same way KV
                        # quantization does -- so either state satisfies it.
                        tq_state = get_installed_state(self.actor_module)
                        if tq_state is None:
                            tq_state = get_installed_wa_state(self.actor_module)
                        if tq_state is None:
                            raise RuntimeError(
                                "actor.block_align_coef > 0 but no fake quantizer "
                                "is installed; the alignment target would equal "
                                "the student and the term would be a no-op. Set "
                                "--kv-scheme to a turboquant_* preset, or "
                                "QAT_WA_SCHEME to an nvfp4 tier."
                            )
                        fwd_inputs = {
                            k: model_inputs[k]
                            for k in ("input_ids", "attention_mask", "position_ids")
                            if k in model_inputs
                        }
                        # response_mask covers only the response suffix; hidden
                        # states cover prompt+response, so right-align it.
                        seq_len = fwd_inputs["input_ids"].shape[1]
                        full_mask = torch.zeros(
                            response_mask.shape[0],
                            seq_len,
                            dtype=response_mask.dtype,
                            device=response_mask.device,
                        )
                        full_mask[:, -response_mask.shape[1]:] = response_mask
                        ba_cfg = BlockAlignConfig(
                            layer_every=int(getattr(self.config, "block_align_layer_every", 4)),
                            keep_fraction=float(
                                getattr(self.config, "block_align_keep_fraction", 0.125)
                            ),
                            metric=str(getattr(self.config, "block_align_metric", "cosine")),
                            coef=block_align_coef,
                            last_only=bool(getattr(self.config, "block_align_last_only", False)),
                        )
                        student_hidden = (
                            self._micro_batch_hidden_states
                            if reuse_hidden_for_block_align
                            else None
                        )
                        if reuse_hidden_for_block_align and student_hidden is None:
                            raise RuntimeError(
                                "block alignment asked the policy forward for hidden "
                                "states but got none; refusing to silently fall back to "
                                "a second forward with different memory behaviour"
                            )
                        block_loss, block_metrics = compute_quantization_block_alignment(
                            self.actor_module,
                            tq_state,
                            fwd_inputs,
                            full_mask,
                            ba_cfg,
                            student_hidden=student_hidden,
                            # _forward_micro_batch runs under autocast, so the
                            # reference forward must too or the loss picks up a
                            # dtype floor unrelated to quantization.
                            autocast_dtype=torch.bfloat16,
                            device_type=self.device_name,
                            # theta_0, not the co-moving current weights; see
                            # fsdp_workers.block_align_reference for why.
                            frozen_model=getattr(self, "block_align_reference", None),
                        )
                        # Do not hold the graph alive past its use.
                        self._micro_batch_hidden_states = None
                        policy_loss = policy_loss + block_loss
                        micro_batch_metrics["actor/block_align_reference_static"] = (
                            1.0 if getattr(self, "block_align_reference", None) is not None else 0.0
                        )
                        micro_batch_metrics["actor/block_align_loss"] = (
                            block_loss.detach().item() * loss_scale_factor
                        )
                        for _k, _v in block_metrics.items():
                            if _k.startswith("block_align/layer_"):
                                micro_batch_metrics[f"actor/{_k}"] = _v

                    # --- QAT: align the student's *quantized K/V* to the frozen
                    # teacher's full-precision K/V.  This is the KV-cache-level
                    # analogue of the block-alignment term above: instead of
                    # anchoring the post-block representation it anchors the
                    # cache itself, which is where the quantization error is
                    # actually injected.  The student K/V are the differentiable
                    # dequantizer outputs captured by the policy forward
                    # (``_forward_micro_batch`` with ``output_kv``); the teacher
                    # K/V come from the frozen theta_0 copy with the quantizer
                    # bypassed.
                    if kv_align_coef > 0:
                        from qat_kv.kv_align import (
                            KVAlignConfig,
                            compute_kv_alignment,
                        )
                        from qat_kv.turboquant_attn_hook import get_installed_state

                        tq_state = get_installed_state(self.actor_module)
                        if tq_state is None:
                            raise RuntimeError(
                                "actor.kv_align_coef > 0 but no TurboQuant fake "
                                "quantizer is installed; the student's quantized "
                                "K/V cannot be captured"
                            )
                        if not capture_kv_for_align:
                            raise RuntimeError(
                                "kv_align_coef > 0 with use_remove_padding: the "
                                "captured K/V are packed and cannot be aligned "
                                "against the unpacked response mask; disable "
                                "one of the two"
                            )
                        fwd_inputs = {
                            k: model_inputs[k]
                            for k in ("input_ids", "attention_mask", "position_ids")
                            if k in model_inputs
                        }
                        seq_len = fwd_inputs["input_ids"].shape[1]
                        full_mask = torch.zeros(
                            response_mask.shape[0],
                            seq_len,
                            dtype=response_mask.dtype,
                            device=response_mask.device,
                        )
                        full_mask[:, -response_mask.shape[1]:] = response_mask
                        kv_cfg = KVAlignConfig(
                            layer_every=int(getattr(self.config, "kv_align_layer_every", 1)),
                            keep_fraction=float(
                                getattr(self.config, "kv_align_keep_fraction", 1.0)
                            ),
                            metric=str(getattr(self.config, "kv_align_metric", "nmse")),
                            coef=kv_align_coef,
                            align_keys=bool(getattr(self.config, "kv_align_align_keys", True)),
                            align_values=bool(getattr(self.config, "kv_align_align_values", True)),
                        )
                        student_kv = self._micro_batch_kv
                        if student_kv is None:
                            raise RuntimeError(
                                "kv alignment asked the policy forward for the "
                                "captured K/V but got none; refusing to silently "
                                "fall back to a second forward"
                            )
                        kv_loss, kv_metrics = compute_kv_alignment(
                            self.actor_module,
                            tq_state,
                            fwd_inputs,
                            full_mask,
                            kv_cfg,
                            student_kv=student_kv,
                            # _forward_micro_batch runs under autocast, so the
                            # reference forward must too or the loss picks up a
                            # dtype floor unrelated to quantization.
                            autocast_dtype=torch.bfloat16,
                            device_type=self.device_name,
                            frozen_model=getattr(self, "block_align_reference", None),
                        )
                        # Do not hold the graph alive past its use.
                        self._micro_batch_kv = None
                        policy_loss = policy_loss + kv_loss
                        micro_batch_metrics["actor/kv_align_loss"] = (
                            kv_loss.detach().item() * loss_scale_factor
                        )
                        for _k, _v in kv_metrics.items():
                            if _k.startswith("kv_align/layer_"):
                                micro_batch_metrics[f"actor/{_k}"] = _v

                    if probe_objective == "kl" and block_align_coef > 0:
                        # The alignment branch was computed only for a probe;
                        # do not include it in the KL-only objective.
                        policy_loss = pg_loss

                    # For a short coefficient probe, measure the two raw
                    # component gradients before the actual backward.  The
                    # values are computed on each FSDP shard and reduced to a
                    # global norm/cosine, so the reported ratio is comparable
                    # across world sizes.  This is off by default.
                    grad_probe_consumed_backward = False
                    if grad_probe_enabled:
                        # Component gradient norms and their cosine, used to set
                        # the block-align coefficient so the two terms enter the
                        # update at a chosen ratio.
                        #
                        # This cannot use torch.autograd.grad against
                        # module.parameters(): under FSDP those are FlatParameters
                        # while the graph is built on the unsharded views created
                        # during forward, so every entry comes back None.  With
                        # allow_unused=True that silently reports a norm of zero
                        # instead of raising, which is what the first version did.
                        # Each component uses its own forward/backward pair.  FSDP1
                        # frees the unsharded FlatParameter storage after a backward,
                        # so attempting two backwards through one retained graph makes
                        # the second pass read a zero-sized parameter view.  A fresh
                        # quantized forward for the alignment component is slower but
                        # follows FSDP's supported lifecycle.  Shards are disjoint, so
                        # summing squares and dot products across ranks gives the global
                        # quantities.  This diagnostic is off by default.
                        params = [p for p in self.actor_module.parameters() if p.requires_grad]

                        def _grad_snapshot(component):
                            if component is None or not component.requires_grad:
                                return None
                            self.actor_module.zero_grad(set_to_none=True)
                            component.backward()
                            return [
                                (p.grad.detach().float().clone() if p.grad is not None else None)
                                for p in params
                            ]

                        g_pg = _grad_snapshot(pg_loss)

                        # The block loss above shares the policy forward whose FSDP
                        # parameters were just resharded by g_pg.backward().  Drop that
                        # graph and rebuild only the quantized alignment branch.
                        policy_loss = None
                        block_loss = None
                        probe_block_loss, _ = compute_quantization_block_alignment(
                            self.actor_module,
                            tq_state,
                            fwd_inputs,
                            full_mask,
                            ba_cfg,
                            student_hidden=None,
                            autocast_dtype=torch.bfloat16,
                            device_type=self.device_name,
                            frozen_model=getattr(self, "block_align_reference", None),
                        )
                        g_ba = _grad_snapshot(probe_block_loss)
                        # Leave the last component gradient installed so the normal
                        # optimizer bookkeeping can finish.  Probe launchers use lr=0,
                        # hence this cannot update the model.  Crucially, do not run a
                        # third backward through either already-consumed graph below.
                        grad_probe_consumed_backward = True

                        # Accumulate the two component gradients using the same
                        # micro-batch scaling as the ordinary optimizer update.
                        # The per-micro metrics below diagnose variability; the
                        # probe_batch metrics emitted after this loop are the
                        # actual global-batch vectors relevant to a 1:1 weight.
                        if probe_pg_accum is None:
                            probe_pg_accum = [
                                (torch.zeros_like(a) if a is not None else None)
                                for a in g_pg
                            ]
                            probe_ba_accum = [
                                (torch.zeros_like(b) if b is not None else None)
                                for b in g_ba
                            ]
                        for i, (a, b) in enumerate(zip(g_pg, g_ba, strict=True)):
                            if a is not None:
                                probe_pg_accum[i].add_(a, alpha=loss_scale_factor)
                            if b is not None:
                                probe_ba_accum[i].add_(b, alpha=loss_scale_factor)

                        stats = torch.zeros(3, device=pg_loss.device, dtype=torch.float64)
                        for i in range(len(params)):
                            a = g_pg[i] if g_pg is not None else None
                            b = g_ba[i] if g_ba is not None else None
                            if a is not None:
                                stats[0] += a.pow(2).sum().double()
                            if b is not None:
                                stats[1] += b.pow(2).sum().double()
                            if a is not None and b is not None:
                                stats[2] += (a * b).sum().double()
                        if torch.distributed.is_available() and torch.distributed.is_initialized():
                            torch.distributed.all_reduce(stats, op=torch.distributed.ReduceOp.SUM)
                        pg_norm = stats[0].clamp_min(0).sqrt()
                        ba_norm = stats[1].clamp_min(0).sqrt()

                        # A zero here means the probe failed to see the graph, not
                        # that the gradient is small.  Say so rather than logging 0.
                        if float(pg_norm) <= 0.0:
                            raise RuntimeError(
                                "QAT_GRAD_PROBE: the policy-gradient component has a "
                                "zero global norm, which means no parameter received "
                                "a gradient.  The probe is not measuring anything."
                            )

                        micro_batch_metrics["probe/pg_grad_norm"] = pg_norm.item()
                        micro_batch_metrics["probe/mse_grad_norm"] = ba_norm.item()
                        micro_batch_metrics["probe/mse_over_pg_norm"] = (
                            (ba_norm / pg_norm).item()
                        )
                        micro_batch_metrics["probe/pg_mse_grad_cosine"] = (
                            (stats[2] / (pg_norm * ba_norm).clamp_min(1e-30)).item()
                        )
                        micro_batch_metrics["probe/pg_loss_raw"] = pg_loss.detach().item()
                        micro_batch_metrics["probe/mse_loss_raw"] = (
                            probe_block_loss.detach().item()
                        )

                    if not grad_probe_consumed_backward:
                        if self.config.use_dynamic_bsz:
                            # relative to the dynamic bsz
                            loss = policy_loss * loss_scale_factor
                        else:
                            loss = policy_loss * loss_scale_factor
                        loss.backward()

                    micro_batch_metrics["actor/pg_loss"] = pg_loss.detach().item() * loss_scale_factor
                    append_to_dict(metrics, micro_batch_metrics)

                grad_norm = self._optimizer_step()
                mini_batch_metrics = {"actor/grad_norm": grad_norm.detach().item()}
                if grad_probe_enabled:
                    batch_stats = torch.zeros(3, device=get_device_id(), dtype=torch.float64)
                    for a, b in zip(probe_pg_accum, probe_ba_accum, strict=True):
                        if a is not None:
                            batch_stats[0] += a.pow(2).sum().double()
                        if b is not None:
                            batch_stats[1] += b.pow(2).sum().double()
                        if a is not None and b is not None:
                            batch_stats[2] += (a * b).sum().double()
                    if torch.distributed.is_available() and torch.distributed.is_initialized():
                        torch.distributed.all_reduce(batch_stats, op=torch.distributed.ReduceOp.SUM)
                    batch_pg_norm = batch_stats[0].clamp_min(0).sqrt()
                    batch_ba_norm = batch_stats[1].clamp_min(0).sqrt()
                    mini_batch_metrics.update(
                        {
                            "probe_batch/pg_grad_norm": batch_pg_norm.item(),
                            "probe_batch/mse_grad_norm": batch_ba_norm.item(),
                            "probe_batch/mse_over_pg_norm": (
                                batch_ba_norm / batch_pg_norm.clamp_min(1e-30)
                            ).item(),
                            "probe_batch/pg_mse_grad_cosine": (
                                batch_stats[2]
                                / (batch_pg_norm * batch_ba_norm).clamp_min(1e-30)
                            ).item(),
                        }
                    )
                append_to_dict(metrics, mini_batch_metrics)
        self.actor_optimizer.zero_grad()
        if _mem_metrics:
            _free2, _total2 = torch.cuda.mem_get_info()
            _mem_metrics["qat/mem_peak_after_update_gb"] = (
                torch.cuda.max_memory_allocated() / 1024**3
            )
            _mem_metrics["qat/mem_driver_used_after_update_gb"] = (
                (_total2 - _free2) / 1024**3
            )
            append_to_dict(metrics, _mem_metrics)
        return metrics
