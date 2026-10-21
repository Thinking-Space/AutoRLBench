# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

"""MoE router z-loss must not depend on ``calculate_per_token_loss``.

Under context parallelism the Megatron bridge enables ``calculate_per_token_loss``,
which makes ``apply_z_loss`` scale the z-loss by the local token count, expecting
``finalize_model_grads`` to divide it back out by ``num_tokens``. verl's loss path
returns a 2-tuple ``(loss, output)``, so the schedule leaves ``num_tokens=0`` and
finalize skips that division -- leaving the per-token factor (~thousands)
uncancelled and blowing up the gradient. The flag must not change the gradient.
"""

import os

os.environ["NCCL_DEBUG"] = "WARN"

from functools import partial

import numpy as np
import ray
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen2MoeConfig

from verl import DataProto
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.utils import tensordict_utils as tu
from verl.utils.model import compute_position_id_with_mask, create_random_mask
from verl.workers.config import ActorConfig, HFModelConfig, McoreEngineConfig, McoreOptimizerConfig
from verl.workers.engine_workers import TrainingWorker, TrainingWorkerConfig
from verl.workers.utils.losses import ppo_loss
from verl.workers.utils.padding import left_right_2_no_padding

Z_LOSS_COEFF = 1e-3
# A small local model dir to borrow a tokenizer from (same convention as test_engine.py).
TOKENIZER_REF = os.environ["TOKENIZER_REF"]


def _create_tiny_moe_model(path):
    """Save a tiny Qwen3-MoE checkpoint so the router z-loss path is exercised. The
    engine's forward needs a tokenizer (pad_token_id), so we borrow one and size the
    vocab to match. Returns the vocab size."""
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_REF)
    config = Qwen2MoeConfig(
        vocab_size=len(tokenizer),
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=64,
        shared_expert_intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        num_experts=8,
        num_experts_per_tok=2,
        decoder_sparse_step=1,
        norm_topk_prob=True,
        max_position_embeddings=512,
        tie_word_embeddings=False,
    )
    AutoModelForCausalLM.from_config(config, torch_dtype=torch.bfloat16).save_pretrained(path)
    tokenizer.save_pretrained(path)
    return len(tokenizer)


def _build_data(vocab_size):
    batch_size, seqlen = 4, 32
    response_length = seqlen // 2
    torch.manual_seed(1)
    np.random.seed(1)
    input_ids = torch.randint(0, vocab_size, (batch_size, seqlen))
    attention_mask = create_random_mask(
        input_ids=input_ids, max_ratio_of_valid_token=0.8, max_ratio_of_left_padding=0.2, min_ratio_of_valid_token=0.6
    )
    position_ids = compute_position_id_with_mask(attention_mask)
    responses = input_ids[:, response_length:]
    response_mask = attention_mask[:, response_length:]
    data = DataProto.from_single_dict(
        {
            "input_ids": input_ids,
            "prompts": input_ids[:, :response_length],
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "responses": responses,
            "response_mask": response_mask,
            "old_log_probs": torch.randn_like(responses, dtype=torch.float32),
            "advantages": torch.randn_like(responses, dtype=torch.float32),
            "ref_log_prob": torch.randn_like(responses, dtype=torch.float32),
        },
        meta_info={"temperature": 1.0, "global_token_num": torch.sum(attention_mask, dim=-1).tolist()},
    )
    data_td = left_right_2_no_padding(data.to_tensordict())
    tu.assign_non_tensor(data_td, global_batch_size=data_td.shape[0])
    return data_td


def _z_only_loss(config, model_output, data, dp_group=None):
    loss, metrics = ppo_loss(config=config, model_output=model_output, data=data, dp_group=dp_group)
    return loss * 0.0, metrics


def _grad_norm(model_path, calculate_per_token_loss, data_td):
    """Run one PPO update through a Megatron MoE engine and return the grad norm.
    A fresh Ray session is used per call so the two single-GPU worker groups don't
    contend for the device."""
    config = TrainingWorkerConfig(
        model_type="language_model",
        model_config=HFModelConfig(path=model_path, use_remove_padding=True),
        engine_config=McoreEngineConfig(
            forward_only=False,
            use_mbridge=True,
            vanilla_mbridge=True,
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            context_parallel_size=2, expert_model_parallel_size=2,
            use_dynamic_bsz=True,
            use_remove_padding=True,
            max_token_len_per_gpu=2048,
            override_transformer_config={
                "moe_aux_loss_coeff": 0.0,
                "moe_z_loss_coeff": 0.001, "moe_shared_expert_gate": True, "moe_router_load_balancing_type": "none",
                "calculate_per_token_loss": calculate_per_token_loss,
            },
        ),
        optimizer_config=McoreOptimizerConfig(lr_decay_steps=10),
        checkpoint_config=None,
    )
    ray.init(
        num_cpus=4,
        num_gpus=2,
        object_store_memory=256 * 1024 * 1024,
        include_dashboard=False,
    )
    try:
        ray_cls = RayClassWithInitArgs(cls=ray.remote(TrainingWorker), config=config)
        pool = RayResourcePool(
            process_on_nodes=[2],
            max_colocate_count=1,
        )
        wg = RayWorkerGroup(
            resource_pool=pool,
            ray_cls_with_init=ray_cls,
        )
        wg.reset()
        actor_config = ActorConfig(strategy="megatron", rollout_n=1, ppo_micro_batch_size_per_gpu=-1)
        wg.set_loss_fn(partial(_z_only_loss, config=actor_config))
        metrics = tu.get(wg.train_batch(data_td).get(), "metrics")
        return float(metrics["grad_norm"])
    finally:
        ray.shutdown()


def test_moe_zloss_invariant_to_per_token_loss(tmp_path):
    model_path = str(tmp_path / "tiny_moe")
    vocab_size = _create_tiny_moe_model(model_path)
    data_td = _build_data(vocab_size)

    grad_norm_default = _grad_norm(model_path, calculate_per_token_loss=False, data_td=data_td)
    grad_norm_per_token = _grad_norm(model_path, calculate_per_token_loss=True, data_td=data_td)

    # With the bug, calculate_per_token_loss=True leaves the ~num_tokens per-token
    # factor uncancelled (finalize num_tokens=0), blowing grad_norm up by orders of
    # magnitude. The z-loss gradient must not depend on the flag.
    print(
    f"RESULT default={grad_norm_default} per_token={grad_norm_per_token}",
    flush=True
    )
    torch.testing.assert_close(grad_norm_per_token, grad_norm_default, rtol=5e-2, atol=1e-3)
