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
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import hashlib
import os
import uuid
from collections import defaultdict
from copy import deepcopy
from pprint import pprint

import numpy as np
import torch
from tqdm import tqdm

from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.metric_utils import compute_data_metrics, compute_throughout_metrics, compute_timing_metrics
from verl.trainer.ppo.ray_trainer import (
    AdvantageEstimator,
    RayPPOTrainer,
    apply_kl_penalty,
    compute_advantage,
    compute_response_mask,
)
from verl.trainer.ppo.reward import extract_reward
from verl.utils.checkpoint.checkpoint_manager import should_save_ckpt_esi
from verl.utils.metric import reduce_metrics
from verl.utils.profiler import marked_timer
from verl.utils.rollout_skip import RolloutSkip


def _get_reward_noise_config(config) -> dict:
    reward_noise = config.algorithm.get("reward_noise", {})
    return {
        "enable": bool(reward_noise.get("enable", False)),
        "flip_prob": float(reward_noise.get("flip_prob", 0.0)),
        "seed": int(reward_noise.get("seed", 0)),
        "mode": str(reward_noise.get("mode", "group_flip_binary_score")),
    }


def _reward_noise_score_key(non_tensor_batch: dict) -> str:
    if "score" in non_tensor_batch:
        return "score"
    if "acc" in non_tensor_batch:
        return "acc"
    raise KeyError("Reward noise requires clean 'score' or 'acc' in batch.non_tensor_batch.")


def _last_response_token_indices(batch: DataProto) -> torch.Tensor:
    if "response_mask" not in batch.batch:
        batch.batch["response_mask"] = compute_response_mask(batch)
    response_lengths = batch.batch["response_mask"].to(torch.long).sum(dim=-1)
    if torch.any(response_lengths <= 0):
        raise ValueError("Reward noise cannot be applied to empty responses.")
    return response_lengths - 1


def apply_group_reward_noise(
    batch: DataProto,
    reward_tensor: torch.Tensor,
    reward_noise_config: dict,
    global_steps: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Flip binary correctness rewards per prompt group for training only."""
    enabled = bool(reward_noise_config.get("enable", False))
    flip_prob = float(reward_noise_config.get("flip_prob", 0.0))
    mode = str(reward_noise_config.get("mode", "group_flip_binary_score"))
    if not enabled:
        metrics = {
            "reward_noise/flip_prob": flip_prob,
            "reward_noise/flipped_group_ratio": 0.0,
            "reward_noise/flipped_traj_ratio": 0.0,
        }
        try:
            score_key = _reward_noise_score_key(batch.non_tensor_batch)
        except KeyError:
            return reward_tensor, metrics
        clean_scores = np.asarray(batch.non_tensor_batch[score_key], dtype=np.float32)
        if clean_scores.size:
            metrics["reward_noise/clean_score_mean"] = float(np.mean(clean_scores))
            metrics["reward_noise/noisy_score_mean"] = float(np.mean(clean_scores))
        return reward_tensor, metrics

    if mode != "group_flip_binary_score":
        raise ValueError(f"Unsupported reward noise mode: {mode!r}.")
    if not 0.0 <= flip_prob <= 1.0:
        raise ValueError(f"algorithm.reward_noise.flip_prob must be in [0, 1], got {flip_prob}.")
    if "uid" not in batch.non_tensor_batch:
        raise KeyError("Reward noise requires prompt-group uid in batch.non_tensor_batch.")

    score_key = _reward_noise_score_key(batch.non_tensor_batch)
    clean_scores = np.asarray(batch.non_tensor_batch[score_key], dtype=np.float32)
    if clean_scores.shape[0] != len(batch):
        raise ValueError(f"{score_key} length {clean_scores.shape[0]} does not match batch length {len(batch)}.")

    uids = np.asarray(batch.non_tensor_batch["uid"], dtype=object)
    unique_uids = list(dict.fromkeys(uids.tolist()))
    seed = int(reward_noise_config.get("seed", 0))
    uid_to_flipped = {}
    for uid in unique_uids:
        digest = hashlib.blake2b(f"{seed}:{global_steps}:{uid}".encode("utf-8"), digest_size=8).digest()
        uid_sample = int.from_bytes(digest, byteorder="little", signed=False) / float(2**64)
        uid_to_flipped[uid] = bool(uid_sample < flip_prob) if flip_prob > 0 else False
    flipped = np.asarray([uid_to_flipped[uid] for uid in uids], dtype=bool)

    noisy_scores = np.where(flipped, 1.0 - clean_scores, clean_scores).astype(np.float32)
    last_token_indices = _last_response_token_indices(batch)
    noisy_reward_tensor = reward_tensor.clone()
    score_delta = torch.as_tensor(noisy_scores - clean_scores, dtype=noisy_reward_tensor.dtype, device=noisy_reward_tensor.device)
    row_indices = torch.arange(noisy_reward_tensor.shape[0], device=noisy_reward_tensor.device)
    noisy_reward_tensor[row_indices, last_token_indices.to(noisy_reward_tensor.device)] += score_delta

    batch.non_tensor_batch["reward_noise_flipped"] = flipped
    batch.non_tensor_batch[f"{score_key}_noisy"] = noisy_scores
    if score_key == "score" and "acc" in batch.non_tensor_batch:
        clean_acc = np.asarray(batch.non_tensor_batch["acc"], dtype=np.float32)
        batch.non_tensor_batch["acc_noisy"] = np.where(flipped, 1.0 - clean_acc, clean_acc).astype(np.float32)
    elif score_key == "acc" and "score" in batch.non_tensor_batch:
        clean_score = np.asarray(batch.non_tensor_batch["score"], dtype=np.float32)
        batch.non_tensor_batch["score_noisy"] = np.where(flipped, 1.0 - clean_score, clean_score).astype(np.float32)

    metrics = {
        "reward_noise/flip_prob": flip_prob,
        "reward_noise/flipped_group_ratio": float(np.mean(list(uid_to_flipped.values()))) if unique_uids else 0.0,
        "reward_noise/flipped_traj_ratio": float(np.mean(flipped)) if flipped.size else 0.0,
        "reward_noise/clean_score_mean": float(np.mean(clean_scores)) if clean_scores.size else 0.0,
        "reward_noise/noisy_score_mean": float(np.mean(noisy_scores)) if noisy_scores.size else 0.0,
    }
    return noisy_reward_tensor, metrics


def select_filter_metric_name(batch: DataProto, metric_name: str, reward_noise_config: dict) -> str:
    if not bool(reward_noise_config.get("enable", False)):
        return metric_name
    if metric_name in {"score", "acc"} and f"{metric_name}_noisy" in batch.non_tensor_batch:
        return f"{metric_name}_noisy"
    return metric_name


class RayDAPOTrainer(RayPPOTrainer):
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    def compute_kl_related_metrics(self, batch: DataProto, metrics: dict, timing_raw: dict):
        batch.batch["response_mask"] = compute_response_mask(batch)

        # recompute old_log_probs
        with marked_timer("old_log_prob", timing_raw, "blue"):
            old_log_prob, old_log_prob_mfu = self._compute_old_log_prob(batch)
            entropys = old_log_prob.batch["entropys"]
            response_masks = batch.batch["response_mask"]
            actor_config = self.config.actor_rollout_ref.actor
            entropy_agg = agg_loss(
                loss_mat=entropys,
                loss_mask=response_masks,
                loss_agg_mode=actor_config.loss_agg_mode,
                loss_scale_factor=actor_config.loss_scale_factor,
            )
            old_log_prob_metrics = {
                "actor/entropy": entropy_agg.detach().item(),
                "perf/mfu/actor_infer": old_log_prob_mfu,
            }
            metrics.update(old_log_prob_metrics)
            old_log_prob.batch.pop("entropys")
            batch = batch.union(old_log_prob)

        if self.use_reference_policy:
            # compute reference log_prob
            with marked_timer("ref", timing_raw, "olive"):
                ref_log_prob = self._compute_ref_log_prob(batch)
                batch = batch.union(ref_log_prob)

        return batch

    def _collect_true_accuracy_metrics(self, batch: DataProto, metrics: dict):
        scores = batch.non_tensor_batch.get("score")
        if scores is None:
            scores = batch.non_tensor_batch.get("acc")
        uids = batch.non_tensor_batch.get("uid")
        if scores is None or uids is None:
            return

        prompt_scores = defaultdict(list)
        all_scores = []
        for uid, score in zip(uids, scores, strict=True):
            value = float(score)
            prompt_scores[uid].append(value)
            all_scores.append(value)

        total_prompts = len(prompt_scores)
        if total_prompts == 0:
            return

        all_correct = sum(
            1 for grouped_scores in prompt_scores.values() if all(abs(s - 1.0) < 1e-6 for s in grouped_scores)
        )
        all_incorrect = sum(
            1 for grouped_scores in prompt_scores.values() if all(abs(s - 0.0) < 1e-6 for s in grouped_scores)
        )
        metrics.update(
            {
                "true_accuracy/score_mean": float(np.mean(all_scores)),
                "true_accuracy/score_std": float(np.std(all_scores)),
                "true_accuracy/all_correct_ratio": float(all_correct / total_prompts),
                "true_accuracy/all_incorrect_ratio": float(all_incorrect / total_prompts),
                "true_accuracy/all_correct_count": int(all_correct),
                "true_accuracy/all_incorrect_count": int(all_incorrect),
                "true_accuracy/total_prompts": int(total_prompts),
            }
        )

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0
        self.gen_steps = 0
        self.max_steps_duration = 0

        # load checkpoint before doing anything
        self._load_checkpoint()
        self.checkpoint_manager.update_weights()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        if self.config.actor_rollout_ref.rollout.get("skip_rollout", False):
            rollout_skip = RolloutSkip(self.config, self.async_rollout_manager)
            rollout_skip.wrap_generate_sequences()

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        self.gen_steps += 1
        last_val_metrics = None

        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        next_step_profile = False

        timing_raw = defaultdict(float)
        batch = None
        num_prompt_in_batch = 0
        num_gen_batches = 0
        current_epoch = self.global_steps // len(self.train_dataloader)

        for epoch in range(current_epoch, self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                    self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=False)
                metrics = {}

                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )

                new_batch: DataProto = DataProto.from_single_dict(batch_dict)
                new_batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature
                num_gen_batches += 1
                gen_batch = self._get_gen_batch(new_batch)
                gen_batch_output = gen_batch.repeat(
                    repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True
                )

                is_last_step = self.global_steps >= self.total_training_steps

                with marked_timer("step", timing_raw):
                    # generate a batch
                    with marked_timer("gen", timing_raw, "red"):
                        gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch_output)
                        timing_raw.update(gen_batch_output.meta_info["timing"])
                        gen_batch_output.meta_info.pop("timing", None)

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        with marked_timer("gen_max", timing_raw, "red"):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            gen_baseline_output = self.async_rollout_manager.generate_sequences(gen_baseline_batch)

                            new_batch = new_batch.union(gen_baseline_output)
                            # compute reward model score on new_batch
                            rm_scores = None
                            if self.use_rm and "rm_scores" not in new_batch.batch.keys():
                                rm_scores = self._compute_reward_colocate(new_batch)
                                new_batch = new_batch.union(rm_scores)
                            reward_baseline_tensor, _ = extract_reward(new_batch)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            keys_to_pop = set(gen_baseline_output.batch.keys())
                            if rm_scores is not None:
                                keys_to_pop.update(rm_scores.batch.keys())
                            new_batch.pop(batch_keys=list(keys_to_pop))

                            new_batch.batch["reward_baselines"] = reward_baseline_tensor

                            del rm_scores, gen_baseline_batch, gen_baseline_output

                    new_batch.non_tensor_batch["uid"] = np.array(
                        [str(uuid.uuid4()) for _ in range(len(new_batch.batch))], dtype=object
                    )
                    # repeat to align with repeated responses in rollout
                    new_batch = new_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    new_batch = new_batch.union(gen_batch_output)

                    if self.config.algorithm.use_kl_in_reward:
                        # We need these metrics for apply_kl_penalty if using kl in reward
                        new_batch = self.compute_kl_related_metrics(new_batch, metrics, timing_raw)
                        # otherwise, we will compute those after dynamic sampling

                    with marked_timer("reward", timing_raw, "yellow"):
                        # compute scores. Support both model and function-based.
                        # We first compute the scores using reward model. Then, we call reward_fn to combine
                        # the results from reward model and rule-based results.
                        if self.use_rm and "rm_scores" not in new_batch.batch.keys():
                            # we first compute reward model score
                            batch_reward = self._compute_reward_colocate(new_batch)
                            new_batch = new_batch.union(batch_reward)

                        # we combine with rule-based rm
                        reward_tensor, reward_extra_infos_dict = extract_reward(new_batch)

                        if reward_extra_infos_dict:
                            new_batch.non_tensor_batch.update(
                                {k: np.array(v) for k, v in reward_extra_infos_dict.items()}
                            )

                        reward_tensor, reward_noise_metrics = apply_group_reward_noise(
                            batch=new_batch,
                            reward_tensor=reward_tensor,
                            reward_noise_config=_get_reward_noise_config(self.config),
                            global_steps=self.global_steps,
                        )
                        metrics.update(reward_noise_metrics)

                        new_batch.batch["token_level_scores"] = reward_tensor

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            new_batch, kl_metrics = apply_kl_penalty(
                                new_batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(
                                kl_metrics
                            )  # TODO: This will be cleared if we use multiple genenration batches
                        else:
                            new_batch.batch["token_level_rewards"] = new_batch.batch["token_level_scores"]

                    self._collect_true_accuracy_metrics(new_batch, metrics)

                    if not self.config.algorithm.filter_groups.enable:
                        batch = new_batch
                    else:  # NOTE: When prompts after filtering is less than train batch size,
                        # we skip to the next generation batch
                        metric_name = select_filter_metric_name(
                            new_batch,
                            self.config.algorithm.filter_groups.metric,
                            _get_reward_noise_config(self.config),
                        )
                        if metric_name == "seq_final_reward":
                            # Turn to numpy for easier filtering
                            new_batch.non_tensor_batch["seq_final_reward"] = (
                                new_batch.batch["token_level_rewards"].sum(dim=-1).numpy()
                            )
                        elif metric_name == "seq_reward":
                            new_batch.non_tensor_batch["seq_reward"] = (
                                new_batch.batch["token_level_scores"].sum(dim=-1).numpy()
                            )

                        # Collect the sequence reward for each trajectory
                        prompt_uid2metric_vals = defaultdict(list)
                        for uid, metric_val in zip(
                            new_batch.non_tensor_batch["uid"], new_batch.non_tensor_batch[metric_name], strict=True
                        ):
                            prompt_uid2metric_vals[uid].append(metric_val)

                        prompt_uid2metric_std = {}
                        for prompt_uid, metric_vals in prompt_uid2metric_vals.items():
                            prompt_uid2metric_std[prompt_uid] = np.std(metric_vals)

                        kept_prompt_uids = [
                            uid
                            for uid, std in prompt_uid2metric_std.items()
                            if std > 0 or len(prompt_uid2metric_vals[uid]) == 1
                        ]
                        num_prompt_in_batch += len(kept_prompt_uids)

                        kept_traj_idxs = []
                        for idx, traj_from_prompt_uid in enumerate(new_batch.non_tensor_batch["uid"]):
                            if traj_from_prompt_uid in kept_prompt_uids:
                                kept_traj_idxs.append(idx)

                        new_batch = new_batch[kept_traj_idxs]
                        batch = new_batch if batch is None else DataProto.concat([batch, new_batch])

                        prompt_bsz = self.config.data.train_batch_size
                        if num_prompt_in_batch < prompt_bsz:
                            print(f"{num_prompt_in_batch=} < {prompt_bsz=}")
                            max_num_gen_batches = self.config.algorithm.filter_groups.max_num_gen_batches
                            if max_num_gen_batches <= 0 or num_gen_batches < max_num_gen_batches:
                                print(f"{num_gen_batches=}. Keep generating...")
                                self.gen_steps += 1
                                is_last_step = self.global_steps >= self.total_training_steps
                                continue
                            else:
                                raise ValueError(
                                    f"{num_gen_batches=} >= {max_num_gen_batches=}."
                                    + " Generated too many. Please check if your data are too difficult."
                                    + " You could also try set max_num_gen_batches=0 to enable endless trials."
                                )
                        else:
                            # Align the batch
                            traj_bsz = self.config.data.train_batch_size * self.config.actor_rollout_ref.rollout.n
                            batch = batch[:traj_bsz]

                    self.checkpoint_manager.sleep_replicas()

                    # === Updating ===
                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    # TODO: Decouple the DP balancing and mini-batching.
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    if not self.config.algorithm.use_kl_in_reward:
                        batch = self.compute_kl_related_metrics(batch, metrics, timing_raw)

                    # compute values
                    if self.use_critic:
                        with marked_timer("values", timing_raw, "cyan"):
                            values = self._compute_values(batch)
                            batch = batch.union(values)

                    # Compute rollout correction weights and off-policy metrics (inherited from RayPPOTrainer)
                    from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_and_add_to_batch

                    rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
                    if rollout_corr_config is not None and "rollout_log_probs" in batch.batch:
                        batch, is_metrics = compute_rollout_correction_and_add_to_batch(batch, rollout_corr_config)
                        # IS and off-policy metrics already have rollout_corr/ prefix
                        metrics.update(is_metrics)

                    with marked_timer("adv", timing_raw, "brown"):
                        # compute advantages, executed on the driver process
                        norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)
                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            config=self.config.algorithm,
                        )

                    # update critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, "pink"):
                            critic_output = self._update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with marked_timer("update_actor", timing_raw, "red"):
                            actor_output = self._update_actor(batch)

                        # Check if ESI/training plan is close to expiration
                        esi_close_to_expiration = should_save_ckpt_esi(
                            max_steps_duration=self.max_steps_duration,
                            redundant_time=self.config.trainer.esi_redundant_time,
                        )
                        if self.config.trainer.save_freq > 0 and (
                            is_last_step
                            or self.global_steps % self.config.trainer.save_freq == 0
                            or esi_close_to_expiration
                        ):
                            if esi_close_to_expiration:
                                print("Force saving checkpoint: ESI instance expiration approaching.")
                            with marked_timer("save_checkpoint", timing_raw, "green"):
                                self._save_checkpoint()

                        with marked_timer("update_weights", timing_raw, "red"):
                            self.checkpoint_manager.update_weights()
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        value_logit_rms = actor_output_metrics.get("actor/vimpo_value_logit_grad_rms")
                        actor_logit_rms = actor_output_metrics.get("actor/vimpo_actor_logit_grad_rms")
                        if value_logit_rms is not None and actor_logit_rms is not None:
                            actor_output_metrics["actor/vimpo_actor_to_value_logit_grad_rms_ratio"] = (
                                actor_logit_rms / (value_logit_rms + 1e-12)
                            )
                        metrics.update(actor_output_metrics)

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        self._log_rollout_data(batch, reward_extra_infos_dict, timing_raw, rollout_data_dir)

                # validate
                if self.config.trainer.test_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.test_freq == 0
                ):
                    with marked_timer("testing", timing_raw, "green"):
                        val_metrics: dict = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                with marked_timer("stop_profile", timing_raw):
                    next_step_profile = (
                        self.global_steps + 1 in self.config.global_profiler.steps
                        if self.config.global_profiler.steps is not None
                        else False
                    )
                    self._stop_profiling(
                        curr_step_profile and not next_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                    prev_step_profile = curr_step_profile
                    curr_step_profile = next_step_profile

                steps_duration = timing_raw.get("step", 0)
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)

                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                timing_raw = defaultdict(float)  # clear timing

                metrics["train/num_gen_batches"] = num_gen_batches
                batch = None
                num_prompt_in_batch = 0
                num_gen_batches = 0

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                if is_last_step:
                    if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                        self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=True)
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                progress_bar.update(1)
                self.global_steps += 1
                self.gen_steps += 1
        # check if last step checkpint exists
        checkpoint_dir = os.path.join(self.config.trainer.default_local_dir, f"global_step_{self.global_steps}")
        if not os.path.exists(checkpoint_dir):
            # save last step checkpoint
            timing_raw = defaultdict(float)
            with marked_timer("save_checkpoint", timing_raw, "green"):
                self._save_checkpoint()
            metrics = {f"timing/{k}": v for k, v in timing_raw.items()}
            logger.log(data=metrics, step=self.global_steps)
