import math

import torch

from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.ray_trainer import compute_response_mask
from verl.utils.profiler import marked_timer

from recipe.dapo.dapo_ray_trainer import RayDAPOTrainer


def _optional_float(value, name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, str) and value.lower() in {"none", "null"}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"vimpo_config.{name} must be null/None or a float, got {value!r}.") from exc


class RayVIMPOTrainer(RayDAPOTrainer):
    def _get_vimpo_config(self):
        return self.config.get("vimpo_config", {})

    def _use_vimpo_loss(self) -> bool:
        return bool(self._get_vimpo_config().get("use_vimpo_loss", False))

    def _get_value_type(self) -> str:
        return str(self._get_vimpo_config().get("value_type", "raw"))

    def _normalized_vimpo_config(self) -> dict:
        vimpo_config = dict(self._get_vimpo_config())
        vimpo_config.setdefault("value_type", "raw")
        vimpo_config.setdefault("vimpo_value_loss_type", "squared_error")
        vimpo_config.setdefault("vimpo_value_huber_delta", 1.0)
        vimpo_config.setdefault("vimpo_value_loss_normalized", False)
        vimpo_config.setdefault("vimpo_value_loss_coeff", 1.0)
        vimpo_config.setdefault("vimpo_ppo_whiten_adv", True)
        vimpo_config.setdefault("vimpo_detach_kl", True)
        vimpo_config.setdefault("vimpo_adaptive_actor_coeff", False)
        vimpo_config.setdefault("vimpo_adaptive_actor_alpha", 0.15)
        vimpo_config.setdefault("vimpo_adaptive_actor_ema_decay", 0.9)
        vimpo_config.setdefault("vimpo_adaptive_actor_eps", 1e-12)
        vimpo_config["vimpo_cap_v0"] = _optional_float(vimpo_config.get("vimpo_cap_v0", None), "vimpo_cap_v0")
        vimpo_config.setdefault("lam", float(self.config.algorithm.lam))
        return vimpo_config

    def _validate_vimpo_mode(self):
        vimpo_config = self._get_vimpo_config()
        value_type = str(vimpo_config.get("value_type", "raw"))
        use_grpo_actor = bool(vimpo_config.get("use_grpo_actor", False))
        use_ppo_actor = bool(vimpo_config.get("use_ppo_actor", False))
        vimpo_kl_estimator = vimpo_config.get("vimpo_kl_estimator", None)
        candidate_kl_topk = int(vimpo_config.get("candidate_kl_topk", 0))
        candidate_kl_M = int(vimpo_config.get("candidate_kl_M", 0))
        gamma = float(vimpo_config.get("gamma", 1.0))
        value_loss_type = str(vimpo_config.get("vimpo_value_loss_type", "squared_error"))
        value_huber_delta = float(vimpo_config.get("vimpo_value_huber_delta", 1.0))
        value_loss_coeff = float(vimpo_config.get("vimpo_value_loss_coeff", 1.0))
        update_ref_freq = int(vimpo_config.get("vimpo_update_ref_freq", 0))
        _optional_float(vimpo_config.get("vimpo_cap_v0", None), "vimpo_cap_v0")

        if self._use_vimpo_loss() and value_type not in {"raw", "soft"}:
            raise ValueError(f"vimpo_config.value_type must be 'raw' or 'soft', got {value_type!r}.")
        if self._use_vimpo_loss() and value_loss_type not in {"squared_error", "huber"}:
            raise ValueError(
                f"vimpo_config.vimpo_value_loss_type must be 'squared_error' or 'huber', got {value_loss_type!r}."
            )
        if self._use_vimpo_loss() and value_huber_delta <= 0:
            raise ValueError(f"vimpo_config.vimpo_value_huber_delta must be positive, got {value_huber_delta}.")
        if self._use_vimpo_loss() and (not math.isfinite(value_loss_coeff) or value_loss_coeff < 0):
            raise ValueError(f"vimpo_config.vimpo_value_loss_coeff must be non-negative, got {value_loss_coeff}.")
        if use_grpo_actor and use_ppo_actor:
            raise ValueError("vimpo_config.use_grpo_actor and vimpo_config.use_ppo_actor cannot both be True.")
        if self._use_vimpo_loss() and value_type == "soft" and (use_grpo_actor or use_ppo_actor):
            raise NotImplementedError(
                "vimpo_config.value_type='soft' does not support use_grpo_actor or use_ppo_actor in v1."
            )
        if self._use_vimpo_loss() and vimpo_kl_estimator not in {"exact", "candidate_set", "k1", "k2", "k3"}:
            raise NotImplementedError(
                f"VIMPO KL estimator {vimpo_kl_estimator!r} is not supported yet. "
                "Only 'exact', 'candidate_set', 'k1', 'k2', and 'k3' are implemented."
            )
        if self._use_vimpo_loss() and value_type == "raw" and vimpo_kl_estimator == "k1":
            raise ValueError(
                "vimpo_config.vimpo_kl_estimator='k1' is degenerate for raw VIMPO: "
                "the sampled KL equals log(pi/ref), so the token term cancels to zero. "
                "Use 'exact', 'candidate_set', 'k2', or 'k3'."
            )
        if self._use_vimpo_loss() and vimpo_kl_estimator == "candidate_set" and candidate_kl_topk <= 0:
            raise ValueError(
                f"VIMPO candidate_set KL requires candidate_kl_topk > 0, got {candidate_kl_topk}."
            )
        if self._use_vimpo_loss() and vimpo_kl_estimator == "candidate_set" and candidate_kl_M != 0:
            raise NotImplementedError(
                "VIMPO candidate_set KL currently supports policy top-k only; candidate_kl_M must be 0."
            )
        if self._use_vimpo_loss() and self.config.actor_rollout_ref.actor.use_kl_loss:
            raise NotImplementedError("VIMPO exact-KL mode does not support actor.use_kl_loss at the same time.")
        if self._use_vimpo_loss() and self.config.algorithm.use_kl_in_reward:
            raise NotImplementedError("VIMPO exact-KL mode does not support algorithm.use_kl_in_reward.")
        if self._use_vimpo_loss() and not self.use_reference_policy:
            raise RuntimeError("VIMPO requires a reference policy for ref log-prob or KL computation.")
        if self._use_vimpo_loss() and gamma != 1.0:
            raise ValueError("VIMPO currently enforces gamma=1.0; remove vimpo_config.gamma or set it to 1.0.")
        if self._use_vimpo_loss() and update_ref_freq < 0:
            raise ValueError(f"vimpo_config.vimpo_update_ref_freq must be >= 0, got {update_ref_freq}.")
        if self._use_vimpo_loss() and update_ref_freq > 0 and self.ref_in_actor:
            raise NotImplementedError(
                "vimpo_config.vimpo_update_ref_freq > 0 is not supported when the reference policy is the actor "
                "with LoRA adapters disabled."
            )

    def _requires_old_log_prob(self) -> bool:
        if not self._use_vimpo_loss():
            return True
        vimpo_config = self._get_vimpo_config()
        if str(vimpo_config.get("value_type", "raw")) == "soft":
            return True
        return bool(vimpo_config.get("use_grpo_actor", False) or vimpo_config.get("use_ppo_actor", False))

    def _requires_ref_log_prob(self) -> bool:
        if not self._use_vimpo_loss():
            return self.use_reference_policy
        return False

    def _skip_rollout_correction(self) -> bool:
        return self._use_vimpo_loss() and not self._requires_old_log_prob()

    def _attach_raw_value_baseline(self, batch: DataProto) -> DataProto:
        if "uid" not in batch.non_tensor_batch:
            raise RuntimeError("Raw VIMPO requires prompt-group uid in batch.non_tensor_batch.")
        if "token_level_rewards" not in batch.batch:
            raise RuntimeError("Raw VIMPO requires token_level_rewards in batch.batch.")
        if "response_mask" not in batch.batch:
            batch.batch["response_mask"] = compute_response_mask(batch)

        from recipe.vimpo.vimpo_core_algos import compute_reward_baseline_by_uid

        vimpo_config = self._normalized_vimpo_config()
        vimpo_reward_baseline = compute_reward_baseline_by_uid(
            token_level_rewards=batch.batch["token_level_rewards"],
            response_mask=batch.batch["response_mask"],
            uids=batch.non_tensor_batch["uid"],
            cap_v0=vimpo_config.get("vimpo_cap_v0"),
        )
        batch.batch["vimpo_reward_baseline"] = vimpo_reward_baseline
        return batch

    def _attach_value_loss_norm_denom(self, batch: DataProto, baseline_key: str) -> DataProto:
        vimpo_config = self._normalized_vimpo_config()
        if not bool(vimpo_config.get("vimpo_value_loss_normalized", False)):
            return batch
        if baseline_key not in batch.batch:
            raise RuntimeError(f"VIMPO value-loss normalization requires {baseline_key!r} in batch.batch.")

        from recipe.vimpo.vimpo_core_algos import compute_value_loss_norm_denom

        response_mask = batch.batch["response_mask"].to(torch.float32)
        final_reward = torch.sum(batch.batch["token_level_rewards"].to(torch.float32) * response_mask, dim=-1)
        target = final_reward - batch.batch[baseline_key].to(torch.float32)
        denom = compute_value_loss_norm_denom(target)
        if denom.item() < 1e-4:
            raise ValueError(
                "vimpo_value_loss_normalized=True requires full actor-update batch target MSE denominator >= 1e-4, "
                f"got {denom.item()}."
            )
        batch.meta_info["vimpo_value_loss_norm_denom"] = float(denom.item())
        return batch

    def compute_kl_related_metrics(self, batch: DataProto, metrics: dict, timing_raw: dict):
        self._validate_vimpo_mode()
        batch.batch["response_mask"] = compute_response_mask(batch)
        if self._use_vimpo_loss():
            batch.meta_info["vimpo_config"] = self._normalized_vimpo_config()

        if self._use_vimpo_loss() and self._get_value_type() == "raw":
            batch = self._attach_raw_value_baseline(batch)
            batch = self._attach_value_loss_norm_denom(batch, baseline_key="vimpo_reward_baseline")

        if not self._requires_old_log_prob():
            return batch

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
            metrics.update(
                {
                    "actor/old_entropy": entropy_agg.detach().item(),
                    "perf/mfu/actor_infer": old_log_prob_mfu,
                }
            )
            old_log_prob.batch.pop("entropys")
            batch = batch.union(old_log_prob)

        if self._use_vimpo_loss() and self._get_value_type() == "soft":
            required_keys = {"old_log_probs", "ref_log_prob", "vimpo_old_token_kl", "token_level_rewards"}
            missing_keys = sorted(key for key in required_keys if key not in batch.batch)
            if missing_keys:
                raise RuntimeError(f"VIMPO soft value is missing required batch tensors: {missing_keys}.")
            if "uid" not in batch.non_tensor_batch:
                raise RuntimeError("VIMPO soft value requires prompt-group uid in batch.non_tensor_batch.")

            from recipe.vimpo.vimpo_core_algos import compute_soft_value_baseline_by_uid

            vimpo_config = self._normalized_vimpo_config()
            vimpo_soft_v0, _ = compute_soft_value_baseline_by_uid(
                old_log_prob=batch.batch["old_log_probs"],
                ref_log_prob=batch.batch["ref_log_prob"],
                old_token_kl=batch.batch["vimpo_old_token_kl"],
                token_level_rewards=batch.batch["token_level_rewards"],
                response_mask=batch.batch["response_mask"],
                beta=float(vimpo_config.get("vimpo_beta", 0.0)),
                uids=batch.non_tensor_batch["uid"],
                cap_v0=vimpo_config.get("vimpo_cap_v0"),
            )
            batch.batch["vimpo_soft_v0"] = vimpo_soft_v0
            batch = self._attach_value_loss_norm_denom(batch, baseline_key="vimpo_soft_v0")

        if self._requires_ref_log_prob():
            with marked_timer("ref", timing_raw, "olive"):
                ref_log_prob = self._compute_ref_log_prob(batch)
                batch = batch.union(ref_log_prob)

        return batch

    def _update_actor(self, batch: DataProto) -> DataProto:
        if self._use_vimpo_loss():
            self._validate_vimpo_mode()
            batch.meta_info["vimpo_config"] = self._normalized_vimpo_config()
        actor_output = super()._update_actor(batch)

        update_ref_freq = int(self._get_vimpo_config().get("vimpo_update_ref_freq", 0))
        if self._use_vimpo_loss() and update_ref_freq > 0 and self.global_steps % update_ref_freq == 0:
            if not hasattr(self.actor_rollout_wg, "update_ref_policy_from_actor"):
                raise NotImplementedError(
                    "vimpo_config.vimpo_update_ref_freq requires actor_rollout_wg.update_ref_policy_from_actor(); "
                    "the current worker backend does not implement reference refresh."
            )
            self.actor_rollout_wg.update_ref_policy_from_actor()
            actor_output.meta_info["metrics"]["vimpo/ref_updated"] = 1.0

        return actor_output

    def fit(self):
        self._validate_vimpo_mode()
        rollout_corr_backup = None
        if self._skip_rollout_correction():
            rollout_corr_backup = self.config.algorithm.get("rollout_correction", None)
            self.config.algorithm.rollout_correction = None
        try:
            return super().fit()
        finally:
            if rollout_corr_backup is not None:
                self.config.algorithm.rollout_correction = rollout_corr_backup
