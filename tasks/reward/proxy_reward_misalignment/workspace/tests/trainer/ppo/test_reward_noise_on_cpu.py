# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

import numpy as np
import torch

from verl import DataProto

from recipe.dapo.dapo_ray_trainer import apply_group_reward_noise, select_filter_metric_name


def _make_batch(scores: list[float], uids: list[str]) -> DataProto:
    batch_size = len(scores)
    return DataProto.from_dict(
        tensors={
            "responses": torch.ones(batch_size, 3, dtype=torch.long),
            "response_mask": torch.ones(batch_size, 3, dtype=torch.long),
        },
        non_tensors={
            "uid": np.asarray(uids, dtype=object),
            "score": np.asarray(scores, dtype=np.float32),
            "acc": np.asarray(scores, dtype=np.float32),
        },
    )


def _terminal_reward(scores: list[float]) -> torch.Tensor:
    reward = torch.zeros(len(scores), 3)
    reward[:, -1] = torch.tensor(scores)
    return reward


def test_reward_noise_p0_preserves_reward_and_adds_noisy_metadata():
    scores = [1.0, 0.0, 1.0, 0.0]
    batch = _make_batch(scores=scores, uids=["a", "a", "b", "b"])
    reward = _terminal_reward(scores)

    noisy_reward, metrics = apply_group_reward_noise(
        batch=batch,
        reward_tensor=reward,
        reward_noise_config={
            "enable": True,
            "flip_prob": 0.0,
            "seed": 7,
            "mode": "group_flip_binary_score",
        },
        global_steps=3,
    )

    assert torch.equal(noisy_reward, reward)
    np.testing.assert_array_equal(batch.non_tensor_batch["score"], np.asarray(scores, dtype=np.float32))
    np.testing.assert_array_equal(batch.non_tensor_batch["acc"], np.asarray(scores, dtype=np.float32))
    np.testing.assert_array_equal(batch.non_tensor_batch["score_noisy"], np.asarray(scores, dtype=np.float32))
    np.testing.assert_array_equal(batch.non_tensor_batch["acc_noisy"], np.asarray(scores, dtype=np.float32))
    assert not np.any(batch.non_tensor_batch["reward_noise_flipped"])
    assert metrics["reward_noise/flipped_group_ratio"] == 0.0
    assert metrics["reward_noise/flipped_traj_ratio"] == 0.0


def test_reward_noise_p1_flips_binary_score_and_keeps_clean_metadata():
    scores = [1.0, 0.0, 0.0, 1.0]
    batch = _make_batch(scores=scores, uids=["a", "a", "b", "b"])
    reward = _terminal_reward(scores)

    noisy_reward, metrics = apply_group_reward_noise(
        batch=batch,
        reward_tensor=reward,
        reward_noise_config={
            "enable": True,
            "flip_prob": 1.0,
            "seed": 7,
            "mode": "group_flip_binary_score",
        },
        global_steps=3,
    )

    expected_scores = np.asarray([0.0, 1.0, 1.0, 0.0], dtype=np.float32)
    expected_reward = _terminal_reward(expected_scores.tolist())
    assert torch.equal(noisy_reward, expected_reward)
    np.testing.assert_array_equal(batch.non_tensor_batch["score"], np.asarray(scores, dtype=np.float32))
    np.testing.assert_array_equal(batch.non_tensor_batch["acc"], np.asarray(scores, dtype=np.float32))
    np.testing.assert_array_equal(batch.non_tensor_batch["score_noisy"], expected_scores)
    np.testing.assert_array_equal(batch.non_tensor_batch["acc_noisy"], expected_scores)
    assert np.all(batch.non_tensor_batch["reward_noise_flipped"])
    assert metrics["reward_noise/flipped_group_ratio"] == 1.0
    assert metrics["reward_noise/flipped_traj_ratio"] == 1.0


def test_reward_noise_decision_is_shared_by_prompt_group():
    scores = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0]
    uids = ["a", "a", "b", "b", "c", "c"]
    batch = _make_batch(scores=scores, uids=uids)

    apply_group_reward_noise(
        batch=batch,
        reward_tensor=_terminal_reward(scores),
        reward_noise_config={
            "enable": True,
            "flip_prob": 0.5,
            "seed": 7,
            "mode": "group_flip_binary_score",
        },
        global_steps=3,
    )

    flipped_by_uid = {}
    for uid, flipped in zip(batch.non_tensor_batch["uid"], batch.non_tensor_batch["reward_noise_flipped"], strict=True):
        flipped_by_uid.setdefault(uid, bool(flipped))
        assert flipped_by_uid[uid] == bool(flipped)


def test_filter_metric_selection_prefers_noisy_score_or_acc_when_enabled():
    batch = _make_batch(scores=[1.0, 0.0], uids=["a", "a"])
    batch.non_tensor_batch["score_noisy"] = np.asarray([0.0, 1.0], dtype=np.float32)
    batch.non_tensor_batch["acc_noisy"] = np.asarray([0.0, 1.0], dtype=np.float32)

    enabled = {"enable": True}
    disabled = {"enable": False}

    assert select_filter_metric_name(batch, "score", enabled) == "score_noisy"
    assert select_filter_metric_name(batch, "acc", enabled) == "acc_noisy"
    assert select_filter_metric_name(batch, "seq_reward", enabled) == "seq_reward"
    assert select_filter_metric_name(batch, "score", disabled) == "score"
