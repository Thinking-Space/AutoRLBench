import pytest

from idea.skill_gen import (
    SkillGenRewardConfig,
    compute_skill_gen_reward,
)


def test_skill_gen_reward_combines_gain_and_format_bonuses():
    reward = compute_skill_gen_reward(
        downstream_logprob_gain=0.5,
        episode_success=1.0,
        valid_json=True,
        episode_skill="Check all constraints before selecting.",
        step_skills={},
        raw_output='{"episode_skill":"Check all constraints before selecting."}',
        config=SkillGenRewardConfig(
            valid_json_bonus=0.1,
            non_empty_skill_bonus=0.2,
            too_long_penalty=0.3,
            reward_clip=None,
        ),
    )

    assert reward["reward"] == pytest.approx(0.8)
    assert reward["downstream_logprob_gain_on_success_steps"] == 0.5
    assert reward["valid_json_bonus"] == 0.1
    assert reward["non_empty_skill_bonus"] == 0.2
    assert reward["too_long_penalty"] == 0.0


def test_skill_gen_reward_zeros_downstream_gain_for_failed_episode():
    reward = compute_skill_gen_reward(
        downstream_logprob_gain=1.0,
        episode_success=0.0,
        valid_json=True,
        episode_skill="Avoid guessing.",
        step_skills={},
        raw_output="{}",
        config=SkillGenRewardConfig(valid_json_bonus=0.0, non_empty_skill_bonus=0.0),
    )

    assert reward["downstream_logprob_gain_on_success_steps"] == 0.0
    assert reward["raw_downstream_logprob_gain"] == 1.0
    assert reward["effective_downstream_logprob_gain"] == 0.0
    assert reward["reward"] == 0.0


def test_skill_gen_reward_can_negate_downstream_gain_for_failed_episode():
    reward = compute_skill_gen_reward(
        downstream_logprob_gain=-0.4,
        episode_success=0.0,
        valid_json=True,
        episode_skill="Avoid repeating the failed action.",
        step_skills={},
        raw_output="{}",
        config=SkillGenRewardConfig(
            valid_json_bonus=0.0,
            non_empty_skill_bonus=0.0,
            failed_reward_mode="negate",
            reward_clip=None,
        ),
    )

    assert reward["raw_downstream_logprob_gain"] == pytest.approx(-0.4)
    assert reward["effective_downstream_logprob_gain"] == pytest.approx(0.4)
    assert reward["downstream_logprob_gain_on_success_steps"] == pytest.approx(0.4)
    assert reward["reward"] == pytest.approx(0.4)


def test_skill_gen_reward_penalizes_too_long():
    reward = compute_skill_gen_reward(
        downstream_logprob_gain=0.0,
        episode_success=1.0,
        valid_json=False,
        episode_skill="Use concise reusable guidance.",
        step_skills={},
        raw_output="x" * 20,
        config=SkillGenRewardConfig(
            valid_json_bonus=0.1,
            non_empty_skill_bonus=0.2,
            too_long_penalty=0.3,
            max_output_chars=8,
            reward_clip=None,
        ),
    )

    assert reward["reward"] == pytest.approx(-0.1)
    assert reward["too_long_penalty"] == 0.3
