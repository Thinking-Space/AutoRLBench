from verl.trainer.ppo.env_aux_loss_utils import (
    create_inverse_dynamics_messages,
    create_search_inverse_dynamics_messages,
    create_state_prediction_messages,
)


def test_state_prediction_message_uses_observation_target():
    messages = create_state_prediction_messages(
        history_pairs=[("obs0", "act0")],
        current_obs="obs1",
        action="act1",
        next_obs="obs2",
        step_number=2,
        history_start_step=1,
    )

    assert messages[-1]["role"] == "assistant"
    assert messages[-1]["content"] == "<observation>obs2</observation>"


def test_inverse_dynamics_message_includes_actual_action():
    messages = create_inverse_dynamics_messages(
        history_pairs=[],
        current_obs="obs1",
        next_obs="obs2",
        action="open door",
        admissible_actions=["look"],
        step_number=1,
        history_start_step=1,
    )

    assert "open door" in messages[0]["content"]
    assert messages[-1]["content"] == "<action>open door</action>"


def test_search_inverse_dynamics_message_uses_search_target():
    messages = create_search_inverse_dynamics_messages(
        history_pairs=[],
        current_obs="obs1",
        next_obs="obs2",
        action="weather today",
        step_number=1,
        history_start_step=1,
    )

    assert "What search query" in messages[0]["content"]
    assert messages[-1]["content"] == "<search>weather today</search>"
