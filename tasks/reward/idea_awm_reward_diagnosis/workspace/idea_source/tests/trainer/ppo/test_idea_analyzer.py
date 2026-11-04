import pytest

from idea.analysis import SEEDEpisodeAnalyzer


def _sample_steps():
    return [
        {
            "step_index": 0,
            "observation": "Search page with several options.",
            "observation_prompt": (
                "You are an expert autonomous agent operating in the WebShop e-commerce environment.\n"
                "Your task is to: find a red mug under 20 dollars.\n"
                "Your current observation is: Search page with several options."
            ),
            "response": "<action>search[red mug]</action>",
            "step_reward": 0.1,
        },
        {
            "step_index": 1,
            "observation": "Results include a matching red mug.",
            "observation_prompt": (
                "You are an expert autonomous agent operating in the WebShop e-commerce environment.\n"
                "Your task is to: find a red mug under 20 dollars.\n"
                "Your current observation is: Results include a matching red mug."
            ),
            "response": "<action>click[item 2]</action>",
            "step_reward": 0.5,
        },
    ]


def test_strategy_bank_prompt_version_is_not_supported():
    with pytest.raises(ValueError, match="Unsupported SEED analysis_prompt_version"):
        SEEDEpisodeAnalyzer(analysis_prompt_version="strategy_bank")


def test_search_skill_only_prompt_version_is_not_supported():
    with pytest.raises(ValueError, match="Unsupported SEED analysis_prompt_version"):
        SEEDEpisodeAnalyzer(analysis_prompt_version="search_skill_only")


def test_search_strategy_bank_prompt_version_is_not_supported():
    with pytest.raises(ValueError, match="Unsupported SEED analysis_prompt_version"):
        SEEDEpisodeAnalyzer(analysis_prompt_version="search_strategy_bank")


def test_sokoban_visual_prompt_version_is_not_supported():
    with pytest.raises(ValueError, match="Unsupported SEED analysis_prompt_version"):
        SEEDEpisodeAnalyzer(analysis_prompt_version="sokoban_visual")


def test_seed_openai_prompt_requests_episode_skill_and_step_skills():
    analyzer = SEEDEpisodeAnalyzer()

    prompt = analyzer._build_episode_analysis_prompt(
        steps=_sample_steps(),
        candidate_step_indices=[0, 1],
        episode_success=1.0,
    )
    user_prompt = prompt["messages"][0]["content"]

    assert "selected_steps" not in user_prompt
    assert "overall_skill" not in user_prompt
    assert "episode_skill" in user_prompt
    assert "step_skills" in user_prompt
    assert "Return format:" in user_prompt
    assert "Task description:" in user_prompt
    assert "find a red mug under 20 dollars" in user_prompt
    assert "successful trajectory into workflow" in user_prompt
    assert "critical step" in user_prompt
    assert "Candidate step indices" in user_prompt
    assert "Return only these top-level fields: episode_summary, episode_skill, step_skills" in user_prompt
    assert "episode_success: success" in user_prompt
    assert "interpreted_outcome" not in user_prompt
    assert "Because this is a successful episode" not in user_prompt


def test_seed_step_only_prompt_requests_only_step_skills():
    analyzer = SEEDEpisodeAnalyzer(skill_mode="step_only")

    prompt = analyzer._build_episode_analysis_prompt(
        steps=_sample_steps(),
        candidate_step_indices=[0, 1],
        episode_success=1.0,
    )
    user_prompt = prompt["messages"][0]["content"]

    assert "step_skills" in user_prompt
    assert '"episode_skill": "string"' not in user_prompt
    assert "Return only these top-level fields: episode_summary, step_skills" in user_prompt


def test_seed_episode_only_prompt_requests_only_episode_skill():
    analyzer = SEEDEpisodeAnalyzer(skill_mode="episode_only")

    prompt = analyzer._build_episode_analysis_prompt(
        steps=_sample_steps(),
        candidate_step_indices=[0, 1],
        episode_success=1.0,
    )
    user_prompt = prompt["messages"][0]["content"]

    assert "episode_skill" in user_prompt
    assert '"step_skills": {' not in user_prompt
    assert "Return only these top-level fields: episode_summary, episode_skill" in user_prompt


def test_seed_episode_only_no_summary_prompt_requests_only_episode_skill():
    analyzer = SEEDEpisodeAnalyzer(skill_mode="episode_only", include_episode_summary=False)

    prompt = analyzer._build_episode_analysis_prompt(
        steps=_sample_steps(),
        candidate_step_indices=[0, 1],
        episode_success=1.0,
    )
    user_prompt = prompt["messages"][0]["content"]

    assert "episode_skill" in user_prompt
    assert "episode_summary" not in user_prompt
    assert '"step_skills": {' not in user_prompt
    assert "Return only this top-level field: episode_skill" in user_prompt


def test_seed_visual_episode_only_prompt_omits_step_skills():
    analyzer = SEEDEpisodeAnalyzer(
        skill_mode="episode_only",
        analysis_prompt_version="seed_visual",
    )

    prompt = analyzer._build_episode_analysis_prompt(
        steps=[
            {
                "step_index": 0,
                "has_observation_image": True,
                "response": "<think>route around the box</think><action>right</action>",
                "action_valid": True,
                "step_reward": 0.0,
            },
            {
                "step_index": 1,
                "has_observation_image": True,
                "response": "<think>push toward target</think><action>up</action>",
                "action_valid": True,
                "step_reward": 1.0,
            },
        ],
        candidate_step_indices=[0, 1],
        episode_success=1.0,
    )
    user_prompt = prompt["messages"][0]["content"]

    assert "episode_skill" in user_prompt
    assert "episode_summary" in user_prompt
    assert "step_skills" not in user_prompt
    assert "Candidate step indices" not in user_prompt
    assert "Return format:" in user_prompt
    assert "successful trajectory into workflow" in user_prompt
    assert "Observation: <image>" in user_prompt
    assert "Board observation:" not in user_prompt
    assert "visual Sokoban" not in user_prompt


def test_seed_episode_step_no_summary_prompt_requests_skill_fields_without_summary():
    analyzer = SEEDEpisodeAnalyzer(include_episode_summary=False)

    prompt = analyzer._build_episode_analysis_prompt(
        steps=_sample_steps(),
        candidate_step_indices=[0, 1],
        episode_success=1.0,
    )
    user_prompt = prompt["messages"][0]["content"]

    assert "episode_skill" in user_prompt
    assert "step_skills" in user_prompt
    assert "episode_summary" not in user_prompt
    assert "Return only these top-level fields: episode_skill, step_skills" in user_prompt


def test_seed_prompt_for_failed_episode_emphasizes_avoidance():
    analyzer = SEEDEpisodeAnalyzer()

    prompt = analyzer._build_episode_analysis_prompt(
        steps=_sample_steps(),
        candidate_step_indices=[0, 1],
        episode_success=0.0,
    )
    user_prompt = prompt["messages"][0]["content"]

    assert "failed trajectory into avoidance rules" in user_prompt
    assert "core mistake and warning signs" in user_prompt
    assert "episode_success: failure" in user_prompt
    assert "interpreted_outcome" not in user_prompt
    assert "Because this is a failed episode" not in user_prompt


def test_seed_policy_vllm_backend_defers_generation_to_trainer():
    analyzer = SEEDEpisodeAnalyzer(backend="policy_vllm")

    assert analyzer.backend == "policy_vllm"
    with pytest.raises(ValueError, match="does not use an OpenAI client"):
        analyzer._get_openai_client()
    with pytest.raises(ValueError, match="only implemented for openai"):
        analyzer.analyze_episode(steps=_sample_steps())


def test_seed_parse_uses_episode_skill_and_step_skills():
    analyzer = SEEDEpisodeAnalyzer()

    parsed = analyzer._parse_analysis_response(
        """
        {
          "episode_summary": "summary",
          "episode_skill": "skill",
          "selected_steps": [99],
          "step_skills": {
            "1": "click the matching item"
          }
        }
        """
    )

    assert "selected_steps" not in parsed
    assert parsed["episode_skill"] == "skill"
    assert parsed["step_skills"] == {1: "click the matching item"}


def test_seed_step_only_parse_drops_episode_skill():
    analyzer = SEEDEpisodeAnalyzer(skill_mode="step_only")

    parsed = analyzer._parse_analysis_response(
        """
        {
          "episode_summary": "summary",
          "episode_skill": "should be ignored",
          "step_skills": {
            "1": "click the matching item"
          }
        }
        """
    )

    assert parsed["episode_skill"] == ""
    assert parsed["step_skills"] == {1: "click the matching item"}


def test_seed_episode_only_parse_drops_step_skills():
    analyzer = SEEDEpisodeAnalyzer(skill_mode="episode_only")

    parsed = analyzer._parse_analysis_response(
        """
        {
          "episode_summary": "summary",
          "episode_skill": "check all constraints first",
          "step_skills": {
            "1": "should be ignored"
          }
        }
        """
    )

    assert parsed["episode_skill"] == "check all constraints first"
    assert parsed["step_skills"] == {}


def test_seed_no_summary_parse_keeps_summary_empty():
    analyzer = SEEDEpisodeAnalyzer(skill_mode="episode_only", include_episode_summary=False)

    parsed = analyzer._parse_analysis_response(
        """
        {
          "episode_summary": "should be ignored",
          "episode_skill": "check all constraints first"
        }
        """
    )

    assert parsed["episode_summary"] == ""
    assert parsed["episode_skill"] == "check all constraints first"


def test_seed_parse_extracts_json_from_markdown_fence():
    analyzer = SEEDEpisodeAnalyzer()

    parsed = analyzer._parse_analysis_response(
        """
        Here is the result:
        ```json
        {
          "episode_skill": "check the constraints before choosing",
          "step_skills": {"0": "verify the key constraint first"}
        }
        ```
        """
    )

    assert parsed["episode_skill"] == "check the constraints before choosing"
    assert parsed["step_skills"] == {0: "verify the key constraint first"}


def test_seed_parse_prefers_object_with_episode_skill():
    analyzer = SEEDEpisodeAnalyzer()

    parsed = analyzer._parse_analysis_response(
        """
        Ignore this helper object: {"note": "not the answer"}
        Final answer:
        {
          "episode_skill": "use the final matching evidence",
          "step_skills": {"1": "click only after matching the task"}
        }
        """
    )

    assert parsed["episode_skill"] == "use the final matching evidence"
    assert parsed["step_skills"] == {1: "click only after matching the task"}


def test_seed_parse_repairs_common_json_noise():
    analyzer = SEEDEpisodeAnalyzer()

    parsed = analyzer._parse_analysis_response(
        """
        {
          'episode_skill': 'compare all required attributes before acting',
          'step_skills': {'1': 'check the attributes before selecting'},
        }
        """
    )

    assert parsed["episode_skill"] == "compare all required attributes before acting"
    assert parsed["step_skills"] == {1: "check the attributes before selecting"}
