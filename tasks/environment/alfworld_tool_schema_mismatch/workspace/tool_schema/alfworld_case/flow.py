"""Agent-R1 rollout integration without changing vendored training code."""
from uuid import uuid4

from agent_r1.agent_flow.agent_flow import AgentFlowBase, AgentFlowOutput, AgentFlowStep, register
from recipes.alfworld.alfworld_agent_flow import _recover_tool_calls_from_text
from recipes.alfworld.env.alfworld_wrapper import AlfworldTextworldEnv
from recipes.alfworld.reward_fn import compute_score
from verl.experimental.agent_loop.tool_parser import ToolParser
from verl.utils.profiler import simple_timer

from alfworld_case.episode import Episode
from alfworld_case.tools import tool_schemas


@register("alfworld_schema")
class SchemaAgentFlow(AgentFlowBase):
    def __init__(self, *args, max_steps=40, **kwargs):
        super().__init__(*args, **kwargs)
        self.schemas = tool_schemas()
        self.max_steps = max_steps
        self.tool_parser = ToolParser.get_tool_parser(
            self.config.actor_rollout_ref.rollout.multi_turn.format, self.tokenizer)

    async def run(self, sampling_params, **kwargs):
        info = kwargs.get("extra_info") or {}
        episode = Episode(AlfworldTextworldEnv(max_episode_steps=self.max_steps), self.max_steps)
        steps, metrics = [], {}
        try:
            episode.reset(info["game_relative_path"], info.get("task_id"))
            while not episode.done:
                messages = episode.messages()
                prompt_ids = await self.apply_chat_template(messages, tools=self.schemas)
                # Drop only oldest complete turns; retain initial scene/task and newest feedback.
                while len(prompt_ids) > self.config.actor_rollout_ref.rollout.prompt_length and len(episode.history) > 1:
                    episode.history.pop(0)
                    prompt_ids = await self.apply_chat_template(episode.messages(), tools=self.schemas)
                if len(prompt_ids) > self.config.actor_rollout_ref.rollout.prompt_length:
                    raise ValueError("ALFWorld prompt exceeds configured budget even without old history")
                anchor = episode.observation
                with simple_timer("generate_sequences", metrics):
                    output = await self.server_manager.generate(
                        request_id=uuid4().hex, prompt_ids=prompt_ids,
                        sampling_params={**sampling_params,
                                         "max_tokens": self.config.actor_rollout_ref.rollout.response_length})
                response_ids = output.token_ids[:self.config.actor_rollout_ref.rollout.response_length]
                _, calls = await self.tool_parser.extract_tool_calls(response_ids)
                if not calls:
                    calls = _recover_tool_calls_from_text(self.tokenizer.decode(response_ids, skip_special_tokens=True))
                with simple_timer("tool_calls", metrics):
                    result = episode.act([(call.name, call.arguments) for call in calls],
                                         self.tokenizer.decode(response_ids, skip_special_tokens=True))
                reward_info = {key: info.get(key, "") for key in
                               ("task_id", "split", "task_type_raw", "task_family")}
                reward_info.update({key: result[key] for key in
                                    ("success", "num_steps", "is_action_valid",
                                     "step_env_reward", "dense_reward_sum")})
                # Use the terminal reward schema on intermediate steps as well.
                reward_info = compute_score("alfworld", "", {"success": None},
                                            {"reward_extra_info": reward_info})
                # Score the terminal action once, including failures at the turn limit.
                step = AgentFlowStep(
                    prompt_ids=prompt_ids, response_ids=response_ids,
                    response_logprobs=output.log_probs[:len(response_ids)] if output.log_probs else None,
                    reward_score=None if episode.done else 0.0,
                    extra_fields={"anchor_obs": anchor, "reward_extra_info": reward_info,
                                  "tool_feedback": result["feedback"], "tool_error": result["tool_error"]})
                steps.append(await self._postprocess(step, **kwargs))
            return AgentFlowOutput(steps=steps, metrics=metrics)
        finally:
            episode.close()
