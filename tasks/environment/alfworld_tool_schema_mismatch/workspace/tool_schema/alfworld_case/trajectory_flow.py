"""One training row per episode, with append-only sampling token histories."""
from uuid import uuid4

from agent_r1.agent_flow.agent_flow import AgentFlowOutput, AgentFlowStep, register
from recipes.alfworld.alfworld_agent_flow import _recover_tool_calls_from_text
from recipes.alfworld.env.alfworld_wrapper import AlfworldTextworldEnv
from verl.utils.profiler import simple_timer

from alfworld_case.episode import Episode
from alfworld_case.flow import SchemaAgentFlow


def feedback_suffix(tokenizer, feedback, generated_ids, template_kwargs):
    # Derive role delimiters from the model template, without re-encoding sampled tokens.
    prefix = [{"role": "user", "content": "template anchor"},
              {"role": "assistant", "content": "template response"}]
    full = tokenizer.apply_chat_template(prefix + [{"role": "tool", "content": feedback}],
                                         tokenize=False, add_generation_prompt=True, **template_kwargs)
    # vLLM may return EOS, or stop at the per-action token budget without EOS.
    end = tokenizer.eos_token
    marker = "template response"
    boundary_pos = full.find(end, full.index(marker) + len(marker)) if end else -1
    if boundary_pos < 0:
        raise ValueError("Unsupported assistant turn terminator; expected Qwen chat template")
    suffix = full[boundary_pos + len(end):]
    boundary = "" if generated_ids[-1] == tokenizer.eos_token_id else end
    return tokenizer.encode(boundary + suffix, add_special_tokens=False)


@register("alfworld_schema")
class TrajectoryAgentFlow(SchemaAgentFlow):
    def __init__(self, *args, max_action_tokens=256, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_action_tokens = max_action_tokens
        if max_action_tokens < 1:
            raise ValueError("max_action_tokens must be positive")

    async def run(self, sampling_params, **kwargs):
        info = kwargs.get("extra_info") or {}
        episode = Episode(AlfworldTextworldEnv(max_episode_steps=self.max_steps), self.max_steps)
        metrics, trace = {}, []
        response, mask, logprobs = [], [], []
        have_logprobs = True
        budget = self.config.actor_rollout_ref.rollout.response_length
        try:
            episode.reset(info["game_relative_path"], info.get("task_id"))
            initial = await self.apply_chat_template(episode.messages(), tools=self.schemas)
            if len(initial) > self.config.actor_rollout_ref.rollout.prompt_length:
                raise ValueError("Initial ALFWorld prompt exceeds prompt budget")
            if budget < self.max_action_tokens:
                raise ValueError("Trajectory budget must fit one complete action budget")
            budget_exhausted = False
            while not episode.done:
                prompt = initial + response
                with simple_timer("generate_sequences", metrics):
                    output = await self.server_manager.generate(
                        request_id=uuid4().hex, prompt_ids=prompt,
                        sampling_params={**sampling_params, "max_tokens": self.max_action_tokens})
                ids = list(output.token_ids)
                if not ids or len(ids) > self.max_action_tokens:
                    raise ValueError("Model returned empty or over-budget action tokens")
                if output.log_probs is not None and len(output.log_probs) != len(ids):
                    raise ValueError("Sampled log-probabilities do not align with action tokens")
                start = len(response)
                response.extend(ids)
                mask.extend([1] * len(ids))
                have_logprobs = have_logprobs and output.log_probs is not None
                logprobs.extend(output.log_probs if output.log_probs is not None else [0.0] * len(ids))
                text = self.tokenizer.decode(ids, skip_special_tokens=True)
                _, calls = await self.tool_parser.extract_tool_calls(ids)
                if not calls:
                    calls = _recover_tool_calls_from_text(text)
                with simple_timer("tool_calls", metrics):
                    result = episode.act([(call.name, call.arguments) for call in calls], text)
                trace.append({"response_start": start, "response_end": len(response),
                              "response": text, "feedback": result["feedback"],
                              "tool_error": result["tool_error"], "success": result["success"]})
                if episode.done:
                    break
                suffix = feedback_suffix(self.tokenizer, result["feedback"], ids,
                                         self.apply_chat_template_kwargs)
                # Never rewrite old context or cut a sampled action. Finish at this boundary.
                if len(response) + len(suffix) + self.max_action_tokens > budget:
                    budget_exhausted = True
                    break
                response.extend(suffix)
                mask.extend([0] * len(suffix))
                logprobs.extend([0.0] * len(suffix))
            reward_info = {key: info.get(key, "") for key in
                           ("task_id", "split", "task_type_raw", "task_family")}
            reward_info.update({key: result[key] for key in
                                ("success", "num_steps", "is_action_valid",
                                 "step_env_reward", "dense_reward_sum")})
            step = AgentFlowStep(
                prompt_ids=initial, response_ids=response, response_mask=mask,
                response_logprobs=logprobs if have_logprobs else None,
                reward_score=None, num_turns=episode.num_steps,
                extra_fields={"reward_extra_info": reward_info, "trajectory_trace": trace,
                              "context_budget_exhausted": budget_exhausted,
                              "tool_feedback": result["feedback"], "tool_error": result["tool_error"]})
            return AgentFlowOutput(steps=[await self._postprocess(step, **kwargs)], metrics=metrics)
        finally:
            episode.close()
