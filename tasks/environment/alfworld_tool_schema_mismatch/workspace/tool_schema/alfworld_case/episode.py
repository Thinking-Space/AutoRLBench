"""Per-rollout state, shared by training, validation and integration tests."""
import json

from alfworld_case.tools import SYSTEM_PROMPT, ToolCallError, command_for


class Episode:
    def __init__(self, env, max_steps=40):
        if max_steps < 1:
            raise ValueError("max_steps must be positive")
        self.env = env
        self.max_steps = max_steps

    def reset(self, game_relative_path, task_id=None):
        self.observation, self.info = self.env.reset_with_info(game_relative_path, task_id)
        self.initial_observation = self.observation
        self.history = []
        self.num_steps = 0
        self.done = False
        self.success = False
        self.env_reward_sum = 0.0
        return self.observation

    def messages(self):
        messages = [{"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": self.initial_observation}]
        for turn in self.history:
            messages.extend(turn)
        return messages

    def act(self, calls, response_text=""):
        if self.done:
            raise RuntimeError("Episode already ended")
        self.num_steps += 1
        env_reward = 0.0
        valid = False
        error = None
        try:
            if len(calls) != 1:
                raise ToolCallError("invalid_tool_call: expected exactly one call")
            name, arguments = calls[0]
            command = command_for(name, arguments)
            valid = command in self.info.get("admissible_commands", [])
            self.observation, env_reward, done, self.info = self.env.step(command)
            self.success = bool(self.info.get("success", self.info.get("won", False)))
            self.done = bool(done or self.success)
            feedback = self.observation
        except ToolCallError as exc:
            error = str(exc)
            feedback = error + ". The environment state did not change."
        self.env_reward_sum += float(env_reward)
        self.done = self.done or self.num_steps >= self.max_steps
        # Preserve standard assistant/tool roles, using only model-visible tool names.
        if calls:
            tool_calls = []
            for index, (name, arguments) in enumerate(calls):
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except ValueError:
                        pass
                tool_calls.append({"id": f"call_{self.num_steps}_{index}", "type": "function",
                                   "function": {"name": name, "arguments": arguments}})
            turn = [{"role": "assistant", "content": "", "tool_calls": tool_calls}]
            turn.extend({"role": "tool", "tool_call_id": call["id"], "content": feedback}
                        for call in tool_calls)
        else:
            turn = [{"role": "assistant", "content": response_text},
                    {"role": "user", "content": feedback}]
        self.history.append(turn)
        return {"success": self.success, "score": float(self.done and self.success),
                "num_steps": self.num_steps, "is_action_valid": valid,
                "step_env_reward": float(env_reward), "dense_reward_sum": self.env_reward_sum,
                "tool_error": error or "", "done": self.done, "feedback": feedback}

    def close(self):
        self.env._close_env()
