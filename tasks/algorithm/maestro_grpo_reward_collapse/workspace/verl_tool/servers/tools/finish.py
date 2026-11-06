"""Finish a trajectory and release its tool environments.

Restored from the source project's finish tool, which the Ray manager always
initializes alongside the configured execution tools.
"""
import ray

from .base import BaseTool, register_tool


@register_tool
class FinishTool(BaseTool):
    tool_type = "finish"
    timeout = 10

    def __init__(self, num_workers=1, other_tools=None):
        super().__init__(num_workers)
        self.other_tools = other_tools if other_tools is not None else {}

    def get_usage_inst(self):
        return ""

    def parse_action(self, action):
        # The manager routes here only for an explicit finish=True request.
        return "", True

    def conduct_action(self, trajectory_id, action, extra_field):
        for tool in self.other_tools.values():
            if isinstance(tool, ray.actor.ActorHandle):
                if ray.get(tool.has_env.remote(trajectory_id)):
                    ray.get(tool.delete_env.remote(trajectory_id))
            elif tool.has_env(trajectory_id):
                tool.delete_env(trajectory_id)
        return "", True, True
