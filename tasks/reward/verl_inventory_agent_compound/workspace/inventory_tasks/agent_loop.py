"""veRL registration for the multi-turn inventory benchmark."""

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register

from .rollout import InventoryRollout


@register("inventory_agent")
class InventoryAgentLoop(AgentLoopBase):
    async def run(self, sampling_params, **kwargs):
        result = await InventoryRollout.run(self, sampling_params, **kwargs)
        return AgentLoopOutput(**result)
