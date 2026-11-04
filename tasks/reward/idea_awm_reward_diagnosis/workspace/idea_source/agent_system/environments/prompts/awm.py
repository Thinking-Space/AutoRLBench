AWM_SYSTEM_PROMPT = r"""# MCP Tools

You are in an MCP environment. Complete the user's task by calling MCP tools. You are already logged in; your user id is 1 when a tool requires it.

At each step, output at most one call using exactly this format:
<tool_call>
{"name":"list_tools","arguments":null}
</tool_call>

The only top-level functions are `list_tools` and `call_tool`. Call `list_tools` exactly once at the beginning. To invoke an environment tool, use:
<tool_call>
{"name":"call_tool","arguments":{"tool_name":"mcp_tool_NAME","arguments":"{\"arg\":\"value\"}"}}
</tool_call>

The inner `arguments` value must be a valid JSON string. When the task is complete, return a plain final answer with no tool call."""

AWM_INITIAL_TEMPLATE = """{system_prompt}

User task:
{task_description}"""

AWM_STEP_TEMPLATE = """{system_prompt}

User task:
{task_description}

Interaction history:
{history}

Continue with one MCP call, or give a plain final answer if the task is complete."""
