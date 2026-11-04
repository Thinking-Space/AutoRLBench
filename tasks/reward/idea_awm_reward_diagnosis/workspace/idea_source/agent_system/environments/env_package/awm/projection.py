import json
import re
from typing import List, Optional, Tuple


_TOOL_CALL_PATTERN = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
_THINK_CLOSE = "</think>"


def _payload_after_think(action: str, require_think: bool) -> Optional[str]:
    """Return the text that may contain a tool call / final answer.

    Tool-call extraction only looks after the first ``</think>``.  Content
    inside the reasoning block is ignored.  A missing close tag is a format
    failure when thinking is required.
    """
    close_idx = action.lower().find(_THINK_CLOSE)
    if close_idx == -1:
        if require_think:
            return None
        return action
    return action[close_idx + len(_THINK_CLOSE) :]


def awm_projection(actions: List[str], require_think: bool = False) -> Tuple[List[dict], List[int]]:
    """Parse the single MCP meta-tool call emitted by the AWM agent prompt.

    When thinking is enabled, an incomplete reasoning block is a format
    failure, just like malformed tool-call JSON.  Tool-call / final-answer
    parsing uses only the suffix after ``</think>``.
    """
    projected = []
    valid = []
    for action in actions:
        payload = _payload_after_think(action, require_think=require_think)
        if payload is None:
            projected.append({"name": "invalid", "content": action})
            valid.append(0)
            continue
        matches = _TOOL_CALL_PATTERN.findall(payload)
        if not matches:
            content = payload.strip()
            projected.append({"name": "final", "content": content})
            valid.append(1 if content else 0)
            continue
        try:
            call = json.loads(matches[0])
        except (json.JSONDecodeError, TypeError):
            projected.append({"name": "invalid", "content": action})
            valid.append(0)
            continue
        name = call.get("name", "") if isinstance(call, dict) else ""
        arguments = call.get("arguments") if isinstance(call, dict) else None
        if name not in {"list_tools", "call_tool"} or len(matches) != 1:
            projected.append({"name": "invalid", "content": action})
            valid.append(0)
        else:
            projected.append({"name": name, "arguments": arguments})
            valid.append(1)
    return projected, valid
