#!/usr/bin/env python3
"""Unit tests for AWM action projection / format validation."""

from agent_system.environments.env_package.awm.projection import awm_projection


VALID_CALL = """<tool_call>
{"name":"list_tools","arguments":null}
</tool_call>"""

MALFORMED_CALL = """<tool_call>
{"name":"list_tools"
</tool_call>"""

IN_THINK_CALL = """<tool_call>
{"name":"call_tool","arguments":{"tool_name":"create_item","arguments":"{}"}}
</tool_call>"""


def test_missing_think_is_format_error_when_required():
    projected, valid = awm_projection([VALID_CALL], require_think=True)
    assert projected == [{"name": "invalid", "content": VALID_CALL}]
    assert valid == [0]


def test_closed_think_keeps_valid_tool_call():
    action = f"</think>\n{VALID_CALL}"
    projected, valid = awm_projection([action], require_think=True)
    assert projected == [{"name": "list_tools", "arguments": None}]
    assert valid == [1]


def test_closed_think_malformed_tool_call_still_invalid():
    action = f"</think>\n{MALFORMED_CALL}"
    projected, valid = awm_projection([action], require_think=True)
    assert projected[0]["name"] == "invalid"
    assert valid == [0]


def test_closed_think_plain_final_is_valid():
    action = "</think>\nThe appointment is booked."
    projected, valid = awm_projection([action], require_think=True)
    assert projected == [{"name": "final", "content": "The appointment is booked."}]
    assert valid == [1]


def test_think_not_required_allows_tool_call_without_close_tag():
    projected, valid = awm_projection([VALID_CALL], require_think=False)
    assert projected == [{"name": "list_tools", "arguments": None}]
    assert valid == [1]


def test_think_close_tag_is_case_insensitive():
    action = f"</THINK>\n{VALID_CALL}"
    projected, valid = awm_projection([action], require_think=True)
    assert projected == [{"name": "list_tools", "arguments": None}]
    assert valid == [1]


def test_tool_calls_inside_think_are_ignored():
    action = (
        f"<think>example:\n{IN_THINK_CALL}\nand another:\n{VALID_CALL}\n</think>\n{VALID_CALL}"
    )
    projected, valid = awm_projection([action], require_think=True)
    assert projected == [{"name": "list_tools", "arguments": None}]
    assert valid == [1]


def test_tool_call_only_inside_think_is_final_after_close():
    action = f"<think>{VALID_CALL}</think>\nThe appointment is booked."
    projected, valid = awm_projection([action], require_think=True)
    assert projected == [{"name": "final", "content": "The appointment is booked."}]
    assert valid == [1]


def test_multiple_tool_calls_after_think_still_invalid():
    action = f"</think>\n{VALID_CALL}\n{VALID_CALL}"
    projected, valid = awm_projection([action], require_think=True)
    assert projected[0]["name"] == "invalid"
    assert valid == [0]


def test_think_present_without_require_still_parses_after_close():
    action = f"<think>{IN_THINK_CALL}</think>\n{VALID_CALL}"
    projected, valid = awm_projection([action], require_think=False)
    assert projected == [{"name": "list_tools", "arguments": None}]
    assert valid == [1]
