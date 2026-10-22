"""Tests for context management: shrinking old tool results.

Run with:  uv run pytest
"""

import copy
import math

from wakey_agent.client import Message
from wakey_agent.context import KEEP_RECENT, SHRINK_PLACEHOLDER, SHRINK_TARGET, shrink_context

LIMIT = 10_000  # pretend context size, in tokens
CHARS_PER_RESULT = 4_000  # each fake tool result ~1000 tokens by the //4 estimate
TOKENS_PER_RESULT = CHARS_PER_RESULT // 4


def make_conversation(n_rounds: int) -> list[Message]:
    """System prompt, task, then n_rounds of (assistant tool call, tool result)."""
    messages: list[Message] = [
        {"role": "system", "content": "You are an agent."},
        {"role": "user", "content": "Do the task."},
    ]
    for i in range(n_rounds):
        messages.append(
            {
                "role": "assistant",
                "content": f"step {i}",
                "tool_calls": [
                    {
                        "id": f"call_{i}",
                        "type": "function",
                        "function": {"name": "read_file", "arguments": '{"path": "f.c"}'},
                    }
                ],
            }
        )
        # Each result is distinguishable, so tests can tell which ones were cleared.
        messages.append(
            {"role": "tool", "tool_call_id": f"call_{i}", "content": str(i) * CHARS_PER_RESULT}
        )
    return messages


def tool_contents(messages: list[Message]) -> list[str]:
    return [m["content"] for m in messages if m["role"] == "tool"]


def cleared_flags(messages: list[Message]) -> list[bool]:
    return [c == SHRINK_PLACEHOLDER for c in tool_contents(messages)]


# ---------------------------------------------------------------------------


def test_clears_oldest_first_and_keeps_recent() -> None:
    n = KEEP_RECENT + 6
    messages = make_conversation(n)
    # Ask for far more than can be freed, so every eligible result gets cleared.
    shrink_context(messages, used=LIMIT * 10, limit=LIMIT)

    flags = cleared_flags(messages)
    assert flags[: n - KEEP_RECENT] == [True] * (n - KEEP_RECENT)
    assert flags[n - KEEP_RECENT :] == [False] * KEEP_RECENT


def test_stops_once_under_target() -> None:
    n = 12
    messages = make_conversation(n)
    used = LIMIT  # 100% full
    expected = min(n - KEEP_RECENT, math.ceil((used - SHRINK_TARGET * LIMIT) / TOKENS_PER_RESULT))

    shrink_context(messages, used=used, limit=LIMIT)

    flags = cleared_flags(messages)
    assert sum(flags) == expected
    assert flags[:expected] == [True] * expected  # and they are the oldest ones


def test_returns_lower_estimate() -> None:
    messages = make_conversation(KEEP_RECENT + 6)
    new_used = shrink_context(messages, used=LIMIT, limit=LIMIT)
    assert new_used < LIMIT
    assert new_used <= SHRINK_TARGET * LIMIT


def test_only_tool_contents_change() -> None:
    messages = make_conversation(KEEP_RECENT + 6)
    before = copy.deepcopy(messages)
    shrink_context(messages, used=LIMIT * 10, limit=LIMIT)

    assert len(messages) == len(before)
    for old, new in zip(before, messages, strict=True):
        if old["role"] == "tool":
            # Message kept, id kept: every tool_call_id still matches its call.
            assert new["role"] == "tool"
            assert new.get("tool_call_id") == old.get("tool_call_id")
        else:
            assert new == old  # system, user and assistant messages untouched


def test_already_cleared_results_are_skipped() -> None:
    messages = make_conversation(KEEP_RECENT + 6)
    once = shrink_context(messages, used=LIMIT * 10, limit=LIMIT)
    # Everything eligible is already a placeholder: nothing more to free.
    twice = shrink_context(messages, used=once, limit=LIMIT)
    assert twice == once


def test_too_few_results_changes_nothing() -> None:
    messages = make_conversation(KEEP_RECENT)
    before = copy.deepcopy(messages)
    new_used = shrink_context(messages, used=LIMIT * 10, limit=LIMIT)
    assert messages == before
    assert new_used == LIMIT * 10


def test_no_tool_results_at_all() -> None:
    messages = make_conversation(0)
    before = copy.deepcopy(messages)
    shrink_context(messages, used=LIMIT, limit=LIMIT)
    assert messages == before
