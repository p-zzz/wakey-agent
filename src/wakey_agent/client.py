"""Client for Wakey's OpenAI-compatible API: config, requests and stream parsing."""

import json
import os
import stat
from collections.abc import Iterable
from typing import Any, NotRequired, TypedDict

import requests

# Message types (OpenAI chat format)


class FunctionCall(TypedDict):
    name: str
    arguments: str  # a JSON *string*, not a dict


class ToolCallMessage(TypedDict):
    id: str
    type: str  # always "function"
    function: FunctionCall


class TextMessage(TypedDict):  # system and user messages
    role: str
    content: str


class AssistantMessage(TypedDict):
    role: str
    content: str
    tool_calls: NotRequired[list[ToolCallMessage]]  # absent when there are no calls


class ToolResultMessage(TypedDict):
    role: str
    tool_call_id: str
    content: str


type Message = TextMessage | AssistantMessage | ToolResultMessage


class _PartialToolCall(TypedDict):
    """A tool call while its fragments are still arriving from the stream."""

    id: str | None
    name: str | None
    arguments: str


class WakeyError(Exception):
    """Wakey or the backend answered with a non-200 status."""


# Config


def check_private_file(path: str) -> None:
    if os.stat(path).st_mode & (stat.S_IRWXG | stat.S_IRWXO):
        raise PermissionError(f"access refused; to make your key private run: chmod 600 {path}")


def read_key_file(path: str) -> str:
    check_private_file(path)
    with open(path) as f:
        key = f.read().strip()
    if len(key) < 32:
        raise ValueError(f"{path}: key is too short")
    return key


def load_config(path: str = "~/.config/wakey/client.json") -> tuple[str, str]:
    with open(os.path.expanduser(path)) as f:
        data = json.load(f)
    url = data["url"]
    api_key = read_key_file(os.path.expanduser(data["api_key_file"]))
    return url, api_key


# Requests


def get_headers(api_key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}


def get_payload(
    model: str, messages: list[Message], tools: list[dict[str, Any]], stream: bool
) -> dict[str, Any]:
    return {"model": model, "stream": stream, "messages": messages, "tools": tools}


def parse_stream(lines: Iterable[str]) -> tuple[AssistantMessage, str | None]:
    """Turn SSE lines into one assistant message, printing text as it arrives.

    Returns the message in the same shape the server uses without streaming,
    so it can be appended to the messages list as is, plus the finish reason.
    """
    calls: dict[int, _PartialToolCall] = {}
    content = ""
    finish_reason: str | None = None

    for line in lines:
        if not line.startswith("data: "):
            continue
        line = line.removeprefix("data: ")
        if line == "[DONE]":
            break
        chunk = json.loads(line)
        if not chunk.get("choices"):  # usage/timings-only chunks
            continue
        choice = chunk["choices"][0]

        if choice.get("finish_reason") is not None:
            finish_reason = choice["finish_reason"]

        delta = choice.get("delta", {})

        text = delta.get("content")
        if text:
            print(text, end="", flush=True)
            content += text

        for fragment in delta.get("tool_calls", []):
            func = fragment.get("function", {})
            idx = fragment["index"]
            if idx not in calls:
                calls[idx] = {"id": None, "name": None, "arguments": ""}
            entry = calls[idx]
            if fragment.get("id") is not None:
                entry["id"] = fragment["id"]
            if func.get("name") is not None:
                entry["name"] = func["name"]
            if func.get("arguments") is not None:
                entry["arguments"] += func["arguments"]

    if content:
        print()  # end the streamed text with a newline

    # Convert the accumulator into the wire format, in index order. A missing
    # id gets a fallback; a missing name becomes "" so the loop reports it to
    # the model as an unknown tool instead of silently dropping the call.
    tool_calls: list[ToolCallMessage] = []
    for idx, c in sorted(calls.items()):
        tool_calls.append(
            {
                "id": c["id"] or f"call_{idx}",
                "type": "function",
                "function": {"name": c["name"] or "", "arguments": c["arguments"]},
            }
        )

    message: AssistantMessage = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return message, finish_reason


def chat(
    url: str,
    api_key: str,
    model: str,
    messages: list[Message],
    tools: list[dict[str, Any]],
) -> tuple[AssistantMessage, str | None]:
    """Send the conversation, stream the reply, return (assistant message, finish reason).

    Raises WakeyError on a non-200 answer (e.g. 503 when the PC is down), with
    Wakey's response body in the message, and requests.RequestException on
    network errors.
    """
    with requests.post(
        f"{url}/v1/chat/completions",
        json=get_payload(model, messages, tools, stream=True),
        headers=get_headers(api_key),
        timeout=(5, 120),
        stream=True,
    ) as r:
        if r.status_code != 200:
            raise WakeyError(f"HTTP {r.status_code}: {r.text.strip()}")
        lines = (raw.decode("utf-8") for raw in r.iter_lines())
        return parse_stream(lines)
