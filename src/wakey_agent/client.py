import json
import os
import stat
from collections.abc import Iterable
from typing import TypedDict


class ToolCall(TypedDict):
    id: str | None
    name: str | None
    arguments: str


def check_private_file(path: str):
    if os.stat(path).st_mode & (stat.S_IRWXG | stat.S_IRWXO):
        raise PermissionError(f"access refused; to make your key private run: chmod 600 {path}")


def read_key_file(path: str) -> str:
    check_private_file(path)
    with open(path) as f:
        key = f.read().strip()
        if len(key) < 32:
            raise ValueError(f"{path}: key is too short")
        return key


def load_config(path: str = "~/.config/wakey/client.json"):
    with open(os.path.expanduser(path)) as f:
        data = json.load(f)
    url = data["url"]
    api_key = read_key_file(os.path.expanduser(data["api_key_file"]))
    return url, api_key


def get_headers(api_key: str) -> dict:
    return {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}


def get_payload(model: str, stream: bool) -> dict:
    return {
        "model": model,
        "stream": stream,
        "messages": [{"role": "user", "content": "What is the weather in Tokyo?"}],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the current temperature in a city",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                },
            }
        ],
    }


def parse_stream(lines: Iterable[str]) -> tuple[dict[int, ToolCall], str | None]:
    calls: dict[int, ToolCall] = {}
    finish_reason: str | None = None
    for line in lines:
        if not line.startswith("data: "):
            continue
        line = line.removeprefix("data: ")
        if line == "[DONE]":
            break
        chunk = json.loads(line)
        if not chunk.get("choices"):
            continue
        choice = chunk["choices"][0]

        if choice.get("finish_reason") is not None:
            finish_reason = choice["finish_reason"]

        for fragment in choice["delta"].get("tool_calls", []):
            func = fragment.get("function", {})
            idx = fragment["index"]
            if idx not in calls:
                calls[idx] = {"id": None, "name": None, "arguments": ""}
            entry = calls[idx]
            if fragment.get("id") is not None:
                entry["id"] = fragment["id"]
            if func.get("name") is not None:
                entry["name"] = fragment["function"]["name"]
            if func.get("arguments") is not None:
                entry["arguments"] += fragment["function"]["arguments"]
    return calls, finish_reason
