import argparse
import json
import os
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

from wakey_agent.client import Message, WakeyError, chat, load_config
from wakey_agent.context import CONTEXT_SIZE, SHRINK_THRESHOLD, shrink_context
from wakey_agent.tools import (
    TOOLS,
    Approver,
    always_approve,
    execute_tool_call,
    log_entry,
    new_log_file,
)

type Event = dict[str, Any]
type Emit = Callable[[Event], None]


@dataclass
class RunResult:
    answer: str | None
    stop_reason: str
    steps: int
    tokens: int


SYSTEM_PROMPT: str = (
    "You're an agent working in a workspace directory, "
    "with tools for files and shell commands. "
    "All paths are relative to the workspace, "
    "and shell commands need the user's approval. "
    "Always check your work, for example, compile and run code you write. "
    "Always finish with writing a short summary of what you've done."
)


class Session:
    def __init__(
        self,
        url: str,
        api_key: str,
        model: str,
        workspace: str,
        approve: Approver,
        emit: Emit,
        max_steps: int = 20,
        context_size: int = CONTEXT_SIZE,
    ) -> None:
        self.url = url
        self.api_key = api_key
        self.model = model
        self.workspace = workspace
        self.approve = approve
        self.emit = emit
        self.max_steps = max_steps
        self.context_size = context_size
        self.messages: list[Message] = [{"role": "system", "content": SYSTEM_PROMPT}]
        self.log_path = new_log_file()
        self.used = 0
        self.tools = [schema for schema, _ in TOOLS.values()]

    def _run_turn(self, text: str) -> RunResult:
        self.messages.append({"role": "user", "content": text})
        for step in range(1, self.max_steps + 1):
            self.emit(
                {
                    "type": "step",
                    "step": step,
                    "max": self.max_steps,
                    "used": self.used,
                    "limit": self.context_size,
                }
            )
            try:
                result = chat(self.url, self.api_key, self.model, self.messages, self.tools)
            except (WakeyError, requests.RequestException) as e:
                self.emit({"type": "error", "message": str(e)})
                self.emit({"type": "done", "reason": "error", "steps": step, "tokens": self.used})
                return RunResult(answer=None, stop_reason="error", steps=step, tokens=self.used)

            self.used = result.prompt_tokens + result.completion_tokens
            self.emit({"type": "usage", "used": self.used, "limit": self.context_size})

            if self.used > SHRINK_THRESHOLD * self.context_size:
                self.used = shrink_context(self.messages, self.used, self.context_size)
            self.messages.append(result.message)

            if "tool_calls" not in result.message:
                self.emit({"type": "done", "reason": "done", "steps": step, "tokens": self.used})
                return RunResult(
                    answer=result.message["content"],
                    stop_reason="done",
                    steps=step,
                    tokens=self.used,
                )

            for call in result.message["tool_calls"]:
                name = call["function"]["name"]
                raw_args = call["function"]["arguments"]
                self.emit({"type": "tool_call", "name": name, "args": raw_args})

                output = execute_tool_call(call, self.workspace, self.approve)
                self.emit({"type": "tool_result", "output": output})

                try:
                    log_entry(
                        self.log_path,
                        {
                            "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
                            "step": step,
                            "tool": name,
                            "args": raw_args,
                            "result": output,
                        },
                    )
                except OSError as e:
                    self.emit({"type": "warning", "message": f"could not write log: {e.strerror}"})

                self.messages.append(
                    {"role": "tool", "tool_call_id": call["id"], "content": output}
                )
        self.emit(
            {"type": "done", "reason": "max_steps", "steps": self.max_steps, "tokens": self.used}
        )
        return RunResult(
            answer=None, stop_reason="max_steps", steps=self.max_steps, tokens=self.used
        )

    def send(self, text: str) -> RunResult:
        start = len(self.messages)
        try:
            return self._run_turn(text)
        except KeyboardInterrupt:
            del self.messages[start:]
            self.emit({"type": "done", "reason": "cancelled", "steps": 0, "tokens": self.used})
            return RunResult(answer=None, stop_reason="cancelled", steps=0, tokens=self.used)


class ProtocolApprover:
    """Asks the frontend over stdin/stdout whether to run a command."""

    def __init__(self) -> None:
        self.next_id = 1

    def __call__(self, command: str) -> bool:
        request_id = self.next_id
        self.next_id += 1
        emit_json({"type": "approve", "id": request_id, "command": command})

        while True:
            line = sys.stdin.readline()
            if not line:
                return False
            try:
                line_json = json.loads(line)
            except json.JSONDecodeError:
                emit_json({"type": "warning", "message": "Invalid JSON"})
                continue
            if not isinstance(line_json, dict):
                emit_json({"type": "warning", "message": "ignored non-object message"})
                continue
            if line_json.get("type") == "approval" and line_json.get("id") == request_id:
                return line_json.get("ok") is True
            emit_json(
                {
                    "type": "warning",
                    "message": f"ignored message while waiting for approval {request_id}: {line.strip()[:80]}",
                }
            )


def emit_json(event: Event) -> None:
    try:
        sys.stdout.write(json.dumps(event, default=str) + "\n")
        sys.stdout.flush()
    except BrokenPipeError:
        # The frontend is gone. Point stdout at /dev/null so Python's final flush
        # at exit doesn't fail again, then end quietly.
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, sys.stdout.fileno())
        sys.exit(0)


def print_event(event: Event) -> None:
    match event["type"]:
        case "text":
            print(event["delta"], end="", flush=True)
        case "tool_call":
            print(f"→ {event['name']}({event['args'][:80]})")
        case "tool_result":
            print(preview(event["output"]))
        case "error":
            print(f"Error: {event['message']}", file=sys.stderr)
        case "warning":
            print(f"Warning: {event['message']}", file=sys.stderr)
        case "step":
            line = f"--- step {event['step']}/{event['max']}"
            if event["used"]:
                line += f" · ctx {event['used']}/{event['limit']} ({event['used'] / event['limit']:.0%})"
            print(line)
        case "done":
            if event["reason"] == "max_steps":
                print("Max steps reached", file=sys.stderr)
        case "text_end":
            print()
        case _:
            pass


def preview(text: str, max_lines: int = 3) -> str:
    lines = text.splitlines()
    shown = "\n".join(lines[:max_lines])
    if len(lines) > max_lines:
        shown += f"\n  ... ({len(lines) - max_lines} more lines)"
    return shown


def run_agent(
    url: str,
    api_key: str,
    max_steps: int,
    model: str,
    task: str,
    approve: Approver,
    workspace: str,
    emit: Emit,
) -> RunResult:
    session = Session(url, api_key, model, workspace, approve, emit, max_steps)
    return session.send(task)


def main():
    parser = argparse.ArgumentParser(description="Loop tool calls")
    parser.add_argument(
        "--max-steps", type=int, default=20, help="Maximum amount of model calls (default: 20)"
    )
    parser.add_argument("--model", type=str, default="gemma-4-12b", help="model name to request")
    parser.add_argument("--task", type=str, default="", help="User defined task for the model")
    parser.add_argument(
        "--yes",
        action="store_true",
        help="run shell commands without asking for confirmation",
    )
    parser.add_argument("--workspace", type=str, default=None, help="User defined workspace folder")
    parser.add_argument("--chat", action="store_true", help="just to test things out")
    parser.add_argument(
        "--serve", action="store_true", help="Interface with wakeyctl in agent mode"
    )
    parser.add_argument(
        "--context-size", type=int, default=CONTEXT_SIZE, help="maximum context size in tokens"
    )
    args = parser.parse_args()

    url, api_key = load_config()

    ws = Path(args.workspace).expanduser().resolve() if args.workspace else Path.cwd()

    emit = emit_json if args.serve else print_event

    print(f"--- wakey-agent started {time.strftime('%Y-%m-%d %H:%M:%S')} in {ws}", file=sys.stderr)

    if ws in (Path("/"), Path.home(), Path.home().parent):
        emit({"type": "error", "message": f"refusing to use {ws} as the workspace"})
        sys.exit(1)
    if not ws.is_dir():
        emit({"type": "error", "message": f"Workspace {ws} is not a directory."})
        sys.exit(1)

    approve = always_approve if args.yes else ProtocolApprover()
    session = Session(
        url,
        api_key,
        args.model,
        str(ws),
        approve=approve,
        emit=emit_json,
        max_steps=args.max_steps,
    )

    emit_json({"type": "ready", "model": args.model, "workspace": str(ws)})

    while line := sys.stdin.readline():
        msg = json.loads(line)
        if msg["type"] == "user":
            session.send(msg["content"])


if __name__ == "__main__":
    main()
