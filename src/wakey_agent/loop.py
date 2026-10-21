import argparse
import os
import sys
import time

import requests

from wakey_agent.client import Message, WakeyError, chat, load_config
from wakey_agent.tools import TOOLS, execute_tool_call, log_entry, new_log_file

CONTEXT_SIZE = 32768  # must match llama-server's -c setting on the backend
SHRINK_THRESHOLD = 0.75

SYSTEM_PROMPT: str = (
    "You're an agent working in a workspace directory, "
    "with tools for files and shell commands. "
    "All paths are relative to the workspace, "
    "and shell commands need the user's approval. "
    "Always check your work, for example, compile and run code you write. "
    "Always finish with writing a short summary of what you've done."
)


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
    auto_approve: bool,
    workspace: str,
):
    tools = [schema for schema, _ in TOOLS.values()]
    messages: list[Message] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task},
    ]

    log_path = new_log_file()

    for step in range(1, max_steps + 1):
        try:
            result = chat(url, api_key, model, messages, tools)
            print(
                f" · ctx {result.prompt_tokens + result.completion_tokens}/{CONTEXT_SIZE} \
                ({(result.prompt_tokens + result.completion_tokens) / CONTEXT_SIZE})"
            )
            messages.append(result.message)
        except (WakeyError, requests.RequestException) as e:
            print(f"Error: {e}", file=sys.stderr)
            return None

        if "tool_calls" not in result.message:
            return result.message["content"]

        for call in result.message["tool_calls"]:
            name = call["function"]["name"]
            raw_args = call["function"]["arguments"]
            print(f"→ {name}({raw_args[:80]})")

            result = execute_tool_call(call, workspace, auto_approve)
            print(preview(result))

            try:
                log_entry(
                    log_path,
                    {
                        "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
                        "step": step,
                        "tool": name,
                        "args": raw_args,
                        "result": result,
                    },
                )
            except OSError as e:
                print(f"Error: {e.strerror}: {log_path}", file=sys.stderr)

            messages.append({"role": "tool", "tool_call_id": call["id"], "content": result})

    print("Max steps reached", file=sys.stderr)


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
    parser.add_argument(
        "--workspace", type=str, default="~/throwaway", help="User defined workspace folder"
    )
    args = parser.parse_args()

    url, api_key = load_config()

    run_agent(
        url,
        api_key,
        args.max_steps,
        args.model,
        args.task,
        args.yes,
        os.path.expanduser(args.workspace),
    )


if __name__ == "__main__":
    main()
