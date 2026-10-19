import json
import os
import subprocess

from wakey_agent.client import ToolCallMessage


def build_argv(command: str, workspace: str) -> list[str]:
    return ["bash", "-c", command]


READ_FILE_SCHEMA = {
    "type": "function",
    "function": {
        "name": "read_file",
        "description": "Read a file from the workspace and return its contents. "
        "Paths are relative to the workspace root.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "e.g. 'src/main.c'"},
            },
            "required": ["path"],
        },
    },
}


def read_file(path: str, workspace: str) -> str:
    if os.path.isabs(path):
        return "Error: paths must be relative to the workspace"

    try:
        with open(os.path.join(workspace, path)) as f:
            data = f.read()
            return data
    except (OSError, UnicodeDecodeError) as e:
        return f"Error: {e}"


WRITE_FILE_SCHEMA = {
    "type": "function",
    "function": {
        "name": "write_file",
        "description": "Overwrite to an file from the workspace, or create and write to it if it "
        "doesn't exist and return number of bytes written. Paths are relative to the "
        "workspace root.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "e.g. 'src/main.c'"},
                "content": {"type": "string", "description": "e.g. 'printf('Hello World')'"},
            },
            "required": ["path"],
        },
    },
}


def write_file(path: str, content: str, workspace: str) -> str:
    if os.path.isabs(path):
        return "Error: paths must be relative to the workspace"

    try:
        if os.path.dirname(path) != "":
            os.makedirs(os.path.dirname(os.path.join(workspace, path)), exist_ok=True)
        with open(os.path.join(workspace, path), "w") as f:
            f.write(content)
        return f"Wrote {len(content.encode('utf-8'))} bytes to {path}"
    except (OSError, UnicodeDecodeError) as e:
        return f"Error: {e}"


RUN_SHELL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "run_shell",
        "description": "Run a shell command in workspace and return status code, stdout, stderr "
        "Paths are relative to the workspace root.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "e.g. 'ls -l'"},
            },
            "required": ["path"],
        },
    },
}


def run_shell(command: str, workspace: str) -> str:
    ans = input(f"Run '{command}'? [y/N]: ")
    if ans.lower() in ("y", "yes"):
        try:
            res = subprocess.run(
                args=build_argv(command, workspace),
                cwd=workspace,
                capture_output=True,
                timeout=60,
                text=True,
            )
            return f"exit code: {res.returncode}\nstdout:\n{res.stdout}\nstderr:\n{res.stderr}"
        except (subprocess.TimeoutExpired, OSError) as e:
            return f"Error: {e}"

    else:
        return "User declined to run this command"


TOOLS = {
    "read_file": (READ_FILE_SCHEMA, read_file),
    "write_file": (WRITE_FILE_SCHEMA, write_file),
    "run_shell": (RUN_SHELL_SCHEMA, run_shell),
}


def execute_tool_call(call: ToolCallMessage, workspace: str) -> str:
    name = call["function"]["name"]
    raw_args = call["function"]["arguments"]

    if name not in TOOLS:
        return f"Error: unknown tool {name!r}"
    _, tool_func = TOOLS[name]

    try:
        args = json.loads(raw_args)
    except json.JSONDecodeError:
        return f"Error: arguments are not valid JSON: {raw_args!r}"
    if not isinstance(args, dict):
        return f"Error: arguments must be a JSON object, got {type(args).__name__}"

    try:
        return tool_func(**args, workspace=workspace)
    except TypeError as e:
        return f"Error: bad arguments for {name}: {e}"


def make_call(name: str, arguments: str) -> ToolCallMessage:
    return {
        "id": "test",
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


def main():
    workspace = "~/throwaway"
    workspace = os.path.expanduser(workspace)

    cases = [
        make_call("read_file", json.dumps({"path": "test.txt"})),
        make_call("scratch_head", json.dumps({"path": "test.txt"})),
        make_call("read_file", "{path: test.txt"),
        make_call("read_file", json.dumps(["test.txt"])),
        make_call("read_file", json.dumps({})),
        make_call("read_file", json.dumps({"path": "test.txt", "workspace": "/"})),
        make_call("read_file", json.dumps({"file": "test.txt"})),
    ]

    for call in cases:
        print(call["function"], "->", execute_tool_call(call, workspace))


if __name__ == "__main__":
    main()
