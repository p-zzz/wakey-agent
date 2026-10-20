import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from wakey_agent.client import ToolCallMessage


# fmt: off
def build_argv(command: str, workspace: str) -> list[str]:
    """Wrap a shell command in a bubblewrap sandbox rooted at the workspace."""
    return [
        "bwrap",
        # system files: read-only
        "--ro-bind", "/usr", "/usr",
        "--symlink", "usr/bin", "/bin",
        "--symlink", "usr/lib", "/lib",
        "--symlink", "usr/lib", "/lib64",
        "--proc", "/proc",
        "--dev", "/dev",
        "--tmpfs", "/tmp",
        # the only writable place, at a neutral path
        "--bind", workspace, "/workspace",
        "--chdir", "/workspace",
        # clean environment: nothing inherited from the user's shell
        "--clearenv",
        "--setenv", "PATH", "/usr/bin",
        "--setenv", "HOME", "/workspace",
        "--setenv", "LANG", "C.UTF-8",
        # isolation: no network, own PIDs, own session
        "--unshare-all",
        "--new-session",
        "--die-with-parent",
        "bash", "-c", command,
    ]
# fmt: on


def resolve_path(path: str, workspace: str) -> Path:
    ws = Path(workspace).resolve()
    target = (ws / path).resolve()
    if not target.is_relative_to(ws):
        raise ValueError(f"path {path!r} is outside the workspace")
    return target


LOG_DIR = Path("~/.local/state/wakey-agent/").expanduser()
LOG_RETENTION_DAYS = 7


def cleanup_logs() -> None:
    """Cleanup logs that are over LOG_RETENTION_DAYS old"""
    cutoff = time.time() - LOG_RETENTION_DAYS * 24 * 3600
    for p in LOG_DIR.glob("*.jsonl"):
        try:
            if p.stat().st_mtime < cutoff:
                p.unlink()
        except OSError as e:
            print(f"Warning: could not delete old log {e}", file=sys.stderr)


def new_log_file() -> Path:
    """Create the log directory and an empty, private log file for this run"""
    LOG_DIR.mkdir(mode=0o700, parents=True, exist_ok=True)
    cleanup_logs()
    log_path = LOG_DIR / f"{time.strftime('%Y-%m-%dT%H-%M-%S')}.jsonl"
    log_path.touch(mode=0o600)
    return log_path


def log_entry(log_path: Path, entry: dict[str, Any]) -> None:
    """Append one JSON object as a line"""
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


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
    try:
        return resolve_path(path, workspace).read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return f"Error: {path} is not a UTF-8 text file (binary?)"
    except ValueError as e:
        return f"Error: {e}"
    except OSError as e:
        return f"Error: {e.strerror}: {path}"


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
    try:
        target: Path = resolve_path(path, workspace)
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w") as f:
            f.write(content)
        return f"Wrote {len(content.encode('utf-8'))} bytes to {path}"
    except ValueError as e:
        return f"Error: {e}"
    except OSError as e:
        return f"Error: {e.strerror}: {path}"


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


def run_shell(command: str, workspace: str, auto_approve=False) -> str:
    if not auto_approve:
        ans = input(f"Run '{command}'? [y/N]: ")
        if ans.lower() not in ("y", "yes"):
            return "User declined to run this command"

    try:
        res = subprocess.run(
            args=build_argv(command, workspace),
            cwd=workspace,
            capture_output=True,
            timeout=60,
            text=True,
            stdin=subprocess.DEVNULL,
        )
        return f"exit code: {res.returncode}\nstdout:\n{res.stdout}\nstderr:\n{res.stderr}"
    except (subprocess.TimeoutExpired, OSError) as e:
        return f"Error: {e}"


TOOLS = {
    "read_file": (READ_FILE_SCHEMA, read_file),
    "write_file": (WRITE_FILE_SCHEMA, write_file),
    "run_shell": (RUN_SHELL_SCHEMA, run_shell),
}

TOOL_OUTPUT_CAP = 4000


def truncate(text: str, max_chars: int = TOOL_OUTPUT_CAP) -> str:
    text_len = len(text)
    if text_len <= max_chars:
        return text

    n = max_chars // 2
    text = text[:n] + f"\n[... {text_len - 2 * n} characters omitted ...]\n" + text[-n:]
    return text


def execute_tool_call(call: ToolCallMessage, workspace: str, auto_approve: bool = False) -> str:
    name = call["function"]["name"]
    raw_args = call["function"]["arguments"]
    kwargs: dict[str, str | bool] = {"workspace": workspace}

    if name == "run_shell":
        kwargs["auto_approve"] = auto_approve

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
        return truncate(tool_func(**args, **kwargs))
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
        # escape cases
        make_call("read_file", json.dumps({"path": "../.bashrc"})),
        make_call("read_file", json.dumps({"path": "/etc/passwd"})),
        make_call("read_file", json.dumps({"path": "../throwaway2/x"})),
        make_call("read_file", json.dumps({"path": "root/etc/passwd"})),
        make_call("read_file", json.dumps({"path": "alias.c"})),
        make_call("write_file", json.dumps({"path": "../escape.txt", "content": "x"})),
        make_call("write_file", json.dumps({"path": "root/tmp/escape.txt", "content": "x"})),
    ]

    for call in cases:
        print(call["function"], "->", execute_tool_call(call, workspace))


if __name__ == "__main__":
    main()
