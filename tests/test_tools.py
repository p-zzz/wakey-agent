"""Tests for the agent's tools: workspace confinement, error handling and the sandbox.

Run with:  uv run pytest
"""

import json
import shutil
from pathlib import Path

import pytest

from wakey_agent.client import ToolCallMessage
from wakey_agent.tools import (
    build_argv,
    execute_tool_call,
    read_file,
    resolve_path,
    run_shell,
    truncate,
    write_file,
)

# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def ws(tmp_path: Path) -> Path:
    """A fresh workspace for every test, inside pytest's temporary directory.

    tmp_path itself plays the role of "everything outside the workspace".
    """
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "hello.c").write_text("int main(void) { return 0; }\n")
    return workspace


def make_call(name: str, args: object) -> ToolCallMessage:
    """Build a tool call like the model sends; args is serialised unless already a string."""
    arguments = args if isinstance(args, str) else json.dumps(args)
    return {"id": "test", "type": "function", "function": {"name": name, "arguments": arguments}}


def contains_sequence(haystack: list[str], needle: list[str]) -> bool:
    """True if needle appears in haystack as consecutive items."""
    n = len(needle)
    return any(haystack[i : i + n] == needle for i in range(len(haystack) - n + 1))


# ---------------------------------------------------------------------------
# resolve_path: what counts as inside the workspace
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("path", ["hello.c", "src/new.c", "./hello.c", "src/../hello.c"])
def test_resolve_path_accepts_paths_inside(ws: Path, path: str) -> None:
    assert resolve_path(path, str(ws)).is_relative_to(ws.resolve())


@pytest.mark.parametrize("path", ["../outside.txt", "/etc/passwd", "src/../../outside.txt"])
def test_resolve_path_rejects_paths_outside(ws: Path, path: str) -> None:
    with pytest.raises(ValueError):
        resolve_path(path, str(ws))


def test_resolve_path_rejects_sibling_with_same_prefix(ws: Path, tmp_path: Path) -> None:
    # "ws2" starts with "ws" as a string; a startswith() check would let it through.
    sibling = tmp_path / "ws2"
    sibling.mkdir()
    (sibling / "secret.txt").write_text("secret")
    with pytest.raises(ValueError):
        resolve_path("../ws2/secret.txt", str(ws))


def test_resolve_path_rejects_symlink_pointing_out(ws: Path) -> None:
    (ws / "root").symlink_to("/")
    with pytest.raises(ValueError):
        resolve_path("root/etc/passwd", str(ws))


def test_resolve_path_follows_symlink_inside(ws: Path) -> None:
    (ws / "alias.c").symlink_to("hello.c")
    assert resolve_path("alias.c", str(ws)) == (ws / "hello.c").resolve()


# ---------------------------------------------------------------------------
# read_file / write_file
# ---------------------------------------------------------------------------


def test_read_file_returns_contents(ws: Path) -> None:
    assert read_file("hello.c", str(ws)) == "int main(void) { return 0; }\n"


def test_read_file_missing_file_is_an_error_string(ws: Path) -> None:
    result = read_file("nope.c", str(ws))
    assert result.startswith("Error")


def test_read_file_outside_is_an_error_string(ws: Path, tmp_path: Path) -> None:
    (tmp_path / "outside.txt").write_text("secret")
    result = read_file("../outside.txt", str(ws))
    assert result.startswith("Error")
    assert "secret" not in result


def test_write_file_creates_file_and_parent_dirs(ws: Path) -> None:
    result = write_file("src/main.c", "int x;\n", str(ws))
    assert not result.startswith("Error")
    assert (ws / "src" / "main.c").read_text() == "int x;\n"


def test_write_file_overwrites(ws: Path) -> None:
    write_file("hello.c", "new\n", str(ws))
    assert (ws / "hello.c").read_text() == "new\n"


@pytest.mark.parametrize("path", ["../escape.txt", "/tmp/wakey-agent-test-escape.txt"])
def test_write_file_outside_writes_nothing(ws: Path, tmp_path: Path, path: str) -> None:
    result = write_file(path, "x", str(ws))
    assert result.startswith("Error")
    assert not (tmp_path / "escape.txt").exists()
    assert not Path("/tmp/wakey-agent-test-escape.txt").exists()


def test_write_file_through_symlink_out_writes_nothing(ws: Path, tmp_path: Path) -> None:
    (ws / "out").symlink_to(tmp_path)
    result = write_file("out/escape.txt", "x", str(ws))
    assert result.startswith("Error")
    assert not (tmp_path / "escape.txt").exists()


def test_file_tool_messages_do_not_leak_host_paths(ws: Path, tmp_path: Path) -> None:
    host = str(tmp_path)
    assert host not in write_file("a.c", "x", str(ws))
    assert host not in read_file("missing.c", str(ws))
    assert host not in read_file("../outside.txt", str(ws))


# ---------------------------------------------------------------------------
# execute_tool_call: every bad call becomes an error string, never an exception
# ---------------------------------------------------------------------------


def test_execute_valid_call(ws: Path) -> None:
    result = execute_tool_call(make_call("read_file", {"path": "hello.c"}), str(ws))
    assert result == "int main(void) { return 0; }\n"


@pytest.mark.parametrize(
    "call",
    [
        make_call("delete_everything", {"path": "hello.c"}),  # unknown tool
        make_call("read_file", "{path: hello.c"),  # invalid JSON
        make_call("read_file", ["hello.c"]),  # JSON, but not an object
        make_call("read_file", {}),  # missing argument
        make_call("read_file", {"file": "hello.c"}),  # wrong argument name
        make_call("read_file", {"path": "hello.c", "workspace": "/"}),  # tries to set workspace
        make_call("", {}),  # empty name (stream sent no name)
        make_call(
            "run_shell", {"command": "true", "approve": True}
        ),  # model tries to skip the prompt
    ],
)
def test_execute_bad_calls_return_errors(ws: Path, call: ToolCallMessage) -> None:
    result = execute_tool_call(call, str(ws))
    assert result.startswith("Error")


def test_execute_cannot_override_workspace(ws: Path, tmp_path: Path) -> None:
    (tmp_path / "outside.txt").write_text("secret")
    call = make_call("read_file", {"path": "outside.txt", "workspace": str(tmp_path)})
    assert "secret" not in execute_tool_call(call, str(ws))


def test_truncate_leaves_short_text_alone() -> None:
    assert truncate("short", max_chars=10) == "short"


def test_truncate_keeps_head_and_tail() -> None:
    result = truncate("abcdefghij", max_chars=4)
    assert result.startswith("ab") and result.endswith("ij")
    assert "6 characters omitted" in result


# ---------------------------------------------------------------------------
# build_argv: the sandbox configuration
# ---------------------------------------------------------------------------


def test_build_argv_wraps_command_in_bwrap(ws: Path) -> None:
    argv = build_argv("echo hi", str(ws))
    assert argv[0] == "bwrap"
    assert argv[-3:] == ["bash", "-c", "echo hi"]


def test_build_argv_mounts_workspace_writable_at_neutral_path(ws: Path) -> None:
    argv = build_argv("true", str(ws))
    assert contains_sequence(argv, ["--bind", str(ws), "/workspace"])
    assert contains_sequence(argv, ["--chdir", "/workspace"])


def test_build_argv_mounts_usr_read_only(ws: Path) -> None:
    assert contains_sequence(build_argv("true", str(ws)), ["--ro-bind", "/usr", "/usr"])


@pytest.mark.parametrize(
    "flag", ["--clearenv", "--unshare-all", "--new-session", "--die-with-parent"]
)
def test_build_argv_has_isolation_flag(ws: Path, flag: str) -> None:
    assert flag in build_argv("true", str(ws))


def test_build_argv_never_binds_home_or_root(ws: Path) -> None:
    argv = build_argv("true", str(ws))
    for bind in ("--bind", "--ro-bind"):
        for i, item in enumerate(argv[:-1]):
            if item == bind:
                assert argv[i + 1] not in ("/", str(Path.home()))


# ---------------------------------------------------------------------------
# run_shell inside the real sandbox (skipped if bubblewrap is not installed)
# ---------------------------------------------------------------------------

needs_bwrap = pytest.mark.skipif(shutil.which("bwrap") is None, reason="bubblewrap not installed")


@pytest.fixture
def approve(monkeypatch: pytest.MonkeyPatch) -> None:
    """Answer 'y' to run_shell's confirmation prompt."""
    monkeypatch.setattr("builtins.input", lambda _prompt="": "y")


def test_run_shell_declined_runs_nothing(ws: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("builtins.input", lambda _prompt="": "n")
    result = run_shell("touch should_not_exist", str(ws))
    assert "declined" in result.lower()
    assert not (ws / "should_not_exist").exists()


@needs_bwrap
def test_sandbox_can_write_in_workspace(ws: Path, approve: None) -> None:
    result = run_shell("echo hi > out.txt", str(ws))
    assert "exit code: 0" in result
    assert (ws / "out.txt").read_text() == "hi\n"


@needs_bwrap
def test_sandbox_reports_failing_commands(ws: Path, approve: None) -> None:
    result = run_shell("ls does-not-exist", str(ws))
    assert "exit code: 0" not in result
    assert "does-not-exist" in result  # stderr is passed back to the model


@needs_bwrap
def test_sandbox_hides_etc_and_home(ws: Path, approve: None) -> None:
    assert "exit code: 0" not in run_shell("cat /etc/passwd", str(ws))
    assert "exit code: 0" not in run_shell(f"ls {Path.home()}/.ssh", str(ws))


@needs_bwrap
def test_sandbox_usr_is_read_only(ws: Path, approve: None) -> None:
    assert "exit code: 0" not in run_shell("touch /usr/wakey-test", str(ws))


@needs_bwrap
def test_sandbox_home_is_workspace(ws: Path, approve: None) -> None:
    assert "/workspace" in run_shell("echo $HOME", str(ws))


@needs_bwrap
def test_sandbox_does_not_inherit_environment(
    ws: Path, approve: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("WAKEY_TEST_SECRET", "hunter2")
    assert "hunter2" not in run_shell("env", str(ws))


@needs_bwrap
def test_sandbox_has_no_network(ws: Path, approve: None) -> None:
    # bash's /dev/tcp opens a TCP connection without needing curl or ping.
    result = run_shell("echo > /dev/tcp/1.1.1.1/80", str(ws))
    assert "exit code: 0" not in result


@needs_bwrap
def test_sandbox_can_compile_c(ws: Path, approve: None) -> None:
    if shutil.which("gcc") is None:
        pytest.skip("gcc not installed")
    result = run_shell("gcc hello.c -o hello && ./hello", str(ws))
    assert "exit code: 0" in result
