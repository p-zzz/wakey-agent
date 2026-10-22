"""Tests for read_file's offset/limit ranges and the tool output cap.

Run with:  uv run pytest
"""

import json
from pathlib import Path

import pytest

from wakey_agent.tools import execute_tool_call, read_file, truncate

LINES = [f"line{i}\n" for i in range(1, 11)]  # line1 .. line10


@pytest.fixture
def ws(tmp_path: Path) -> Path:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "ten.txt").write_text("".join(LINES))
    (workspace / "empty.txt").write_text("")
    return workspace


# ---------------------------------------------------------------------------
# Ranges
# ---------------------------------------------------------------------------


def test_no_range_returns_whole_file_without_note(ws: Path) -> None:
    assert read_file("ten.txt", str(ws)) == "".join(LINES)


def test_exact_range(ws: Path) -> None:
    result = read_file("ten.txt", str(ws), offset=3, limit=2)
    assert result.startswith("line3\nline4\n")
    assert "line5" not in result
    assert "lines 3-4 of 10" in result


def test_offset_only_reads_to_end(ws: Path) -> None:
    result = read_file("ten.txt", str(ws), offset=9)
    assert result.startswith("line9\nline10\n")
    assert "lines 9-10 of 10" in result


def test_limit_past_end_stops_at_last_line(ws: Path) -> None:
    result = read_file("ten.txt", str(ws), offset=8, limit=50)
    assert result.startswith("line8\nline9\nline10\n")
    assert "lines 8-10 of 10" in result


def test_range_covering_whole_file_has_no_note(ws: Path) -> None:
    assert read_file("ten.txt", str(ws), offset=1, limit=10) == "".join(LINES)
    assert read_file("ten.txt", str(ws), offset=1, limit=500) == "".join(LINES)


def test_first_line_only(ws: Path) -> None:
    result = read_file("ten.txt", str(ws), offset=1, limit=1)
    assert result.startswith("line1\n")
    assert "line2" not in result
    assert "lines 1-1 of 10" in result


@pytest.mark.parametrize(
    ("offset", "limit"),
    [(0, None), (-1, None), (1, 0), (1, -5), (11, None), (11, 3)],
)
def test_invalid_ranges_are_errors(ws: Path, offset: int, limit: int | None) -> None:
    assert read_file("ten.txt", str(ws), offset=offset, limit=limit).startswith("Error")


def test_empty_file(ws: Path) -> None:
    assert read_file("empty.txt", str(ws)) == ""


def test_range_through_execute_tool_call(ws: Path) -> None:
    # The model sends offset/limit as JSON integers in the arguments string.
    call = {
        "id": "t",
        "type": "function",
        "function": {
            "name": "read_file",
            "arguments": json.dumps({"path": "ten.txt", "offset": 2, "limit": 3}),
        },
    }
    result = execute_tool_call(call, str(ws))  # type: ignore[arg-type]
    assert result.startswith("line2\nline3\nline4\n")
    assert "lines 2-4 of 10" in result


# ---------------------------------------------------------------------------
# Output cap
# ---------------------------------------------------------------------------


def test_truncate_leaves_short_text_alone() -> None:
    assert truncate("short", max_chars=10) == "short"


def test_truncate_keeps_head_and_tail() -> None:
    result = truncate("abcdefghij", max_chars=4)
    assert result.startswith("ab")
    assert result.endswith("ij")
    assert "6 characters omitted" in result


def test_truncate_odd_cap_counts_correctly() -> None:
    # Cap 5 keeps 2 + 2 characters, so 10 - 4 = 6 are omitted.
    assert "6 characters omitted" in truncate("abcdefghij", max_chars=5)


def test_long_tool_result_is_capped(ws: Path) -> None:
    (ws / "big.txt").write_text("x" * 20_000)
    call = {
        "id": "t",
        "type": "function",
        "function": {"name": "read_file", "arguments": json.dumps({"path": "big.txt"})},
    }
    result = execute_tool_call(call, str(ws))  # type: ignore[arg-type]
    assert len(result) < 5_000
    assert "characters omitted" in result
