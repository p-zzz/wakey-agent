import subprocess
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from wakey_agent.tools import build_argv


@dataclass
class Task:
    name: str
    prompt: str
    check: Callable[[Path], tuple[bool, str]]
    files: dict[str, str] = field(default_factory=dict)


def check_hello(ws: Path) -> tuple[bool, str]:
    f = ws / "hello.txt"
    if not f.exists():
        return False, "hello.txt was not created"
    lines = f.read_text().splitlines()
    if lines != ["Hello World"] * 5:
        return False, f"unexpected content: {lines!r}"
    return True, ""


HELLO = Task(
    name="hello",
    prompt="Create a file named hello.txt containing the line 'Hello World' five times.",
    check=check_hello,
)


def sandboxed(ws: Path, command: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        build_argv(command, str(ws)),
        capture_output=True,
        text=True,
        timeout=30,
        stdin=subprocess.DEVNULL,
    )


def check_fix_compile(ws: Path) -> tuple[bool, str]:
    build = sandboxed(ws, "gcc main.c -o check_bin")
    if build.returncode != 0:
        return False, f"does not compile: {build.stderr.strip()}"
    run = sandboxed(ws, "./check_bin")
    if run.stdout != "hi\n":
        return False, f"wrong output: {run.stdout!r}"
    return True, ""


FIX_COMPILE = Task(
    name="fix_compile",
    prompt="fix the code in main.c so it compiles and produces the intended output. "
    "The program should print 'hi', followed by a new line",
    check=check_fix_compile,
    files={
        "main.c": '#include <stdio.h>\n\nint main(void) {\n    printf("hi\\n")\n    return 0;\n}\n'
    },
)

# fmt: off
EXPECTED_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41,
    43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
# fmt: on


def check_primes(ws: Path) -> tuple[bool, str]:
    build = sandboxed(ws, "gcc primes.c -o check_bin")
    if build.returncode != 0:
        return False, f"does not compile: {build.stderr.strip()}"
    run = sandboxed(ws, "./check_bin")
    try:
        got = [int(x) for x in run.stdout.split()]
    except ValueError:
        return False, f"output is not just numbers: {run.stdout!r}"
    if got != EXPECTED_PRIMES:
        return False, f"wrong primes: {got}"
    return True, ""


PRIMES = Task(
    name="primes",
    prompt="Write a C program in primes.c that prints all prime numbers from 1 to 100, "
    "separated by spaces or newlines, and nothing else. Compile and run it to check.",
    check=check_primes,
)

TASKS = [HELLO, FIX_COMPILE, PRIMES]
