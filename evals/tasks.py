import random
import subprocess
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
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


def power_eq(wind: float) -> float:
    p = 0.5 * 1.225 * 7854 * wind**3 * 0.6
    return min(p, 3.0e6) / 1e6


def make_log() -> str:
    rng = random.Random(42)
    start = datetime(2026, 10, 1)
    step = timedelta(seconds=10)
    lines: list[str] = []
    for i in range(3000):
        ts = start + i * step
        stamp = ts.isoformat()

        if i == 1734:
            lines.append(f"{stamp} ERROR T{(i % 15) + 1:02d} pitch system fault code=PF-4471")
            continue

        wind = rng.uniform(3, 15)
        lines.append(
            f"{stamp} INFO T{(i % 15) + 1:02d} wind={wind:.1f}m/s power={power_eq(wind):.2f}"
        )

    return "\n".join(lines) + "\n"


def check_log_task(ws: Path) -> tuple[bool, str]:
    f = ws / "answer.txt"
    if not f.exists():
        return False, "answer.txt does not exist"
    line = f.read_text().strip()
    if line != "PF-4471":
        return False, f"unexpected content: {line!r}"
    return True, ""


LOG = Task(
    name="log",
    prompt="The file server.log contains exactly one line with ERROR. "
    "Write only its fault code (for example XX-0000) to answer.txt.",
    check=check_log_task,
    files={"server.log": make_log()},
)


TASKS = [HELLO, FIX_COMPILE, PRIMES, LOG]
