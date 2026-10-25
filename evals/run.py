import argparse
import json
import tempfile
import time
from pathlib import Path
from typing import Any

from tasks import TASKS

from wakey_agent.client import load_config
from wakey_agent.loop import run_agent


def log_to_file(log_path: Path, entry: dict[str, Any]):
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Task runner")
    parser.add_argument(
        "--runs", type=int, default=1, help="how many times to run each check (default: 1)"
    )
    parser.add_argument(
        "--max-steps", type=int, default=20, help="Maximum amount of model calls (default: 20)"
    )
    parser.add_argument("--model", type=str, default="gemma-4-12b", help="model name to request")

    args = parser.parse_args()

    url, api_key = load_config()

    log_dir = Path(__file__).parent / "results"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{time.strftime('%Y-%m-%dT%H-%M-%S')}.jsonl"

    for t in TASKS:
        for i in range(args.runs):
            with tempfile.TemporaryDirectory() as tmp:
                ws = Path(tmp)
                for rel, content in t.files.items():
                    (ws / rel).parent.mkdir(parents=True, exist_ok=True)
                    (ws / rel).write_text(content)
                start = time.perf_counter()
                try:
                    result = run_agent(
                        url=url,
                        api_key=api_key,
                        max_steps=args.max_steps,
                        model=args.model,
                        task=t.prompt,
                        auto_approve=True,
                        workspace=tmp,
                    )
                except Exception as e:
                    log_to_file(
                        log_path, {"task": t.name, "run": i, "passed": False, "reason": repr(e)}
                    )
                    print(f"{t.name:<12} run {i + 1}  ERROR ({e!r})")
                    continue

                elapsed = time.perf_counter() - start

                try:
                    passed, reason = t.check(ws)
                except Exception as e:  # timeouts, bugs in a check, anything
                    passed, reason = False, f"check crashed: {e!r}"

                entry = {
                    "task": t.name,
                    "run": i,
                    "passed": passed,
                    "reason": reason,
                    "stop_reason": result.stop_reason,
                    "steps": result.steps,
                    "tokens": result.tokens,
                    "seconds": elapsed,
                }
                log_to_file(log_path, entry)
                status = "PASS" if passed else f"FAIL ({reason})"
                print(f"{t.name:<12} run {i + 1}  {status}  {result.steps} steps  {elapsed:.1f}s")


if __name__ == "__main__":
    main()
