import argparse
import json
import sys

import requests

from wakey_agent.client import (
    AssistantMessage,
    TextMessage,
    WakeyError,
    chat,
    get_headers,
    get_payload,
    load_config,
)

WEATHER_TOOL = {
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


QUESTION: TextMessage = {"role": "user", "content": "What is the weather in Tokyo?"}


def check_one(url: str, api_key: str, model: str, stream=False) -> requests.Response:
    return requests.post(
        f"{url}/v1/chat/completions",
        json=get_payload(model, [QUESTION], [WEATHER_TOOL], stream),
        headers=get_headers(api_key),
        timeout=(5, 120),
        stream=stream,
    )


def check_two(url: str, api_key: str, resp, model, stream=False):
    msg = resp["choices"][0]["message"]
    call = msg["tool_calls"][0]
    payload = get_payload(model, [QUESTION], [WEATHER_TOOL], stream)
    payload["messages"].append(msg)
    payload["messages"].append(
        {
            "role": "tool",
            "tool_call_id": call["id"],
            "content": "12°C, light rain",
        }
    )
    return requests.post(
        f"{url}/v1/chat/completions", json=payload, headers=get_headers(api_key), timeout=(5, 120)
    )


def check_three(url: str, api_key: str, model: str) -> tuple[AssistantMessage, str | None]:
    message, finish = chat(url, api_key, model, [QUESTION], [WEATHER_TOOL])
    return message, finish


def is_nonempty_str(x: object) -> bool:
    return isinstance(x, str) and bool(x.strip())


def verify_call(name: str | None, arguments: str) -> tuple[bool, str]:
    if name != "get_weather":
        return False, f"function name {name!r}; expected 'get_weather'"
    try:
        args = json.loads(arguments)
    except json.JSONDecodeError:
        return False, f"arguments are not valid JSON: {arguments!r}"
    if not isinstance(args, dict):
        return False, "arguments is not a dict"
    city = args.get("city")
    if city is None:
        return False, 'arguments do not contain "city"'
    if not is_nonempty_str(city):
        return False, f"'city' is not a non-empty string: {city!r}"

    return True, ""


type CheckResult = tuple[bool, str]


def verify_check_one(r: requests.Response) -> CheckResult:
    try:
        if r.status_code != 200:
            return False, f"HTTP {r.status_code}"
        choice = r.json()["choices"][0]
        if choice["finish_reason"] != "tool_calls":
            return False, f"finish reason: {choice['finish_reason']!r}; expected 'tool_calls'"
        msg = choice["message"]
        if msg.get("tool_calls") is None:
            return False, 'No "tool_calls" in message'
        tool_calls = msg["tool_calls"]
        if not tool_calls:
            return False, "No calls in 'tool_calls'"
        call = tool_calls[0]
        func = call["function"]
        ok, reason = verify_call(func["name"], func["arguments"])
        if not ok:
            return False, reason
    except Exception as e:
        return False, repr(e)

    return True, ""


def verify_check_two(r: requests.Response) -> CheckResult:
    try:
        if r.status_code != 200:
            return False, f"HTTP {r.status_code}"
        choice = r.json()["choices"][0]
        if choice["finish_reason"] != "stop":
            return False, f"finish reason: {choice['finish_reason']!r}; expected 'stop'"
        content = choice["message"].get("content")
        if content is None:
            return False, "message has no 'content'"
        if not is_nonempty_str(content):
            return False, f"'content' is not a non-empty string: {content!r}"
        if "12" not in content:
            return False, "Model did not use our fake result '12'"
    except Exception as e:
        return False, repr(e)

    return True, ""


def verify_check_three(message: AssistantMessage, finish_reason: str | None) -> CheckResult:
    try:
        if finish_reason != "tool_calls":
            return False, f"finish reason: {finish_reason!r}; expected 'tool_calls'"
        tool_calls = message.get("tool_calls", [])
        if len(tool_calls) != 1:
            return False, f"expected 1 call, got {len(tool_calls)}"
        call = tool_calls[0]
        return verify_call(call["function"]["name"], call["function"]["arguments"])
    except Exception as e:
        return False, repr(e)


def run_check_one(url: str, api_key: str, model: str) -> tuple[bool, str, requests.Response | None]:
    try:
        r = check_one(url, api_key, model)
    except requests.RequestException as e:
        return False, f"request failed: {e!r}", None
    ok, reason = verify_check_one(r)
    return ok, reason, r


def run_check_two(url: str, api_key: str, resp, model: str) -> tuple[bool, str]:
    try:
        r = check_two(url, api_key, resp, model)
    except requests.RequestException as e:
        return False, f"request failed: {e!r}"
    ok, reason = verify_check_two(r)
    return ok, reason


def run_check_three(url: str, api_key: str, model: str) -> tuple[bool, str]:
    try:
        calls, finish_reason = check_three(url, api_key, model)
    except (requests.RequestException, WakeyError) as e:
        return False, f"request failed: {e!r}"
    except (ValueError, KeyError, IndexError) as e:
        return False, f"malformed stream: {e!r}"
    ok, reason = verify_check_three(calls, finish_reason)
    return ok, reason


def print_summary(failures: dict[int, list[tuple[int, str]]], skipped_2: int, runs: int):
    ran_2 = runs - skipped_2
    print(f"check 1: {runs - len(failures[1])}/{runs}")
    print(f"check 2: {ran_2 - len(failures[2])}/{ran_2} ({skipped_2} skipped)")
    print(f"check 3: {runs - len(failures[3])}/{runs}")

    for check, list in failures.items():
        for run, reason in list:
            print(f"run {run}, check {check}: {reason}")


def main():
    parser = argparse.ArgumentParser(description="Check tool calling through Wakey")
    parser.add_argument(
        "--runs", type=int, default=1, help="how many times to run each check (default: 1)"
    )
    parser.add_argument("--model", default="gemma-4-12b", help="model name to request")
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="print every response, not just failures"
    )
    args = parser.parse_args()

    failures: dict[int, list[tuple[int, str]]] = {1: [], 2: [], 3: []}
    skipped_2 = 0

    try:
        url, api_key = load_config()

        for i in range(args.runs):
            ok, reason, r = run_check_one(url, api_key, args.model)
            if ok and r is not None:
                ok2, reason2 = run_check_two(url, api_key, r.json(), args.model)
                if not ok2:
                    failures[2].append((i, reason2))
            else:
                failures[1].append((i, reason))
                skipped_2 += 1

            ok3, reason3 = run_check_three(url, api_key, args.model)
            if not ok3:
                failures[3].append((i, reason3))

        print_summary(failures, skipped_2, args.runs)

    except (FileNotFoundError, ValueError, PermissionError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(2)
    except KeyError as e:
        print(f"Error: config is missing key {e}", file=sys.stderr)
        sys.exit(2)

    if any(failures.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
