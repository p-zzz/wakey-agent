# wakey-agent

A small coding agent for Gemma 4 12B, served by [Wakey](https://github.com/p-zzz/Wakey). It reads and writes files and runs shell commands in a sandboxed workspace. You use it from `wakeyctl agent`.

```
$ cd ~/throwaway && wakeyctl agent
> write a C program that prints the first 10 primes, then run it
→ write_file({"path": "primes.c", "content": "#include <stdio.h>\n..."})
→ run_shell({"command": "gcc primes.c -o primes && ./primes"})
  run? [y/N] y
exit code: 0
stdout:
2 3 5 7 11 13 17 19 23 29
I wrote primes.c, compiled it and ran it. It prints the first 10 primes.
```

## How it works

```
wakeyctl agent ──JSON lines──▶ wakey-agent --serve ──HTTPS──▶ Wakey ──▶ llama-server (Gemma 4 12B)
                                     │
                                     └─ runs tools in the workspace (bwrap)
```

- The model only produces text. When it asks for a tool, the loop runs the tool, appends the result and calls the model again. A turn ends when the model answers without a tool call, or after `--max-steps` calls (default 20).
- Tools: `read_file` (with `offset`/`limit` for line ranges), `write_file`, `run_shell`.
- A bad tool call (unknown tool, invalid JSON, wrong arguments, path outside the workspace) is sent back to the model as an `Error: ...` result, so it can correct itself. Nothing is raised.
- Requests use the OpenAI chat format with streaming (SSE). llama-server needs `--jinja` for tool calling.
- `wakeyctl` starts `wakey-agent --serve` as a child process and exchanges JSON lines with it over stdin/stdout. See [docs/protocol.md](docs/protocol.md).

## Security

- **Sandbox.** `run_shell` runs in [bubblewrap](https://github.com/containers/bubblewrap): `/usr` is read-only, the workspace is the only writable mount (at `/workspace`), and there is no network, no inherited environment, no home directory and no `/etc`. The sandbox gets its own session and dies with its parent.
- **Paths.** File tools resolve paths (including symlinks) and refuse anything outside the workspace. Error messages don't contain host paths.
- **Workspace.** The current directory. `/`, `/home` and your home directory are refused.
- **Approvals.** Every shell command needs approval in `wakeyctl`. The backend denies when an approval is missing or malformed. Auto-approve (`--yes`, `/yes`) is decided in the frontend, never by the model.
- **Keys.** The agent only has the client key from `~/.config/wakey/client.json`. The admin key stays with `wakeyctl`.
- **Network.** Wakey listens only on the WireGuard interface. Don't expose it to the internet.
- **Logs.** Every tool call goes to `~/.local/state/wakey-agent/*.jsonl` (directory 700, files 600). Logs older than 7 days are deleted.

## Context

- Tool output is capped at 4000 characters (keeps the start and the end).
- Usage is tracked from the server's token counts. Above 75% of the context, the oldest tool results are replaced with a placeholder until usage is under 50%. The last 4 are kept. The messages themselves stay, so every `tool_call_id` still matches its call.
- The context size comes from Wakey (`n_ctx` in `/admin/status`) and is passed with `--context-size`.

The PC runs llama-server with the full 256k context, quantized KV cache (`-c 262144 -fa on -ctk q8_0 -ctv q8_0`, about 10.8 GiB VRAM). Gemma's sliding-window attention keeps the cache at about 350 MiB per 32k tokens. Speed drops as the context fills:

| Context depth | Prompt (t/s) | Generation (t/s) |
|---|---|---|
| 0 | 3178 | 79.8 |
| 32k | 1945 | 68.9 |
| 128k | 907 | 51.3 |
| 256k | 534 | 38.1 |

`llama-bench`, pp512 / tg128.

## Evals

`evals/` runs tasks in a fresh temporary workspace and checks the result in the sandbox (compile, run, compare output). Results go to `evals/results/`.

| Task | What it does | gemma-4-12b |
|---|---|---|
| `hello` | write a file with fixed content | pass |
| `fix_compile` | fix a C file that doesn't compile | pass |
| `primes` | write, compile and run a C program | pass |
| `log` | find one fault code in a large log | 5/5 |

```
uv run evals/run.py              # all tasks once
uv run evals/run.py --task log --runs 5
```

## Setup

Needs Python 3.12+, [uv](https://docs.astral.sh/uv/), bubblewrap, and a running Wakey with `~/.config/wakey/client.json`.

```
git clone https://github.com/p-zzz/wakey-agent
cd wakey-agent
uv tool install --editable .
```

Then, from the directory you want as the workspace:

```
wakeyctl agent          # asks before each shell command
wakeyctl agent --yes    # auto-approve; toggle with /yes during the chat
```

`wakey-agent --serve` is the backend `wakeyctl` talks to. You don't normally run it yourself.

## Tests

```
uv run pytest
```

Covers path confinement, tool-call errors, the bwrap arguments, the real sandbox (skipped without bwrap), `read_file` ranges, the output cap and context shrinking.

## Roadmap

- Summarising old turns instead of dropping tool results
- More eval tasks, and comparing models
- A spinner in `wakeyctl` while the model is working
