# Protocol

How `wakeyctl agent` (frontend) talks to `wakey-agent --serve` (backend).

## Transport

- One JSON object per line, UTF-8.
- Frontend → backend: the backend's **stdin**.
- Backend → frontend: the backend's **stdout**. Nothing else is ever written there.
- **stderr** is free-form diagnostics, not part of the protocol.

## Lifecycle

1. The frontend starts the backend; it inherits the frontend's current directory, which becomes the workspace.
2. The backend sends `ready`.
3. Each `user` message starts a **turn**. A turn ends with exactly one `done`.
4. The backend exits when its stdin is closed.

A typical turn:

```
→ user
← step, text..., text_end, tool_call, approve
→ approval
← tool_result, usage
← step, text..., text_end, usage, done
```

## Frontend → backend

**`user`**: a message from the user. Starts a turn.

`{"type": "user", "content": "activate Windows, no mistakes this time"}`

**`approval`**: the answer to an `approve` request. `id` must match the request's `id`; `ok` is `true` or `false`. Anything other than `true` counts as a refusal.

`{"type": "approval", "id": 3, "ok": true}`

## Backend → frontend

**`ready`**: the backend has started and is waiting for the first `user` message.

`{"type": "ready", "model": "gemma-4-12b", "workspace": "/home/user/throwaway"}`

**`step`**: a new model call is about to start. `used` is how full the context is going into this step (0 on the first step).

`{"type": "step", "step": 2, "max": 20, "used": 996, "limit": 32768}`

**`text`**: a piece of the model's reply, streamed as it is generated. Append `delta` to what has been shown so far.

`{"type": "text", "delta": "Let me check the"}`

**`text_end`**: the model's text for this step is complete (sent only if there was any text).

`{"type": "text_end"}`

**`tool_call`**: the model is calling a tool. For display only: the backend runs it, the frontend just shows it. `args` is the raw JSON string the model sent.

`{"type": "tool_call", "name": "run_shell", "args": "{\"command\": \"./activate_windows\"}"}`

**`approve`**: the backend wants to run a shell command and needs permission. The frontend must answer with an `approval` carrying the same `id`. The backend waits until it gets one.

`{"type": "approve", "id": 3, "command": "./activate_windows"}`

**`tool_result`**: the full result of the last tool call, as the model sees it (capped at about 4000 characters). For shell commands it includes the exit code, stdout and stderr; errors start with `Error:`.

`{"type": "tool_result", "output": "exit code: 0\nstdout:\nEnjoy your new OS\n\nstderr:\n"}`

**`usage`**: context fill after the model's reply, in tokens.

`{"type": "usage", "used": 1467, "limit": 32768}`

**`warning`**: something went wrong but the turn continues, e.g. invalid input from the frontend or a failure to write the log.

`{"type": "warning", "message": "ignored message while waiting for approval 3: {\"type\": \"user\", ...}"}`

**`error`**: something went wrong that ends the turn, e.g. Wakey is unreachable or the backend PC is off. Always followed by `done`.

`{"type": "error", "message": "HTTP 503: {\"error\":\"backend 'pc' is not running; wake it first\"}"}`

**`done`**: the turn is over. `reason` is `done` (the model answered), `max_steps` (it ran out of steps) or `error`.

`{"type": "done", "reason": "done", "steps": 4, "tokens": 1467}`

## Rules

- Frontends ignore event types they don't know, so new events can be added without breaking them.
- Every `approval` echoes the `id` of its `approve`; approvals with a different `id` are ignored.
- When in doubt, the backend denies: a missing or malformed approval, or a closed stdin, means the command does not run.
- Auto-approval is a frontend feature: the backend always asks, and the frontend may answer on the user's behalf.
