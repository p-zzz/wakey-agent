from wakey_agent.client import Message

CONTEXT_SIZE = 32768  # must match llama-server's -c setting on the backend
SHRINK_THRESHOLD = 0.75
KEEP_RECENT = 4
SHRINK_PLACEHOLDER = "[output removed to save context]"
SHRINK_TARGET = 0.5


def shrink_context(messages: list[Message], used: int, limit: int) -> int:
    tool_indices = [i for i, m in enumerate(messages) if m["role"] == "tool"]
    for i in tool_indices[
        : len(tool_indices) - KEEP_RECENT
    ]:  # oldest first, skipping the newest KEEP_RECENT
        if messages[i]["content"] == SHRINK_PLACEHOLDER:
            continue
        used -= len(messages[i]["content"]) // 4
        messages[i]["content"] = SHRINK_PLACEHOLDER
        if used <= SHRINK_TARGET * limit:
            break
    return used
