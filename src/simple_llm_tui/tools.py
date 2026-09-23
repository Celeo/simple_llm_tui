import os

from .models import ChatMessage


def read_file(tool_call_id: str, args: dict[str, str]) -> ChatMessage:
    try:
        with open(args["path"]) as f:
            content = f.read()
    except OSError as e:
        content = f"Error in 'read_file' tool: {e}"
    return ChatMessage(role="tool", tool_call_id=tool_call_id, content=content)


def list_directory(tool_call_id: str, args: dict[str, str]) -> ChatMessage:
    try:
        content = ""
        with os.scandir(args["path"]) as it:
            for ref in it:
                content += f"{'file' if ref.is_file() else 'dir'} {ref.name}\n"
    except OSError as e:
        content = f"error in 'list_directory' tool: {e}"
    return ChatMessage(role="tool", tool_call_id=tool_call_id, content=content)
