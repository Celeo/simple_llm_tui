import json
import os

from .models import ChatMessage, Tool, ToolDefinition

FILE_READ_SIZE_CAP = 2_000
TOOL_DEFINITIONS = [
    Tool(
        function=ToolDefinition(
            name="read_file",
            description="Read the contents of a file from the local filesystem.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute or relative path to the file to read.",
                    }
                },
                "required": ["path"],
            },
        )
    ),
    Tool(
        function=ToolDefinition(
            name="list_directory",
            description="List files in a directory. Does not recurse down directories.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute or relative path to the directory to list files in. Use '.' for the current directory.",
                    }
                },
                "required": ["path"],
            },
        )
    ),
]


def read_file(tool_call_id: str, args: dict[str, str]) -> ChatMessage:
    try:
        with open(args["path"]) as f:
            content = f.read()
            if len(content) > FILE_READ_SIZE_CAP:
                content = content[:FILE_READ_SIZE_CAP] + "\n...LARGE FILE ELIDED..."
    except KeyError:
        content = "'path' argument is required"
    except OSError as e:
        content = f"Error in 'read_file' tool: {e}"
    return ChatMessage(role="tool", tool_call_id=tool_call_id, content=content)


def list_directory(tool_call_id: str, args: dict[str, str]) -> ChatMessage:
    try:
        content = []
        with os.scandir(args["path"]) as it:
            for ref in it:
                content.append(
                    {"name": ref.name, "type": "file" if ref.is_file() else "dir"}
                )
    except KeyError:
        content = "'path' argument is required"
    except OSError as e:
        content = f"error in 'list_directory' tool: {e}"
    return ChatMessage(
        role="tool", tool_call_id=tool_call_id, content=json.dumps(content)
    )
