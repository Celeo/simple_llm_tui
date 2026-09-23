import argparse
import asyncio
import json
import sys
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict
from pydantic.fields import Field
from rich.console import Console
from rich.markdown import Markdown
from rich.prompt import Prompt

URL = "http://localhost:11434/v1/chat/completions"
MODEL = "qwen3:8b"
LOOP_TOOL_CALL_MAX = 5


class ApiModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ToolCallFunction(ApiModel):
    name: str
    arguments: str


class ToolCall(ApiModel):
    id: str
    index: int
    kind: str = Field(default="function", alias="type")
    function: ToolCallFunction


class ChatMessage(ApiModel):
    role: str
    content: str
    reasoning: str | None = Field(default=None)
    tool_call_id: str | None = Field(default=None)
    tool_calls: list[ToolCall] | None = Field(default=None)


class ToolDefinition(BaseModel):
    name: str
    description: str
    parameters: dict[str, Any]


class Tool(BaseModel):
    kind: str = Field(default="function", alias="type")
    function: ToolDefinition


class ChatRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    tools: list[Tool]


class ChatMessageWrapper(ApiModel):
    index: int
    message: ChatMessage
    finish_reason: str


class PromptTokensDetails(ApiModel):
    cached_tokens: int


class Usage(ApiModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    prompt_tokens_details: PromptTokensDetails | None = Field(default=None)


class ChatResponse(ApiModel):
    id: str
    object: str
    created: int
    model: str
    system_fingerprint: str | None = None
    choices: list[ChatMessageWrapper]
    usage: Usage


async def call_llm(
    client: httpx.AsyncClient, console: Console, debug: bool, request: ChatRequest
) -> ChatResponse | None:
    try:
        response = await client.post(
            URL, json=request.model_dump(by_alias=True, exclude_none=True)
        )
        if response.status_code != 200:
            console.print(
                f"[red]Something went wrong with the request ({response.status_code}): {response.text} [/red]"
            )
            return None
        data = response.json()
        if debug:
            console.print_json(json.dumps(data, indent=2))
        return ChatResponse.model_validate(data)
    except httpx.HTTPError:
        console.print("[red]Ollama could not be reached - is it running?[/red]")
        return None


def wrap_user(content: str) -> ChatMessage:
    return ChatMessage(role="user", content=content)


def wrap_request(messages: list[ChatMessage]) -> ChatRequest:
    return ChatRequest(
        model=MODEL,
        messages=messages,
        tools=[
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
            )
        ],
    )


def extract_output(response: ChatResponse) -> ChatMessage:
    return response.choices[0].message


def print_response(console: Console, response: ChatMessage) -> None:
    print()
    console.print(Markdown(response.content))
    print("\n")


def call_tools(response: ChatMessage) -> list[ChatMessage]:
    ret = []

    for entry in response.tool_calls or []:
        id = entry.id
        tool = entry.function.name
        args = json.loads(entry.function.arguments)

        match tool:
            case "read_file":
                try:
                    with open(args["path"]) as f:
                        content = f.read()
                except OSError as e:
                    content = f"Error in 'read_file' tool: {e}"
                ret.append(ChatMessage(role="tool", tool_call_id=id, content=content))
            case _:
                ret.append(
                    ChatMessage(
                        role="tool",
                        tool_call_id=id,
                        content=f"Unknown tool '{tool}'",
                    )
                )

    return ret


async def async_main() -> None:
    parser = argparse.ArgumentParser(description="Simple LLM TUI")
    parser.add_argument(
        "-d", "--debug", action="store_true", help="Enable debug logging"
    )
    args = parser.parse_args()

    console = Console()
    history: list[ChatMessage] = [
        ChatMessage(role="system", content="You are a simple assistant.")
    ]
    async with httpx.AsyncClient(timeout=60.0) as client:
        while True:
            tool_calls = 0
            message = Prompt.ask("[green]Input[/green]")
            if not message:
                console.print("\n[cyan]Bye! :wave:[/cyan]")
                break

            user_message = wrap_user(message)
            history.append(user_message)
            request = wrap_request(history)
            if args.debug:
                console.print_json(request.model_dump_json(by_alias=True))

            response = await call_llm(client, console, args.debug, request)
            if not response:
                sys.exit(1)

            response = extract_output(response)
            history.append(response)
            while response.tool_calls and tool_calls < LOOP_TOOL_CALL_MAX:
                tools_output = call_tools(response)
                history.extend(tools_output)
                tool_calls += 1
                request = wrap_request(history)
                if args.debug:
                    console.print_json(request.model_dump_json(by_alias=True))

                response = await call_llm(client, console, args.debug, request)
                if not response:
                    sys.exit(1)
                response = extract_output(response)
                history.append(response)

            print_response(console, response)


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
