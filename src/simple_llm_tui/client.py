import argparse
import json
import sys

import httpx
from rich.console import Console
from rich.markdown import Markdown
from rich.prompt import Prompt

from .models import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
)
from .tools import TOOL_DEFINITIONS, list_directory, read_file

HEALTH_URL = "http://localhost:11434/api/tags"
REQUEST_URL = "http://localhost:11434/v1/chat/completions"
MODEL = "qwen3:8b"
SYSTEM_PROMPT = "You are a simple assistant. `/think` is a control token, not part of the user's question."
LOOP_TOOL_CALL_MAX = 5


async def call_llm(
    client: httpx.AsyncClient, console: Console, debug: bool, request: ChatRequest
) -> ChatResponse | None:
    try:
        response = await client.post(
            REQUEST_URL, json=request.model_dump(by_alias=True, exclude_none=True)
        )
        if response.status_code != 200:
            console.print(
                f"[red]Something went wrong with the request ({response.status_code}): {response.text} [/red]"
            )
            return None
        data = response.json()
        if debug:
            console.print("[cyan]>> Response[/cyan]")
            console.print_json(json.dumps(data, indent=2))
        return ChatResponse.model_validate(data)
    except httpx.HTTPError:
        console.print("[red]Ollama could not be reached - is it running?[/red]")
        return None


def wrap_user(content: str) -> ChatMessage:
    return ChatMessage(role="user", content=content)


def wrap_request(think: bool, messages: list[ChatMessage]) -> ChatRequest:
    return ChatRequest(
        model=MODEL,
        messages=messages,
        tools=TOOL_DEFINITIONS,
        reasoning_effort=None if think else "none",
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
                ret.append(read_file(id, args))
            case "list_directory":
                ret.append(list_directory(id, args))
            case _:
                ret.append(
                    ChatMessage(
                        role="tool",
                        tool_call_id=id,
                        content=f"Unknown tool '{tool}'",
                    )
                )

    return ret


async def check_ready(console: Console, client: httpx.AsyncClient) -> bool:
    try:
        response = await client.head(HEALTH_URL)
        response.raise_for_status()
        return True
    except httpx.HTTPError as e:
        console.print(
            f"[red]Error reaching localhost Ollama - is it running? {e}[/red]"
        )
        return False


def maybe_dump_history(do: bool, history: list[ChatMessage]) -> None:
    if do:
        with open("history.jsonl", "w") as f:
            f.writelines(
                [
                    f"{entry.model_dump_json(by_alias=True, exclude_none=True)}\n"
                    for entry in history
                ]
            )


async def async_main() -> None:
    parser = argparse.ArgumentParser(description="Simple LLM TUI")
    parser.add_argument(
        "-d", "--debug", action="store_true", help="Enable debug logging"
    )
    parser.add_argument(
        "-l", "--log-to-file", action="store_true", help="Log to a file"
    )
    parser.add_argument(
        "-t",
        "--think",
        action="store_true",
        help="Enable thinking (disabled by default)",
    )
    args = parser.parse_args()
    console = Console()
    history = [ChatMessage(role="system", content=SYSTEM_PROMPT)]

    async with httpx.AsyncClient(timeout=60.0) as client:
        if not await check_ready(console, client):
            sys.exit(1)

        while True:
            tool_calls = 0
            message = Prompt.ask("[green]Input[/green]")
            if not message:
                console.print("\n[cyan]Bye! :wave:[/cyan]")
                maybe_dump_history(args.log_to_file, history)
                break

            user_message = wrap_user(message)
            history.append(user_message)
            request = wrap_request(args.think, history)
            if args.debug:
                console.print("[cyan]>> Request[/cyan]")
                console.print_json(
                    request.model_dump_json(by_alias=True, exclude_none=True)
                )

            response = await call_llm(client, console, args.debug, request)
            if not response:
                maybe_dump_history(args.log_to_file, history)
                sys.exit(1)

            response = extract_output(response)
            history.append(response)
            while response.tool_calls and tool_calls < LOOP_TOOL_CALL_MAX:
                tools_output = call_tools(response)
                history.extend(tools_output)
                tool_calls += 1
                request = wrap_request(args.think, history)
                if args.debug:
                    console.print("[cyan]>> Tool response[/cyan]")
                    console.print_json(
                        request.model_dump_json(by_alias=True, exclude_none=True)
                    )

                response = await call_llm(client, console, args.debug, request)
                if not response:
                    maybe_dump_history(args.log_to_file, history)
                    sys.exit(1)
                response = extract_output(response)
                history.append(response)

            print_response(console, response)
