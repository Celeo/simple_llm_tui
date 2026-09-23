import argparse
import asyncio
import json
import sys
from typing import Any

import httpx
from pydantic import BaseModel
from pydantic.fields import Field
from rich.console import Console
from rich.markdown import Markdown
from rich.prompt import Prompt

URL = "http://localhost:11434/v1/chat/completions"
MODEL = "qwen3:8b"


class ChatMessage(BaseModel):
    role: str
    content: str
    tool_call_id: str | None = Field(default=None)
    tool_calls: list[Any] | None = Field(default=None)


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


class ChatMessageWrapper(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


"""
{
  "id": "chatcmpl-29",
  "object": "chat.completion",
  "created": 1790137871,
  "model": "qwen3:8b",
  "system_fingerprint": "fp_ollama",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "",
        "reasoning": "Okay, the user is asking about the type of FOSS license used in the ./LICENSE file. First, I need to figure out how to determine the license type. The most straightforward way is to check the contents of the LICENSE file. Since the user mentioned the file is in the current directory, the path is ./LICENSE.\n\nI remember that there's a function called read_file that can read the contents of a file from the local filesystem. The function requires the path as an argument. So, I should use that function to read the LICENSE file. Once I have the content, I can analyze it to identify the license type. Common FOSS licenses include MIT, Apache, GPL, etc. Each has a distinctive wording and preamble. For example, the MIT license starts with \"MIT License\" and has a specific copyright notice. The Apache license has a more detailed section about permissions and conditions. The GPL license includes terms about free distribution and source code. So, after reading the file, I can look for these key phrases to determine which license it is. If the file contains multiple licenses, I might need to check each section. But the user is asking for the type used, so it's likely a single license. Therefore, the plan is to read the file and then analyze its content to identify the license type.\n",
        "tool_calls": [
          {
            "id": "call_58hoea1l",
            "index": 0,
            "type": "function",
            "function": {
              "name": "read_file",
              "arguments": "{\"path\":\"./LICENSE\"}"
            }
          }
        ]
      },
      "finish_reason": "tool_calls"
    }
  ],
  "usage": {
    "prompt_tokens": 169,
    "prompt_tokens_details": {
      "cached_tokens": 147
    },
    "completion_tokens": 290,
    "total_tokens": 459
  }
}
"""


class ChatResponse(BaseModel):
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


def call_tools(console: Console, response: ChatMessage) -> list[ChatMessage]:
    ret = []

    for entry in response.tool_calls or []:
        id = entry["id"]
        tool = entry["function"]["name"]
        args = json.loads(entry["function"]["arguments"])

        match tool:
            case "read_file":
                with open(args["path"]) as f:
                    content = f.read()
                ret.append(ChatMessage(role="tool", tool_call_id=id, content=content))
            case _:
                console.print(f"[red]Unknown tool '{tool}' called[/red]")
                sys.exit(1)

    return ret


async def main() -> None:
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
            while response.tool_calls:
                tools_output = call_tools(console, response)
                history.extend(tools_output)
                request = wrap_request(history)
                if args.debug:
                    console.print_json(request.model_dump_json(by_alias=True))

                response = await call_llm(client, console, args.debug, request)
                if not response:
                    sys.exit(1)
                response = extract_output(response)
                history.append(response)

            print_response(console, response)


if __name__ == "__main__":
    asyncio.run(main())
