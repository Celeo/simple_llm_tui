import asyncio

import httpx
from pydantic import BaseModel
from rich.console import Console
from rich.markdown import Markdown
from rich.prompt import Prompt

URL = "http://localhost:11434/v1/chat/completions"
MODEL = "qwen3:8b"


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str
    messages: list[ChatMessage]


class ChatMessageWrapper(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    system_fingerprint: str | None = None
    choices: list[ChatMessageWrapper]
    usage: Usage


async def call_llm(input: ChatRequest) -> ChatResponse:
    async with httpx.AsyncClient() as client:
        response = await client.post(URL, json=input.model_dump())
        if response.status_code != 200:
            raise ValueError(
                f"Error response code {response.status_code} from Ollama: {response.text}"
            )
        return ChatResponse.model_validate(response.json())


def wrap_user(content: str) -> ChatMessage:
    return ChatMessage(role="user", content=content)


def wrap_request(messages: list[ChatMessage]) -> ChatRequest:
    return ChatRequest(model=MODEL, messages=messages)


def extract_output(response: ChatResponse) -> ChatMessage:
    return response.choices[0].message


def print_response(console: Console, response: ChatMessage) -> None:
    print()
    console.print(Markdown(response.content))
    print("\n")


async def main() -> None:
    console = Console()
    history: list[ChatMessage] = []
    while True:
        message = Prompt.ask("[green]Input[/green]")
        if not message:
            console.print("\n[cyan]Bye! :wave:[/cyan]")
            break

        next = wrap_user(message)
        history.append(next)
        request = wrap_request(history)

        response = await call_llm(request)
        response = extract_output(response)
        print_response(console, response)

        history.append(response)


if __name__ == "__main__":
    asyncio.run(main())
