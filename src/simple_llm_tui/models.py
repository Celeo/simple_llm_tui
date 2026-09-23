from typing import Any

from pydantic import BaseModel, ConfigDict
from pydantic.fields import Field


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
    reasoning_effort: str | None = Field(default=None)
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
