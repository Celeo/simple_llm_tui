# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project purpose

An extremely simple LLM harness/TUI, built for **learning purposes** (not intended for practical use — see README for alternatives like crush, aider, pi). The entire implementation currently lives in a single file: `src/simple_llm_tui/__init__.py`.

When the user asks a question, they want to focus on **learning**. Unless explicitly asked, do not write or even suggest code changes. Instead, base your answers on the OpenAI chat completions specification, general LLM & harness knowledge, etc. The user may ask for code review after completing a task - read the code at this time and suggest improvements, but wait for futher instruction before writing any files.

## Commands

This project uses `uv` for dependency management and packaging.

- Install dependencies: `uv sync`
- Run the app: `uv run simple-llm-tui` (or `uv run simple-llm-tui --debug` / `-d` to print raw request/response JSON)
- Lint: `uv run ruff check .`
- Format: `uv run ruff format .`

There is no test suite in this repository.

## Architecture

The app is a single-file async CLI chat client that talks to a local OpenAI-compatible chat completions endpoint (default: Ollama at `http://localhost:11434/v1/chat/completions`, model `qwen3:8b`, both hardcoded as `URL`/`MODEL` constants).

Flow in `main()`:
1. Prompt the user for input via `rich.prompt.Prompt` in a REPL loop; an empty input exits.
2. Append the user message to a running `history: list[ChatMessage]` (seeded with a system message).
3. Build a `ChatRequest` (`wrap_request`) and POST it via `call_llm`, which returns a validated `ChatResponse` or `None` on HTTP/connection failure (caller then exits with `sys.exit(1)`).
4. If the model's response includes `tool_calls`, dispatch them via `call_tools` and loop — appending tool results as `role="tool"` messages and re-calling the LLM — up to `LOOP_TOOL_CALL_MAX` (5) iterations per user turn.
5. Once there are no more tool calls, render the final response as Markdown via `print_response`.

Key design points:
- All API request/response shapes are modeled as Pydantic models (`ApiModel` subclasses use `extra="forbid"` to catch schema drift from the API). `ChatRequest`/`ChatMessageWrapper`/`ToolDefinition`/`Tool` are not `ApiModel` subclasses since they're either outbound-only or intentionally lenient.
- Tool calling: currently one tool is defined (`read_file`, in `wrap_request`), dispatched in `call_tools` via a `match` statement on tool name. Adding a new tool means: add its `ToolDefinition` in `wrap_request`, and add a matching `case` in `call_tools` that returns a `ChatMessage(role="tool", tool_call_id=..., content=...)`.
