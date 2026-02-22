# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

Tanrenai (鍛錬AI) is a local-first LLM system: a Go CLI/server that manages llama-server subprocesses for inference, with agentic tool use, context windowing, and memory/RAG.

## Build & Test Commands

All commands run from `server/`:

```bash
# Build
go build -o tanrenai .

# Run all tests
go test ./...

# Run a single package's tests
go test ./internal/memory/... -v

# Run a single test by name
go test ./internal/e2e/... -v -run TestE2EMemoryWithToolCalls

# Run the server (must have llama-server binary in ~/.local/share/tanrenai/bin/)
go run main.go serve

# Run interactive agent with memory
go run main.go run <model-name> --agent --memory --ctx-size 16384
```

## Architecture

### Inference Path (subprocess, not CGo)

The system spawns `llama-server` as a child process (`internal/runner/process.go`) and communicates via HTTP. A separate subprocess handles embedding for memory/RAG. This means `{DataDir}/bin/llama-server` must exist.

### Two Entry Points to the Agent Loop

1. **CLI REPL** (`cmd/run.go` → `agentLoop()`): User types in terminal, streaming responses print directly. Memory search/inject and storage happen here, wrapping `agent.RunStreaming()`.
2. **HTTP API** (`internal/server/handlers/agent.go`): Server-side agent via `POST /v1/agent/completions`, uses `agent.Run()` (non-streaming).

The agent loop itself (`internal/agent/agent.go`) is decoupled from both — it takes a `StreamingCompletionFunc` or `CompletionFunc` and a `tools.Registry`, so it works identically in CLI and server contexts.

### Context Management (`internal/chatctx/`)

`Manager` maintains a token-budgeted message window:
- **Pinned** (never evicted): system prompt, context files, injected memories
- **Windowed**: conversation history, oldest evicted first
- **Budget**: `CtxSize - System - Memory - ResponseBudget = available for history`

Token estimation defaults to 3.5 chars/token, optionally calibrated against the server's `/tokenize` endpoint.

### Memory/RAG (`internal/memory/`)

ChromemStore wraps chromem-go for in-process vector storage with hybrid search (70% semantic + 30% keyword). `EmbedFunc` is the abstraction over embedding providers — production uses a second llama-server subprocess; tests use FNV-hash mock vectors.

The memory flow in `agentLoop`: search → inject as system messages → run agent → store turn.

### Tools (`internal/tools/`)

Registry pattern. Built-in: `file_read`, `file_write`, `list_dir`, `shell_exec`. Tools return `ToolResult{Output, IsError}` — tool-level errors go to the LLM; Go errors mean infrastructure failure.

### API Types (`pkg/api/`)

OpenAI-compatible request/response schemas. Streaming uses SSE with `ChatCompletionChunk` and delta accumulation for tool call arguments.

## Key Conventions

- `memory.Store` is an interface; `ChromemStore` is the implementation. `NewChromemStoreInMemory()` exists for tests.
- The agent's stuck-detection tracks repeated identical failing tool calls and force-stops after 3 consecutive repeats.
- REPL slash commands (`/memory`, `/context`, `/tokens`, `/clear`) are handled in `cmd/run.go` `handleREPLCommand()` before the message reaches the agent.
- Data directories: `~/.local/share/tanrenai/{models,bin,memory}` (override with `TANRENAI_DATA_DIR`).

## Module Path

```
github.com/thatcatdev/tanrenai/server
```

Go module is under `server/`, not the repo root.
