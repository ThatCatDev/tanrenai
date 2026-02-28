# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

Tanrenai (鍛錬AI) is a three-tier LLM system: a GPU inference server, an orchestration backend, and a CLI client with agentic tool use, context windowing, and memory/RAG.

## Architecture Overview

```
Client (client/)  →  Backend (server/)  →  GPU Server (gpu/)
CLI REPL + tools     memory, proxy,        llama-server,
agent loop            vast.ai mgmt         models, training
```

### Tier 1: GPU Server (`gpu/`)
Pure inference + training. Manages llama-server subprocesses. Exposes:
- `POST /v1/chat/completions` — LLM inference (streaming + non-streaming)
- `POST /v1/embeddings` — embedding generation
- `POST /tokenize` — token counting
- `POST /api/load`, `GET /v1/models`, `POST /api/pull` — model management
- `POST /v1/finetune/*` — fine-tuning endpoints

### Tier 2: Backend (`server/`)
Orchestration layer. Owns memory/RAG, manages vast.ai, proxies to GPU:
- Proxies completions, tokenize, models to GPU server
- `POST /v1/memory/search`, `POST /v1/memory/store`, `GET /v1/memory/list`, `DELETE /v1/memory/{id}`, `DELETE /v1/memory`, `GET /v1/memory/count`
- `GET /api/instance/status`, `POST /api/instance/start`, `POST /api/instance/stop`

### Tier 3: Client (`client/`)
Thin REPL + local tools. Agent loop runs here (tools execute on user's filesystem):
- Calls backend for completions, memory, models
- Tools: `file_read`, `file_write`, `patch_file`, `list_dir`, `find_files`, `grep_search`, `git_info`, `shell_exec`, `web_search`

## Build & Test Commands

```bash
# GPU server
cd gpu/ && go build -o tanrenai-gpu . && go test ./...

# Backend server
cd server/ && go build -o tanrenai-server . && go test ./...

# Client
cd client/ && go build -o tanrenai . && go test ./...
```

## Running

```bash
# 1. Start GPU server (locally or on vast.ai)
cd gpu/ && ./tanrenai-gpu serve --port 11435

# 2. Start backend pointing at GPU server
cd server/ && ./tanrenai-server serve --gpu-url http://localhost:11435 --memory --port 8080

# 3. Use client
cd client/ && ./tanrenai run <model-name> --agent --memory --server-url http://localhost:8080
```

## Module Paths

```
github.com/ThatCatDev/tanrenai/gpu      # GPU server
github.com/ThatCatDev/tanrenai/server   # Backend
github.com/ThatCatDev/tanrenai/client   # Client CLI
```

Three independent Go modules. JSON over HTTP is the contract between them.

## Key Packages

### GPU (`gpu/`)
- `internal/runner/` — llama-server subprocess management (process.go, stream.go, client.go)
- `internal/models/` — model store, download, manifest
- `internal/training/` — fine-tuning pipeline (manager, sidecar)
- `internal/server/handlers/` — HTTP handlers (chat, models, embeddings, tokenize, finetune)

### Backend (`server/`)
- `internal/gpuclient/` — typed HTTP client to GPU server
- `internal/memory/` — chromem-go vector store with hybrid search, remote embedding via GPU
- `internal/vastai/` — vast.ai instance management (client, manager)
- `internal/server/handlers/` — HTTP handlers (proxy, memory, instance, health)

### Client (`client/`)
- `internal/apiclient/` — typed HTTP client to backend (stream.go, client.go)
- `internal/agent/` — agent loop with tool calling and stuck detection
- `internal/chatctx/` — token-budgeted context windowing
- `internal/tools/` — tool registry and implementations

## Key Conventions

- `memory.Store` is an interface; `ChromemStore` is the implementation. `NewChromemStoreInMemory()` exists for tests.
- Backend's `memory.NewRemoteEmbedFunc(gpuClient)` calls GPU's `/v1/embeddings` instead of spawning a local subprocess.
- The agent's stuck-detection tracks repeated identical failing tool calls and force-stops after 3 consecutive repeats.
- REPL slash commands (`/memory`, `/context`, `/tokens`, `/clear`) are handled in `client/cmd/run.go`.
- Data directories: `~/.local/share/tanrenai/{models,bin,memory}` (override with `TANRENAI_DATA_DIR`).
- `pkg/api/types.go` is duplicated across all three modules (OpenAI-compatible schemas).
