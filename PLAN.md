# Tanrenai (鍛錬AI) — Project Plan

> "Tempering/discipline + AI" — a self-evolving local LLM system that gets better over time through continuous refinement.

## Vision

Build a system like Ollama that can:
1. Serve and run LLMs locally
2. Update/fine-tune the model based on conversations
3. Run an agentic loop with tool calling
4. Load context and execute commands (like Claude Code)

## Architecture

```
┌─────────────────────────────────────────────┐
│                 Go API Layer                 │
│  (HTTP server, agentic loop, tool system)   │
├─────────────┬───────────────┬───────────────┤
│  llama.cpp  │  Tool Runner  │  Context Mgr  │
│  (CGo bind) │  (sandboxed)  │  (files, RAG) │
├─────────────┴───────────────┴───────────────┤
│           Python Sidecar (gRPC)             │
│  (fine-tuning, embeddings, HF transformers) │
└─────────────────────────────────────────────┘
```

### Why this split

- **Go** handles the core runtime: API, agentic loop, tool execution, context management
- **llama.cpp** binds directly into Go via CGo (libraries like go-llama.cpp exist)
- **Python sidecar** handles ML-heavy work: LoRA fine-tuning, embedding generation, HF transformers
- Communication over **gRPC** — clean boundary, sidecar only runs when needed

### Go-native where possible

- Inference: CGo bindings to llama.cpp (no Python needed)
- Vector store: SQLite-vec or Qdrant Go client
- Tool execution: Go excels at subprocess management and sandboxing
- File I/O, context loading: trivial in Go

### Python only for

- LoRA/QLoRA fine-tuning (PEFT + transformers)
- Tokenizer operations if needed
- Embedding model loading (though llama.cpp can also do embeddings)

## Tech Stack

| Layer              | Technology                              |
|--------------------|-----------------------------------------|
| Language (core)    | Go                                      |
| Inference backend  | llama.cpp via CGo bindings              |
| API layer          | Go HTTP server (OpenAI-compatible)      |
| Fine-tuning        | Python sidecar — HF transformers + PEFT |
| Vector store (RAG) | SQLite-vec or Qdrant                    |
| Agent framework    | Custom agentic loop in Go               |
| Sandboxing         | Docker containers or Firecracker        |
| Communication      | gRPC between Go and Python sidecar      |

## Phases

### Phase 1: Model Serving (Current)
- Load GGUF models via llama.cpp CGo bindings
- Run inference (text generation, chat completion)
- Expose OpenAI-compatible HTTP API
- Model management: list, load, unload, download
- CLI interface: `tanrenai serve`, `tanrenai chat`, `tanrenai list`

### Phase 2: Agentic Loop
- Tool calling system with JSON schema definitions
- Parse LLM output for tool invocations
- Built-in tools: file read/write, shell execution, web search, code execution
- Conversation loop: user → LLM → tool call → execute → feed back → repeat
- Sandboxed command execution (Docker containers)

### Phase 3: Context Management
- Conversation history management with token counting
- Automatic summarization of old messages
- Load files/documents into context
- Context protocol for extensibility (similar to MCP)
- System prompt management

### Phase 4: Memory via RAG
- Store conversations in vector database
- Embedding generation (llama.cpp or dedicated model)
- Retrieve relevant past conversations at inference time
- Hybrid search: semantic + keyword
- Memory management: forget, prioritize, consolidate

### Phase 5: Fine-tuning Pipeline
- Python sidecar for LoRA/QLoRA training
- Conversation data extraction and formatting
- Training pipeline: extract → format → train → merge adapter
- Scheduled fine-tuning (e.g., nightly on new conversations)
- A/B testing between base model and fine-tuned adapter
- Catastrophic forgetting mitigation strategies

## Key Design Decisions

1. **RAG first, fine-tuning later** — RAG is simpler and more reliable. Fine-tuning gives deeper learning but risks degradation
2. **OpenAI-compatible API** — de facto standard, easy to integrate with existing tools
3. **GGUF model format** — community standard for local models via llama.cpp
4. **Single binary** — Go compiles to a single binary (plus llama.cpp shared lib)
5. **Incremental build** — each phase is usable standalone

## Domains

- tanrenai.com
- tanrenai.dev
