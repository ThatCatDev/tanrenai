# Tanrenai CLI

Command-line client for [Tanrenai](https://github.com/ThatCatDev/tanrenai) (鍛錬AI). Connects to the Tanrenai backend for LLM inference, agentic tool use, context management, and memory/RAG.

## Build

```bash
go build -o tanrenai .
```

## Usage

The CLI requires a running Tanrenai backend (default `http://127.0.0.1:8080`). Override with `--server-url`.

### Run a model

Load a model and start an interactive chat session:

```bash
tanrenai run <model-name>
```

### Agent mode

Enable agentic tool calling with filesystem and shell access:

```bash
tanrenai run <model-name> --agent
```

Available tools in agent mode: `file_read`, `file_write`, `patch_file`, `list_dir`, `find_files`, `grep_search`, `git_info`, `shell_exec`, `web_search`.

### Memory

Enable memory/RAG for persistent context across conversations:

```bash
tanrenai run <model-name> --agent --memory
```

### Chat with a loaded model

If a model is already loaded on the backend, connect directly:

```bash
tanrenai chat --model <model-name>
```

### List models

```bash
tanrenai list
```

### Pull a model

Download a GGUF model via the backend:

```bash
tanrenai pull <url>
```

## Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--server-url` | `http://127.0.0.1:8080` | Backend server URL |
| `--agent` | `false` | Enable agent mode with tool calling |
| `--memory` | `false` | Enable memory/RAG |
| `--system` | | System prompt |
| `--system-file` | | Read system prompt from a file |
| `--ctx-size` | `4096` | Context window size in tokens |
| `--response-budget` | `512` | Tokens reserved for model response |
| `--context-file` | | Files to load into context (repeatable) |
| `--max-iterations` | `200` | Max agent tool-call iterations per turn |

## REPL Commands

Inside a chat session:

| Command | Description |
|---------|-------------|
| `/clear` | Clear conversation history |
| `/compact` | Summarize conversation to free context |
| `/tokens` | Show token budget breakdown |
| `/context add <path>` | Load a file into context |
| `/context list` | Show loaded context files |
| `/context clear` | Remove all context files |
| `/memory` | List recent memories |
| `/memory search <query>` | Search memories |
| `/memory forget <id>` | Delete a memory by ID prefix |
| `/memory clear` | Clear all memories |
| `/quit`, `/exit` | Exit |
