<p align="center">
  <img src="resources/logo.png" alt="Tanrenai" width="200">
</p>

<h1 align="center">Tanrenai (鍛錬AI)</h1>

<p align="center">A local AI assistant with agentic tool use, context windowing, and memory/RAG.<br>Download a single binary, pull a model, and start chatting.</p>

<p align="center"><strong><a href="https://tanrenai.com">tanrenai.com</a></strong></p>

## Install

Each release bundles `llama-server` (the inference engine) — no extra downloads needed.

### Linux

**One-line install** (recommended):

```bash
curl -fsSL https://raw.githubusercontent.com/ThatCatDev/tanrenai/main/installers/install.sh | sh
```

This downloads the latest release and installs to `~/.local/bin/`.

**Debian/Ubuntu (.deb)**:

```bash
# Resolve the latest CLI release (the repo also has a separate vscode-v* tag
# line, so /releases/latest can't be used directly), then download its .deb:
VER=$(curl -fsSL "https://api.github.com/repos/ThatCatDev/tanrenai/releases?per_page=100" \
  | grep '"tag_name"' | sed 's/.*"tag_name": *"//;s/".*//' \
  | grep -E '^v[0-9]+\.[0-9]+' | head -n 1)
curl -fsSLO "https://github.com/ThatCatDev/tanrenai/releases/download/${VER}/tanrenai-${VER#v}-amd64.deb"
sudo dpkg -i tanrenai-*-amd64.deb
```

**Arch Linux**:

The one-line installer above works on Arch as-is. To manage tanrenai with
`pacman` instead, build the package from the PKGBUILD in this repo:

```bash
git clone https://github.com/ThatCatDev/tanrenai.git
cd tanrenai/installers/arch
makepkg -si
```

This installs `/usr/bin/tanrenai` plus the bundled `llama-server` under
`/usr/share/tanrenai/bin`. The CLI only looks for the inference engine in its
per-user data dir, so link it once per user:

```bash
mkdir -p ~/.local/share/tanrenai/bin
ln -sf /usr/share/tanrenai/bin/* ~/.local/share/tanrenai/bin/
```

To bump the package to a newer release, edit `pkgver` in the PKGBUILD and run
`updpkgsums` (from `pacman-contrib`) to refresh the checksums.

**Manual**:

```bash
tar xzf tanrenai-cli-linux-amd64.tar.gz
cd tanrenai-cli-linux-amd64
chmod +x tanrenai-cli bin/*
mkdir -p ~/.local/bin ~/.local/share/tanrenai/bin
cp tanrenai-cli ~/.local/bin/tanrenai
cp bin/* ~/.local/share/tanrenai/bin/
```

> The CLI resolves `llama-server` from `~/.local/share/tanrenai/bin` (override
> the parent with `TANRENAI_DATA_DIR`). Copying it to a system path such as
> `/usr/local/share` will not be picked up.

**PATH**: the installer puts the binary in `~/.local/bin`. If that's not on your
PATH yet:

```bash
# bash / zsh
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc   # or ~/.zshrc

# fish
fish_add_path ~/.local/bin
```

### macOS


**One-line install** (recommended):

```bash
curl -fsSL https://raw.githubusercontent.com/ThatCatDev/tanrenai/main/installers/install.sh | sh
```

**Manual**:

```bash
tar xzf tanrenai-cli-darwin-arm64.tar.gz   # or darwin-amd64 for Intel
cd tanrenai-cli-darwin-arm64
chmod +x tanrenai-cli bin/*
cp tanrenai-cli ~/.local/bin/tanrenai
mkdir -p ~/.local/share/tanrenai/bin
cp bin/* ~/.local/share/tanrenai/bin/
```

### Windows

**Installer (recommended):** Download `tanrenai-cli-windows-amd64-setup.exe` from [Releases](https://github.com/ThatCatDev/tanrenai/releases). Adds `tanrenai` to your PATH and shows up in Add/Remove Programs.

**Manual:** Download `tanrenai-cli-windows-amd64.zip`, extract it, and run from that folder.

### All platforms

| Platform | File | GPU Support |
|----------|------|-------------|
| Linux x64 | `tanrenai-cli-linux-amd64.tar.gz` | CPU only* |
| macOS Intel | `tanrenai-cli-darwin-amd64.tar.gz` | - |
| macOS Apple Silicon | `tanrenai-cli-darwin-arm64.tar.gz` | Metal |
| Windows x64 | `tanrenai-cli-windows-amd64.zip` | NVIDIA CUDA 12.4 |

\* **Linux NVIDIA GPU users:** The bundled `llama-server` is CPU-only. To enable CUDA acceleration, build llama.cpp from source and replace the bundled binary:

```bash
# Debian/Ubuntu
sudo apt install -y build-essential cmake libcurl4-openssl-dev nvidia-cuda-toolkit
# Arch
sudo pacman -S --needed base-devel cmake curl cuda

git clone https://github.com/ggerganov/llama.cpp.git /tmp/llama.cpp
cd /tmp/llama.cpp
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release -j$(nproc)
cp /tmp/llama.cpp/build/bin/llama-server ~/.local/share/tanrenai/bin/
cp /tmp/llama.cpp/build/bin/*.so* ~/.local/share/tanrenai/bin/
```

On Arch, `nvcc` is installed to `/opt/cuda/bin`, which is not on PATH by
default — prefix the `cmake` calls with `PATH=/opt/cuda/bin:$PATH` if the CUDA
build can't find it.

### Troubleshooting

**`connection refused` on 127.0.0.1:8080 after logging in.** `tanrenai login`
takes the platform API URL from `--platform-url`, then `TANRENAI_SERVER_URL`,
then `--server-url` — which defaults to `http://127.0.0.1:8080`. Logging in
without any of them records *localhost* as your platform, and every later
command inherits it from the stored credentials. Check what got saved:

```bash
grep -o '"server_url": *"[^"]*"' ~/.local/share/tanrenai/credentials.json
```

If that isn't your platform, log in again with both URLs set:

```bash
tanrenai login \
  --web-url https://dev.tanrenai.com \
  --platform-url https://api.dev.tanrenai.com
```

Setting `TANRENAI_SERVER_URL` alone will not repair an existing bad login — the
stored `server_url` overrides it. Only an explicit `--server-url` wins.

**`tanrenai` errors with a 404, or HTML from some other web app.** Same default
of `http://127.0.0.1:8080`, but something else is answering on it — port 8080 is
widely claimed. Check what owns it with `ss -ltnp | grep :8080`, then either use
single-binary mode or point at the right backend:

```bash
tanrenai --local list                             # embedded GPU + backend
tanrenai --server-url http://127.0.0.1:9090 list  # a backend elsewhere
```

**`command not found: tanrenai`.** `~/.local/bin` isn't on your PATH — see the
PATH note under [Linux](#linux).

## Quick Start

The CLI embeds the GPU server and backend in a single binary — no separate services needed.

### 1. Pull a model

Download a GGUF model from Hugging Face:

```bash
tanrenai --local pull hf://unsloth/Qwen3-8B-GGUF/Q4_K_M
```

Models are stored in `~/.local/share/tanrenai/models/` (Linux/macOS) or `%LOCALAPPDATA%\tanrenai\models\` (Windows).

### 2. Chat with a model

```bash
# Basic chat
tanrenai --local run Qwen3-8B-Q4_K_M

# Agent mode with tool use
tanrenai --local run Qwen3-8B-Q4_K_M --agent

# Agent mode with memory/RAG
tanrenai --local run Qwen3-8B-Q4_K_M --agent --memory
```

### 3. List available models

```bash
tanrenai --local list
```

## CLI Reference

```
tanrenai [flags] <command>

Commands:
  run <model>     Load a model and start interactive chat
  pull <url>      Download a GGUF model
  list            List available models
  version         Print version information

Global Flags:
  --local               Run with embedded servers (recommended)
  --server-url string   Connect to a remote backend instead (default "http://127.0.0.1:8080")
  --gpu-layers int      GPU offload layers; -1 = all (default -1)
  --flash-attn          Enable flash attention (default true)

Run Flags:
  --agent               Enable agent mode with tool calling
  --memory              Enable persistent memory/RAG
  --pipe                Non-interactive pipe mode (stdin/stdout)
  --system string       Custom system prompt
  --system-file string  Load system prompt from file
  --ctx-size int        Override context size (0 = auto-detect from model)
  --max-iterations int  Max agent iterations per turn (0 = unlimited)
  --context-file strings  Inject files into context
```

### Pipe Mode

For programmatic interaction (scripts, CI, other tools), use `--pipe` to bypass the TUI:

```bash
printf 'Create a hello world script at /tmp/hello.py\n---END---' | \
  tanrenai --local run Qwen3.5-4B-Q4_K_M --agent --pipe
```

- **Input**: messages delimited by `---END---` (supports multi-line)
- **stdout**: streamed assistant content + `---END---` after each turn
- **stderr**: status tags (`[thinking]`, `[tool_call]`, `[tool_result]`, etc.)
- Tools are auto-approved in pipe mode
- Full session context (memory, scrolls, context windowing) works across turns

## Agent Mode

In agent mode, the model can use tools to interact with your filesystem and system:

`file_read`, `file_write`, `patch_file`, `list_dir`, `find_files`, `grep_search`, `git_info`, `shell_exec`, `web_search`

```bash
tanrenai --local run Qwen3-8B-Q4_K_M --agent
```

### Tool Permissions

When the agent invokes a tool, you'll be prompted to approve it. The options depend on the tool's risk level:

**Read-only tools** (`file_read`, `list_dir`, `find_files`, `grep_search`, `git_info`):
- **Allow** — run this once
- **Always Allow** — blanket-approve this tool
- **Block** — skip the call

**File-writing tools** (`file_write`, `patch_file`):
- **Allow** — run this once
- **Always (path)** — always allow this specific file path
- **Always (all)** — blanket-approve all writes
- **Block** — skip the call

**Shell commands** (`shell_exec`):
- **Allow** — run this once
- **Always (git \*)** — always allow commands starting with `git` (or whichever prefix)
- **Block** — skip the call

Shell commands cannot be blanket-approved — you must approve by command prefix for safety.

Use arrow keys or mouse to select, Enter to confirm. Permissions are saved to `.tanrenai/permissions.json` in the current directory and merged with `~/.tanrenai/permissions.json` (global).

## Memory / RAG

Enable persistent memory so the model remembers across conversations:

```bash
tanrenai --local run Qwen3-8B-Q4_K_M --agent --memory
```

Memories are stored locally in a vector database with embeddings generated by the model.

## REPL Commands

While chatting, use these slash commands:

- `/memory` — view stored memories
- `/context` — show current context window
- `/tokens` — display token budget usage
- `/clear` — reset the conversation

## Context Size

Context size is auto-detected from GGUF model metadata. Override it if needed:

```bash
# Auto-detect (default)
tanrenai --local run Qwen3-8B-Q4_K_M --agent

# Force 8192 tokens
tanrenai --local run Qwen3-8B-Q4_K_M --agent --ctx-size 8192
```

## GPU Acceleration

By default, all layers are offloaded to the GPU (`--gpu-layers -1`). Adjust if you have limited VRAM:

```bash
# Offload 20 layers to GPU, rest on CPU
tanrenai --local run Qwen3-8B-Q4_K_M --agent --gpu-layers 20

# CPU only
tanrenai --local run Qwen3-8B-Q4_K_M --agent --gpu-layers 0
```

## Desktop App

A GTK4/Adwaita desktop app is also available from [Releases](https://github.com/ThatCatDev/tanrenai/releases). It embeds everything in a GUI — no terminal needed.

| Platform | Notes |
|----------|-------|
| Linux | Requires GTK 4 and libadwaita: `sudo apt install libgtk-4-1 libadwaita-1-0` |
| macOS | Requires GTK 4 and libadwaita: `brew install gtk4 libadwaita` |
| Windows | **Installer:** `tanrenai-desktop-windows-amd64-setup.exe` (adds to PATH + Start Menu). Or use the `.zip` for manual install. |

## Pulling Models

Supports HuggingFace GGUF models with `hf://` references:

```bash
# Auto-pick best quant from a repo
tanrenai --local pull hf://unsloth/Qwen3.5-27B-GGUF

# Specific quant
tanrenai --local pull hf://unsloth/Qwen3.5-27B-GGUF/Q4_K_M

# Split GGUF (multi-file, downloads all parts)
tanrenai --local pull hf://unsloth/Qwen3.5-122B-A10B-GGUF/UD-Q4_K_XL

# Direct URL (still works)
tanrenai --local pull https://huggingface.co/owner/repo/resolve/main/model.gguf
```

## Advanced: Three-Tier Setup

For remote GPU servers or multi-machine setups, run the tiers separately:

```bash
# GPU server (on the machine with the GPU)
tanrenai-gpu serve --port 11435

# Backend (can run anywhere, points at GPU server)
tanrenai-server serve --gpu-url http://gpu-host:11435 --memory --port 8080

# CLI (connects to backend)
tanrenai run model-name --agent --memory --server-url http://backend-host:8080
```

## Cloud Deployment (vast.ai)

Deploy the GPU server to vast.ai, run the backend as a Docker container, and connect the CLI. The three tiers communicate over a Headscale/Tailscale mesh network — no public IPs needed.

### Step 1: Deploy GPU Server to vast.ai

Build the infra CLI and deploy:

```bash
cd infra/ && go build -o tanrenai-infra .

# Interactive — pick existing instance or create new
# --model-size auto-calculates VRAM and disk requirements
tanrenai-infra deploy \
  --model-size 120b \
  --max-cost 3.0 \
  --network headscale \
  --headscale-url https://headscale.floretos.com \
  --headscale-api-key $HEADSCALE_API_KEY
```

| Model Size | Min VRAM | Disk Allocated |
|-----------|----------|----------------|
| 8b | 7 GB | 100 GB |
| 27b | 19 GB | 100 GB |
| 72b | 46 GB | 150 GB |
| 120b | 74 GB | 200 GB |

The deploy process:
1. Search offers or select existing instance (interactive TUI)
2. SSH in, install deps, build llama.cpp with CUDA, build tanrenai-gpu
3. Install Tailscale and join Headscale network
4. Start GPU server, health check through tunnel

### Step 2: Run Backend (Docker)

The backend handles memory/RAG and proxies requests to the GPU server.

**With Headscale** (recommended for vast.ai) — auto-discovers GPU server on mesh:

```bash
docker run -d --name tanrenai-server -p 8080:8080 \
  -e HEADSCALE_API_KEY="$HEADSCALE_API_KEY" \
  -e HEADSCALE_URL="https://headscale.floretos.com" \
  -e TAILSCALE_HOSTNAME="tanrenai-server" \
  harbor.floret.dev/tanrenai/server-tailscale:latest \
  serve --memory
```

The image auto-generates a Headscale auth key, joins the network, discovers the GPU server, and sets `--gpu-url` automatically.

**Without Headscale** — point directly at a GPU server:

```bash
# Build from source
docker build -f server/docker/Dockerfile -t tanrenai-server .

# Run (replace GPU_HOST with your GPU server's address)
docker run -d --name tanrenai-server -p 8080:8080 \
  tanrenai-server \
  serve --host 0.0.0.0 --gpu-url http://GPU_HOST:11435 --memory
```

### Step 3: Connect CLI

```bash
# Pull a model (downloads to the GPU server)
tanrenai pull --server-url http://localhost:8080 hf://unsloth/Qwen3.5-122B-A10B-GGUF/UD-Q4_K_XL

# Chat with agent mode and memory
tanrenai run model-name --agent --memory --server-url http://localhost:8080
```

Or set the env var to avoid passing `--server-url` every time:

```bash
export TANRENAI_SERVER_URL=http://localhost:8080
tanrenai run model-name --agent --memory
```

| Env Var | Description |
|---------|-------------|
| `HEADSCALE_API_KEY` | Headscale API key (auto-generates auth key) |
| `HEADSCALE_URL` | Headscale server URL |
| `TAILSCALE_HOSTNAME` | Node hostname |
| `GPU_HOSTNAME` | GPU node filter (default: `tanrenai-gpu-`) |
| `HEADSCALE_USER` | Headscale user (default: `tanrenai`) |

Or pass a pre-generated auth key directly with `TAILSCALE_AUTH_KEY`.

### Pull Model to GPU Server

```bash
tanrenai pull --server-url http://localhost:8080 hf://unsloth/Qwen3.5-122B-A10B-GGUF/UD-Q4_K_XL
```

### Other Infra Commands

```bash
tanrenai-infra list                           # List vast.ai instances
tanrenai-infra status --vastai-instance-id ID # Instance details
tanrenai-infra destroy --vastai-instance-id ID # Tear down

tanrenai-infra network auth-key               # Generate Headscale pre-auth key
tanrenai-infra network nodes                  # List all Headscale nodes
tanrenai-infra network join --hostname NAME   # Join this machine to network
```

### Docker Images

| Image | Registry | Size |
|-------|----------|------|
| `server` | `harbor.floret.dev/tanrenai/server` | ~10 MB |
| `server-tailscale` | `harbor.floret.dev/tanrenai/server-tailscale` | ~48 MB |

Built and pushed automatically on merge to main via GitHub Actions.

## Architecture

```
Clients (clients/)  →  Backend (server/)  →  GPU Server (gpu/)
CLI, Desktop           memory/RAG, proxy,    llama-server,
agent loop             vast.ai mgmt          models, fine-tuning
                                    ↑
                       Infra (infra/)
                       vast.ai deploy,
                       Headscale tunnel
```

Four independent Go modules communicating via JSON over HTTP. In `--local` mode, all three are embedded in the CLI binary.

## Development Setup

### Prerequisites

**Go** (1.25+):

```bash
# Option A: Ubuntu/Debian package (may lag behind latest)
sudo apt install -y golang-go

# Option B: Install latest from go.dev
curl -fsSL https://go.dev/dl/go1.24.1.linux-amd64.tar.gz -o /tmp/go.tar.gz
sudo rm -rf /usr/local/go && sudo tar -C /usr/local -xzf /tmp/go.tar.gz
echo 'export PATH=$PATH:/usr/local/go/bin:$HOME/go/bin' >> ~/.bashrc
source ~/.bashrc
```

**Build tools:**

```bash
sudo apt install -y build-essential cmake libcurl4-openssl-dev
```

**NVIDIA CUDA** (required for GPU-accelerated inference):

```bash
sudo apt install -y nvidia-cuda-toolkit
```

Verify with `nvcc --version` and `nvidia-smi`.

### Building llama.cpp (CUDA)

The GPU server manages a `llama-server` subprocess. To build it with CUDA support:

```bash
git clone https://github.com/ggerganov/llama.cpp.git /tmp/llama.cpp
cd /tmp/llama.cpp
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release -j$(nproc)

# Install the binary and shared libraries where tanrenai expects them
mkdir -p ~/.local/share/tanrenai/bin
cp /tmp/llama.cpp/build/bin/llama-server ~/.local/share/tanrenai/bin/
cp /tmp/llama.cpp/build/bin/*.so* ~/.local/share/tanrenai/bin/
```

For CPU-only builds, omit `-DGGML_CUDA=ON`.

### Building from Source

```bash
# Build and install CLI to ~/.local/bin/
make install

# Or build individual components
make build          # CLI only
make build-all      # CLI + GPU server + backend + infra

# Or build each module directly
cd gpu/          && go build -o tanrenai-gpu .
cd server/       && go build -o tanrenai-server .
cd clients/cli/  && go build -o tanrenai .
cd infra/        && go build -o tanrenai-infra .

# Desktop (requires GTK4 dev libraries)
cd clients/desktop/ && go build -o tanrenai-desktop .
```

### Testing

```bash
make test

# Or per-module
cd gpu/    && go test ./...
cd server/ && go test ./...
cd clients/cli/ && go test ./...
```

## License

See [LICENSE](LICENSE) for details.
