# Tanrenai for VS Code

The Tanrenai agent inside your editor. Streams chat from the Tanrenai backend, runs tools against the open workspace, and applies edits through VS Code's native edit pipeline so undo Just Works.

## Architecture

This extension is a thin TypeScript shell around the Go agent. It spawns the `tanrenai agent-rpc` subcommand (bundled per-platform inside the VSIX) and drives it over a small NDJSON-over-stdio protocol.

```
VS Code (Node)              ⇄  tanrenai agent-rpc (Go)
├── webview chat UI            ├── shared/agent (loop, swarm, planning)
├── auth + login flow          ├── shared/tools (memory, scrolls, shell, web)
├── intercepted tools:         └── apiclient → tanrenai backend
│   file_read, file_write,
│   patch_file
└── routes everything else to the Go agent
```

Only the editor-aware tools are intercepted — `file_read` reads from `vscode.workspace.openTextDocument` so it sees unsaved buffer contents, `file_write` and `patch_file` apply through `WorkspaceEdit`. Every other tool runs in Go and the extension never sees the implementation.

## Settings

| Setting | Default | What it does |
|---|---|---|
| `tanrenai.serverUrl` | `""` (read from credentials) | Override the backend URL. |
| `tanrenai.model` | `Qwen3.6-35B-A3B-UD-Q4_K_M` | Default model loaded at session start. |
| `tanrenai.agentMode` | `true` | Enable tool calling. |
| `tanrenai.cliPath` | `""` (auto-resolve) | Override the CLI binary. Empty = use the bundled binary, then fall back to `tanrenai` on PATH. |

Changing any of the above prompts a reconnect (settings only take effect on the next handshake).

## Commands

- `Tanrenai: Login` — browser-based OAuth, writes credentials to the same `credentials.json` the CLI uses.
- `Tanrenai: Logout` — disconnects and removes credentials.
- `Tanrenai: Reconnect` — restarts the subprocess.

## Authentication

Credentials are stored at:
- Linux/Mac: `~/.local/share/tanrenai/credentials.json`
- Windows: `%LOCALAPPDATA%\tanrenai\credentials.json`
- Override: `$TANRENAI_DATA_DIR/credentials.json`

Same format as the CLI. If you have already run `tanrenai login` in a terminal, the extension picks that up — and vice versa.

## Building from source (development)

Requirements: Node 20+, Go 1.25+.

```bash
# 1. From the repo root: cross-compile the CLI for every platform the
#    extension supports and copy the binaries into clients/vscode/dist/bin/.
make vscode-bundle

# 2. Build the extension bundle.
cd clients/vscode
npm install
npm run build      # → dist/extension.js (esbuild, ~40 KB)

# 3. Optional: package as an installable .vsix.
make vscode-package   # from repo root — chains bundle + build + vsce package
```

### Running in a dev host

1. `cd clients/vscode`, then open the folder in VS Code.
2. Press **F5** ("Run Extension"). A second VS Code window opens with Tanrenai loaded.
3. Click the Tanrenai icon in the activity bar.
4. If you haven't logged in: click **Sign in to Tanrenai** → browser opens → callback writes credentials → sidebar shows **Connected**.

The dev workflow uses `dist/bin/<platform>-<arch>/tanrenai` if it exists (so `make vscode-bundle` is enough), then falls back to `tanrenai` on PATH (so `go install` from `clients/cli/` also works).

## Troubleshooting

**"Could not find the tanrenai CLI"** — run `make vscode-bundle` from the repo root, or install the CLI with `cd clients/cli && go install .`, or set `tanrenai.cliPath` to your binary.

**"CLI subprocess exited unexpectedly"** — the error message includes the last few lines of stderr. Common causes: backend unreachable (check `tanrenai.serverUrl`), token expired (`Tanrenai: Login`), model not loaded yet (the platform auto-pulls; large models take minutes).

**Login times out** — the OAuth callback uses port 18293. If another process is bound there (e.g. a CLI `tanrenai login` already running), close it first.

**Credentials work in CLI but not extension** — they should share the file. Check `$TANRENAI_DATA_DIR` isn't set to different values in your shell vs VS Code's environment.

## Test plan

- `npx tsc --noEmit` — typecheck
- `npm run build` — esbuild bundle
- Manual E2E in the dev host (see above). No automated TS tests yet; planned for v1.1 with `@vscode/test-electron`.

## Roadmap

**v1.1**
- Inline chat over a selection (`Ctrl+I`)
- Diff-editor preview before applying writes
- Memory search visualization in chat
- Swarm progress panel in the webview
- "Tanrenai: Update CLI" command (download newer binary)

**v2**
- Inline ghost-text completions
- Marketplace listing
