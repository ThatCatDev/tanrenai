# Tanrenai for VS Code

The Tanrenai agent inside your editor. Streams chat from the Tanrenai backend, runs tools against the open workspace, previews file edits in VS Code's native diff editor.

## Architecture

This extension is a thin TypeScript shell around the Go agent. It spawns the `tanrenai agent-rpc` subcommand and drives it over a small NDJSON-over-stdio protocol. The agent loop, swarm, memory, and most tools all run in Go — only `file_read`, `file_write`, and `patch_file` are intercepted on the TypeScript side so they see unsaved buffer contents and apply via `WorkspaceEdit`.

## Development

Requirements: Node 20+, Go 1.25+ (for the CLI).

```bash
cd clients/vscode
npm install
npm run build              # esbuild → dist/extension.js
```

To run inside a VS Code extension-development host:

1. Open `clients/vscode/` in VS Code.
2. Press `F5` (or "Run Extension" from the Run panel).
3. A new VS Code window opens with the extension loaded.
4. Click the Tanrenai icon in the activity bar.

The dev cycle expects `tanrenai` to be on `$PATH` — install it from `clients/cli/` with `go install .`. Production builds bundle the binary in `dist/bin/<platform>-<arch>/` (see `Makefile`'s `vscode-bundle` target).

## Settings

- `tanrenai.serverUrl` — backend URL override (default: read from credentials)
- `tanrenai.model` — default model
- `tanrenai.agentMode` — enable tool calling (default: true)
- `tanrenai.cliPath` — override the bundled CLI binary
