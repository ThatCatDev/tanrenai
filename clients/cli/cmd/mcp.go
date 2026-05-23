package cmd

import (
	"context"
	"fmt"
	"os"

	"github.com/ThatCatDev/tanrenai/shared/mcp"
)

// attachMCP loads the layered mcp.json config (project + user),
// connects every enabled server, attaches the discovered tools to the
// agent's registry, and chains MCP disconnect into deps.cleanupFn so
// session teardown closes everything.
//
// Best-effort by design: connection failures are surfaced as one-line
// startup-log warnings, but don't abort the session. A user with a
// half-broken playwright-mcp config should still be able to use
// tanrenai with its built-in tools and the MCP servers that did
// connect — the alternative is "tanrenai is unusable until you fix
// every mcp.json entry", which is worse for iteration.
func attachMCP(ctx context.Context, p runParams, deps *sessionDeps, log *startupLog) {
	workspace, _ := os.Getwd()
	cfg, err := mcp.Load(workspace)
	if err != nil {
		log.Warn(fmt.Sprintf("mcp config: %v", err))
		return
	}
	if len(cfg.Servers) == 0 {
		return
	}

	registry, connectErr := mcp.ConnectAll(ctx, cfg)
	if connectErr != nil {
		// Joined error — each failing server gets a line so users see
		// every problem at once rather than fixing one and discovering
		// the next on the following run.
		log.Warn(fmt.Sprintf("mcp: %v", connectErr))
	}
	if tools := registry.Tools(); len(tools) > 0 {
		// One summary line per server so the startup log is scannable
		// even with many servers configured.
		bySrv := map[string]int{}
		for _, t := range tools {
			bySrv[t.Server]++
		}
		for srv, n := range bySrv {
			log.Info(fmt.Sprintf("MCP %s: %d tools", srv, n))
		}
	}
	registry.AttachTo(deps.registry)
	deps.cleanupFn = chainCleanup(deps.cleanupFn, func() { _ = registry.Close() })
	_ = p // reserved — future per-command overrides (--no-mcp, --mcp-server=...)
}

// chainCleanup wires a new tear-down function into the existing
// deps.cleanupFn slot. The result calls both in reverse-registration
// order (LIFO — matches defer semantics: thing started latest is
// closed first). nil-safe on either side.
func chainCleanup(existing, added func()) func() {
	if existing == nil {
		return added
	}
	if added == nil {
		return existing
	}
	return func() {
		added()
		existing()
	}
}
