package mcp

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"sync"

	"github.com/ThatCatDev/tanrenai/shared/tools"
)

// Registry owns the set of connected MCP clients for one agent session.
// Lifetime: created during setupSession, used to dispatch tool calls
// throughout the session, Disposed when the session ends.
//
// The agent's existing tools.Registry holds the wrappers — this object
// owns the *underlying* clients so we know how to disconnect them when
// the session ends. Don't confuse the two.
type Registry struct {
	mu      sync.Mutex
	clients []*Client
}

// ConnectAll dials every server in cfg in parallel and returns a
// Registry holding the successful connections. Per-server failures
// are logged + included in the returned (joined) error but don't
// abort the whole startup — a broken server shouldn't make the agent
// unusable when other servers are fine.
func ConnectAll(ctx context.Context, cfg Config) (*Registry, error) {
	type result struct {
		client *Client
		err    error
		name   string
	}
	out := make(chan result, len(cfg.Servers))
	var wg sync.WaitGroup

	for name, sc := range cfg.Servers {
		if sc.Disabled {
			continue
		}
		wg.Add(1)
		go func(name string, sc ServerConfig) {
			defer wg.Done()
			c, err := Connect(ctx, name, sc)
			out <- result{client: c, err: err, name: name}
		}(name, sc)
	}
	go func() { wg.Wait(); close(out) }()

	r := &Registry{}
	var errs []error
	for res := range out {
		if res.err != nil {
			// One bad apple shouldn't block the agent; surface the
			// error to the caller AND log it so the user sees what
			// went wrong even if the rest of startup succeeds.
			slog.Warn("mcp: failed to connect", "server", res.name, "error", res.err)
			errs = append(errs, res.err)
			continue
		}
		r.clients = append(r.clients, res.client)
	}
	return r, errors.Join(errs...)
}

// AttachTo registers every connected client's tools onto the given
// tools.Registry under namespaced names (`<server>__<tool>`).
//
// Collision policy: if the registry already has a tool with a given
// prefixed name, the MCP tool is skipped + a warning logged. Built-in
// tools never collide because none of them contain the `__` separator,
// so a collision can only happen with another MCP server that uses
// the same server name (illegal — Config.Validate already catches it)
// OR a developer-introduced built-in that adopts the same form.
func (r *Registry) AttachTo(reg *tools.Registry) {
	r.mu.Lock()
	defer r.mu.Unlock()
	for _, client := range r.clients {
		for _, t := range adaptTools(client) {
			if existing := reg.Get(t.Name()); existing != nil {
				slog.Warn("mcp: tool name collision — skipping", "name", t.Name())
				continue
			}
			reg.Register(t)
		}
	}
}

// Tools returns a flat list of every (server, tool-name) pair attached
// to this registry. Useful for `tanrenai mcp list`-style diagnostics
// and for the test suite.
func (r *Registry) Tools() []ToolInfo {
	r.mu.Lock()
	defer r.mu.Unlock()
	var out []ToolInfo
	for _, c := range r.clients {
		for _, t := range c.Tools() {
			out = append(out, ToolInfo{
				Server:      c.Name,
				Name:        t.Name,
				Description: t.Description,
			})
		}
	}
	return out
}

// ToolInfo is the simplified view of an MCP tool used by callers that
// don't want to depend on the SDK types directly (UI rendering, logging).
type ToolInfo struct {
	Server      string
	Name        string
	Description string
}

// Close disconnects every client. Errors from individual clients are
// logged and joined into one returned error so cleanup doesn't bail
// halfway through. Idempotent.
func (r *Registry) Close() error {
	r.mu.Lock()
	defer r.mu.Unlock()
	var errs []error
	for _, c := range r.clients {
		if err := c.Close(); err != nil {
			errs = append(errs, fmt.Errorf("mcp %q close: %w", c.Name, err))
		}
	}
	r.clients = nil
	return errors.Join(errs...)
}
