package mcp

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"os/exec"
	"sync"

	sdkmcp "github.com/modelcontextprotocol/go-sdk/mcp"
)

// Client is tanrenai's view of one connected MCP server. Wraps the
// SDK's ClientSession with the per-server config (so error messages
// can name the server) and a cached tool list (one ListTools call at
// connect time; refreshed only on disconnect/reconnect).
//
// Not exposed to callers — they interact with the Registry, which
// holds a Client per configured server and surfaces their tools via
// the existing tools.Tool interface (see tool.go).
type Client struct {
	Name   string
	Config ServerConfig

	session *sdkmcp.ClientSession
	mu      sync.RWMutex
	tools   []*sdkmcp.Tool
}

// Tools returns the MCP-side tool descriptors discovered at connect.
// Stable for the lifetime of the connection.
func (c *Client) Tools() []*sdkmcp.Tool {
	c.mu.RLock()
	defer c.mu.RUnlock()
	out := make([]*sdkmcp.Tool, len(c.tools))
	copy(out, c.tools)
	return out
}

// Call invokes a tool by its server-side name and returns the result.
// The arguments string is the model's raw JSON (matching the schema
// the server advertised); we unmarshal to map[string]any here so the
// SDK can re-marshal cleanly.
func (c *Client) Call(ctx context.Context, toolName, arguments string) (*sdkmcp.CallToolResult, error) {
	c.mu.RLock()
	session := c.session
	c.mu.RUnlock()
	if session == nil {
		return nil, fmt.Errorf("mcp %q: not connected", c.Name)
	}

	args := map[string]any{}
	if arguments != "" {
		if err := json.Unmarshal([]byte(arguments), &args); err != nil {
			return nil, fmt.Errorf("mcp %q: parse arguments: %w", c.Name, err)
		}
	}

	return session.CallTool(ctx, &sdkmcp.CallToolParams{
		Name:      toolName,
		Arguments: args,
	})
}

// Close disconnects the session. Idempotent — calling on an already-
// closed client is a no-op rather than an error.
func (c *Client) Close() error {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.session == nil {
		return nil
	}
	err := c.session.Close()
	c.session = nil
	c.tools = nil
	return err
}

// Connect dials the configured server (stdio or HTTP) and lists its
// tools. The returned Client is ready to Call until Close. On
// connection failure the error includes the server name so callers
// can surface "couldn't reach <name>" to the user.
func Connect(ctx context.Context, name string, cfg ServerConfig) (*Client, error) {
	transport, err := buildTransport(cfg)
	if err != nil {
		return nil, fmt.Errorf("mcp %q: %w", name, err)
	}
	return connectWithTransport(ctx, name, cfg, transport)
}

// connectWithTransport is the shared core of Connect — extracted so
// tests can pass an InMemoryTransport without going through the
// stdio/HTTP config path. Not exported because the only legitimate
// non-config transport is the in-package test rig.
func connectWithTransport(
	ctx context.Context,
	name string,
	cfg ServerConfig,
	transport sdkmcp.Transport,
) (*Client, error) {
	client := sdkmcp.NewClient(
		&sdkmcp.Implementation{Name: "tanrenai", Version: "dev"},
		nil,
	)

	session, err := client.Connect(ctx, transport, nil)
	if err != nil {
		return nil, fmt.Errorf("mcp %q: connect: %w", name, err)
	}

	// One-shot tool discovery. The MCP spec allows servers to send
	// ToolListChanged notifications later; tanrenai doesn't subscribe
	// to those in v1 — agent sessions are short-lived, so the cost of
	// reconnecting on next session is acceptable.
	listed, err := session.ListTools(ctx, &sdkmcp.ListToolsParams{})
	if err != nil {
		_ = session.Close()
		return nil, fmt.Errorf("mcp %q: list tools: %w", name, err)
	}

	return &Client{
		Name:    name,
		Config:  cfg,
		session: session,
		tools:   listed.Tools,
	}, nil
}

// buildTransport picks the SDK transport for a server config.
// stdio → CommandTransport (subprocess); http → StreamableClientTransport
// (the current MCP HTTP spec). SSE-only servers can be added behind an
// explicit `type: "sse"` config field later if real users need it.
func buildTransport(cfg ServerConfig) (sdkmcp.Transport, error) {
	switch cfg.Transport() {
	case TransportStdio:
		cmd := exec.Command(cfg.Command, cfg.Args...)
		if len(cfg.Env) > 0 {
			// exec.Command starts with the parent's env. Append the
			// config's overrides; later entries win in Go's exec.Env.
			env := append([]string(nil), cmd.Environ()...)
			for k, v := range cfg.Env {
				env = append(env, k+"="+v)
			}
			cmd.Env = env
		}
		return &sdkmcp.CommandTransport{Command: cmd}, nil

	case TransportHTTP:
		t := &sdkmcp.StreamableClientTransport{Endpoint: cfg.URL}
		if len(cfg.Headers) > 0 {
			// SDK doesn't expose a header field directly — pass them
			// via a custom HTTP client. The headerRoundTripper below
			// stamps every outbound request with our headers.
			t.HTTPClient = &http.Client{
				Transport: &headerRoundTripper{
					base:    http.DefaultTransport,
					headers: cfg.Headers,
				},
			}
		}
		return t, nil

	default:
		return nil, fmt.Errorf("invalid transport: set either `command` (stdio) or `url` (http)")
	}
}

// headerRoundTripper wraps another RoundTripper to inject static
// headers on every request. Used for HTTP MCP servers that need
// Authorization, X-API-Key, etc. — the config's `headers` map maps
// 1:1 to header lines.
type headerRoundTripper struct {
	base    http.RoundTripper
	headers map[string]string
}

func (h *headerRoundTripper) RoundTrip(req *http.Request) (*http.Response, error) {
	// Clone before mutating — RoundTrip contract requires not modifying
	// the caller's request struct.
	cloned := req.Clone(req.Context())
	for k, v := range h.headers {
		cloned.Header.Set(k, v)
	}
	return h.base.RoundTrip(cloned)
}
