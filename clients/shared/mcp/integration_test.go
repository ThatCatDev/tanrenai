package mcp

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"testing"
	"time"

	sdkmcp "github.com/modelcontextprotocol/go-sdk/mcp"

	"github.com/ThatCatDev/tanrenai/shared/tools"
)

// startTestServer builds an MCP server with a couple of tools and
// returns the client-side transport its peer uses to connect. The
// server runs on a goroutine for the duration of the test.
//
// Tools exposed:
//   - "echo": returns the `msg` argument verbatim as text content
//   - "boom": always returns IsError=true with a "kaboom" message
//   - "image-tool": returns one TextContent + two ImageContent blocks
//     so the projection summary path gets covered
func startTestServer(t *testing.T) sdkmcp.Transport {
	t.Helper()

	server := sdkmcp.NewServer(
		&sdkmcp.Implementation{Name: "test-server", Version: "v0"},
		nil,
	)

	type echoIn struct {
		Msg string `json:"msg"`
	}
	type echoOut struct {
		Echo string `json:"echo"`
	}
	sdkmcp.AddTool(
		server,
		&sdkmcp.Tool{Name: "echo", Description: "echo back"},
		func(ctx context.Context, req *sdkmcp.CallToolRequest, in echoIn) (*sdkmcp.CallToolResult, echoOut, error) {
			return nil, echoOut{Echo: in.Msg}, nil
		},
	)
	type boomIn struct{}
	sdkmcp.AddTool(
		server,
		&sdkmcp.Tool{Name: "boom", Description: "always errors"},
		func(ctx context.Context, req *sdkmcp.CallToolRequest, in boomIn) (*sdkmcp.CallToolResult, any, error) {
			return &sdkmcp.CallToolResult{
				IsError: true,
				Content: []sdkmcp.Content{&sdkmcp.TextContent{Text: "kaboom"}},
			}, nil, nil
		},
	)
	type imgIn struct{}
	sdkmcp.AddTool(
		server,
		&sdkmcp.Tool{Name: "image-tool", Description: "returns text + images"},
		func(ctx context.Context, req *sdkmcp.CallToolRequest, in imgIn) (*sdkmcp.CallToolResult, any, error) {
			return &sdkmcp.CallToolResult{
				Content: []sdkmcp.Content{
					&sdkmcp.TextContent{Text: "saw two screenshots"},
					&sdkmcp.ImageContent{Data: []byte("fake-png-1"), MIMEType: "image/png"},
					&sdkmcp.ImageContent{Data: []byte("fake-png-2"), MIMEType: "image/png"},
				},
			}, nil, nil
		},
	)

	clientT, serverT := sdkmcp.NewInMemoryTransports()
	go func() {
		// Run for the test lifetime; the test cancels via ctx in
		// individual subtests if needed. Errors from Run after the
		// in-memory peer closes are expected and noisy — discard.
		_ = server.Run(context.Background(), serverT)
	}()
	t.Cleanup(func() {
		// In-memory transports close when both ends do; the client's
		// Close (via Registry.Close in tests) handles the cleanup.
	})
	return clientT
}

func TestIntegration_ConnectAndListTools(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), 5*time.Second)
	defer cancel()

	transport := startTestServer(t)
	client, err := connectWithTransport(ctx, "test-srv", ServerConfig{}, transport)
	if err != nil {
		t.Fatalf("connect: %v", err)
	}
	defer client.Close()

	names := map[string]bool{}
	for _, tt := range client.Tools() {
		names[tt.Name] = true
	}
	for _, want := range []string{"echo", "boom", "image-tool"} {
		if !names[want] {
			t.Errorf("missing %q in listed tools: %v", want, names)
		}
	}
}

func TestIntegration_CallEchoTool(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), 5*time.Second)
	defer cancel()

	client, err := connectWithTransport(ctx, "test", ServerConfig{}, startTestServer(t))
	if err != nil {
		t.Fatalf("connect: %v", err)
	}
	defer client.Close()

	res, err := client.Call(ctx, "echo", `{"msg":"hello world"}`)
	if err != nil {
		t.Fatalf("Call: %v", err)
	}
	if res.IsError {
		t.Errorf("unexpected IsError; content=%v", res.Content)
	}
	// Echo tool returns structured output, which the SDK projects into
	// a single TextContent block whose JSON contains the echo string.
	body := contentText(res.Content)
	if !strings.Contains(body, "hello world") {
		t.Errorf("output missing echo text: %q", body)
	}
}

func TestIntegration_IsErrorRoundTrips(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), 5*time.Second)
	defer cancel()

	client, err := connectWithTransport(ctx, "test", ServerConfig{}, startTestServer(t))
	if err != nil {
		t.Fatalf("connect: %v", err)
	}
	defer client.Close()

	// boom returns IsError=true. The agent path relies on that flag
	// to drive its stuck-detection and retry logic — verify it makes
	// it back through the wrapper instead of being lost.
	tool := &Tool{
		BareName: "boom",
		tool:     client.tools[indexOf(client.tools, "boom")],
		client:   client,
	}
	res, err := tool.Execute(ctx, "{}")
	if err != nil {
		t.Fatalf("Execute: %v", err)
	}
	if !res.IsError {
		t.Errorf("expected IsError=true (server returned IsError)")
	}
	if !strings.Contains(res.Output, "kaboom") {
		t.Errorf("output missing error text: %q", res.Output)
	}
}

func TestIntegration_ImageContentSummary(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), 5*time.Second)
	defer cancel()

	client, err := connectWithTransport(ctx, "test", ServerConfig{}, startTestServer(t))
	if err != nil {
		t.Fatalf("connect: %v", err)
	}
	defer client.Close()

	tool := &Tool{
		BareName: "image-tool",
		tool:     client.tools[indexOf(client.tools, "image-tool")],
		client:   client,
	}
	res, err := tool.Execute(ctx, "{}")
	if err != nil {
		t.Fatalf("Execute: %v", err)
	}
	// Projection rule: text comes through verbatim, non-text gets a
	// one-line summary at the end. Both must be visible so the model
	// sees that something more than just the text came back.
	if !strings.Contains(res.Output, "saw two screenshots") {
		t.Errorf("missing text content: %q", res.Output)
	}
	if !strings.Contains(res.Output, "2 image attachments") {
		t.Errorf("missing image-attachment summary: %q", res.Output)
	}
}

func TestIntegration_AttachToRegistryWithNamespace(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), 5*time.Second)
	defer cancel()

	client, err := connectWithTransport(ctx, "playwright", ServerConfig{}, startTestServer(t))
	if err != nil {
		t.Fatalf("connect: %v", err)
	}
	defer client.Close()

	reg := &Registry{clients: []*Client{client}}
	tr := tools.NewRegistry()
	reg.AttachTo(tr)

	// Every MCP tool lands under `<server>__<tool>` so two MCP servers
	// or an MCP server + a built-in can't collide.
	if got := tr.Get("playwright" + NameSeparator + "echo"); got == nil {
		t.Errorf("echo not registered under namespaced name")
	}
	// Bare names are NOT registered — only the namespaced form.
	if got := tr.Get("echo"); got != nil {
		t.Errorf("bare echo should not be registered (would collide with future built-in)")
	}
}

func TestIntegration_DispatchThroughRegistry(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), 5*time.Second)
	defer cancel()

	client, err := connectWithTransport(ctx, "pw", ServerConfig{}, startTestServer(t))
	if err != nil {
		t.Fatalf("connect: %v", err)
	}
	defer client.Close()

	reg := &Registry{clients: []*Client{client}}
	tr := tools.NewRegistry()
	reg.AttachTo(tr)

	tool := tr.Get("pw" + NameSeparator + "echo")
	if tool == nil {
		t.Fatal("tool not found")
	}
	res, err := tool.Execute(ctx, `{"msg":"round trip"}`)
	if err != nil {
		t.Fatalf("Execute: %v", err)
	}
	if res.IsError {
		t.Errorf("unexpected error result: %+v", res)
	}
	if !strings.Contains(res.Output, "round trip") {
		t.Errorf("dispatch did not reach server: %q", res.Output)
	}
}

func TestIntegration_DisconnectedCallReturnsError(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), 5*time.Second)
	defer cancel()

	client, err := connectWithTransport(ctx, "test", ServerConfig{}, startTestServer(t))
	if err != nil {
		t.Fatalf("connect: %v", err)
	}
	// Close before calling. The wrapper must surface this as an
	// error rather than panicking on a nil session.
	if err := client.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	_, err = client.Call(ctx, "echo", "{}")
	if err == nil || !strings.Contains(err.Error(), "not connected") {
		t.Errorf("expected not-connected error, got %v", err)
	}
}

func TestParametersFallbackForNilSchema(t *testing.T) {
	// A pathological MCP server could expose a tool with no schema.
	// The adapter must still produce valid JSON so the agent's
	// OpenAI-format tool list isn't malformed.
	tool := &Tool{
		PrefixedName: "x__y",
		BareName:     "y",
		tool:         &sdkmcp.Tool{Name: "y"},
		client:       &Client{Name: "x"},
	}
	var got any
	if err := json.Unmarshal(tool.Parameters(), &got); err != nil {
		t.Fatalf("fallback schema not valid JSON: %v", err)
	}
}

// ── helpers ───────────────────────────────────────────────────────

// indexOf finds the position of a tool by name in the cached list.
func indexOf(ts []*sdkmcp.Tool, name string) int {
	for i, t := range ts {
		if t.Name == name {
			return i
		}
	}
	return -1
}

// contentText concatenates all text content blocks — used for
// assertions that don't care about ordering, just substring presence.
func contentText(blocks []sdkmcp.Content) string {
	var sb strings.Builder
	for _, b := range blocks {
		if tc, ok := b.(*sdkmcp.TextContent); ok {
			sb.WriteString(tc.Text)
		}
	}
	return sb.String()
}

var _ = errors.Is // keep import in case tests grow Is-checks later
