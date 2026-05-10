package cmd

import (
	"bytes"
	"context"
	"encoding/json"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/ThatCatDev/tanrenai/shared/pkg/api"
	"github.com/ThatCatDev/tanrenai/shared/tools"
)

// ── rpcServer.RequestTool ──────────────────────────────────────────────

func TestRPCServer_RequestToolWritesAndReturnsResult(t *testing.T) {
	var buf bytes.Buffer
	var bufMu sync.Mutex // json.Encoder isn't safe for concurrent reads of the buffer
	srv := newRPCServer(&lockedWriter{w: &buf, mu: &bufMu}, []string{"file_read"})

	type result struct {
		out *tools.ToolResult
		err error
	}
	resultCh := make(chan result, 1)
	go func() {
		out, err := srv.RequestTool(context.Background(), "file_read", `{"path":"x"}`)
		resultCh <- result{out, err}
	}()

	// Wait for the request to be written, then parse the ID from it.
	id := waitForToolRequestID(t, &buf, &bufMu, "file_read")

	// Reply.
	srv.handleToolResult(rpcToolResultMsg{
		Type:    "tool_result",
		ID:      id,
		OK:      true,
		Content: "file contents",
	})

	select {
	case r := <-resultCh:
		if r.err != nil {
			t.Fatalf("RequestTool returned error: %v", r.err)
		}
		if r.out == nil || r.out.Output != "file contents" {
			t.Errorf("RequestTool returned %+v, want Output=%q", r.out, "file contents")
		}
	case <-time.After(2 * time.Second):
		t.Fatal("RequestTool did not return after handleToolResult")
	}
}

func TestRPCServer_RequestToolErrorResult(t *testing.T) {
	var buf bytes.Buffer
	var bufMu sync.Mutex
	srv := newRPCServer(&lockedWriter{w: &buf, mu: &bufMu}, nil)

	resultCh := make(chan *tools.ToolResult, 1)
	go func() {
		out, _ := srv.RequestTool(context.Background(), "x", "{}")
		resultCh <- out
	}()

	id := waitForToolRequestID(t, &buf, &bufMu, "x")
	srv.handleToolResult(rpcToolResultMsg{
		Type:  "tool_result",
		ID:    id,
		OK:    false,
		Error: "permission denied",
	})

	select {
	case out := <-resultCh:
		if out == nil || !out.IsError {
			t.Errorf("expected IsError result, got %+v", out)
		}
		if !strings.Contains(out.Output, "permission denied") {
			t.Errorf("expected error message in output, got %q", out.Output)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("RequestTool did not return")
	}
}

func TestRPCServer_RequestToolRespectsCancellation(t *testing.T) {
	var buf bytes.Buffer
	var bufMu sync.Mutex
	srv := newRPCServer(&lockedWriter{w: &buf, mu: &bufMu}, nil)

	ctx, cancel := context.WithCancel(context.Background())
	errCh := make(chan error, 1)
	go func() {
		_, err := srv.RequestTool(ctx, "x", "{}")
		errCh <- err
	}()

	// Make sure the goroutine started and wrote the request.
	waitForToolRequestID(t, &buf, &bufMu, "x")
	cancel()

	select {
	case err := <-errCh:
		if err == nil {
			t.Error("expected error from cancelled ctx, got nil")
		}
	case <-time.After(2 * time.Second):
		t.Fatal("RequestTool did not honour ctx cancellation")
	}
}

// ── Approval round-trip ────────────────────────────────────────────────

func TestRPCServer_ApprovalRoundTrip(t *testing.T) {
	var buf bytes.Buffer
	var bufMu sync.Mutex
	srv := newRPCServer(&lockedWriter{w: &buf, mu: &bufMu}, nil)

	resCh := make(chan int, 1)
	go func() {
		action := srv.requestApproval(context.Background(), api.ToolCall{
			Function: api.ToolCallFunction{Name: "shell_exec", Arguments: `{"cmd":"ls"}`},
		})
		resCh <- int(action)
	}()

	id := waitForApprovalID(t, &buf, &bufMu)
	srv.handleApprovalResponse(rpcApprovalResponseMsg{
		Type: "approval_response", ID: id, Action: "always",
	})

	select {
	case got := <-resCh:
		if got != 2 { // ApprovalAlwaysAllow == 2
			t.Errorf("got action %d, want ApprovalAlwaysAllow (2)", got)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("requestApproval did not return")
	}
}

// ── ready message contents ─────────────────────────────────────────────

func TestBuildReadyMsg_IncludesToolsAndModel(t *testing.T) {
	registry := tools.NewRegistry()
	registry.Register(&tools.FileReadTool{})
	registry.Register(&tools.ListDirTool{})

	deps := &sessionDeps{registry: registry, modelName: "Qwen3.6"}
	msg := buildReadyMsg(deps, "Qwen3.6")

	if msg.Type != "ready" {
		t.Errorf("Type = %q, want \"ready\"", msg.Type)
	}
	if msg.ProtocolVersion != AgentRPCProtocolVersion {
		t.Errorf("ProtocolVersion = %d, want %d", msg.ProtocolVersion, AgentRPCProtocolVersion)
	}
	if msg.Model != "Qwen3.6" {
		t.Errorf("Model = %q", msg.Model)
	}
	names := make([]string, len(msg.Tools))
	for i, t := range msg.Tools {
		names[i] = t.Name
	}
	want := []string{"file_read", "list_dir"}
	for _, w := range want {
		found := false
		for _, n := range names {
			if n == w {
				found = true

				break
			}
		}
		if !found {
			t.Errorf("tool %q missing from ready, got %v", w, names)
		}
	}
	for _, tool := range msg.Tools {
		if len(tool.Schema) == 0 {
			t.Errorf("tool %q has empty schema", tool.Name)
		}
	}
}

// ── helpers ───────────────────────────────────────────────────────────

// lockedWriter serializes writes to a buffer the test reads concurrently.
type lockedWriter struct {
	w  *bytes.Buffer
	mu *sync.Mutex
}

func (lw *lockedWriter) Write(p []byte) (int, error) {
	lw.mu.Lock()
	defer lw.mu.Unlock()

	return lw.w.Write(p)
}

// waitForToolRequestID polls the buffer until a tool_call_request line for
// `wantName` appears, then returns its id.
func waitForToolRequestID(t *testing.T, buf *bytes.Buffer, mu *sync.Mutex, wantName string) string {
	t.Helper()
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		mu.Lock()
		line, err := buf.ReadString('\n')
		mu.Unlock()
		if err == nil && strings.Contains(line, wantName) {
			var msg rpcToolCallMsg
			if jerr := json.Unmarshal([]byte(line), &msg); jerr == nil && msg.Type == "tool_call_request" {
				return msg.ID
			}
		}
		time.Sleep(5 * time.Millisecond)
	}
	t.Fatalf("did not see tool_call_request for %q within 2s", wantName)

	return ""
}

func waitForApprovalID(t *testing.T, buf *bytes.Buffer, mu *sync.Mutex) string {
	t.Helper()
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		mu.Lock()
		line, err := buf.ReadString('\n')
		mu.Unlock()
		if err == nil && strings.Contains(line, "approval_required") {
			var msg rpcApprovalRequiredMsg
			if jerr := json.Unmarshal([]byte(line), &msg); jerr == nil {
				return msg.ID
			}
		}
		time.Sleep(5 * time.Millisecond)
	}
	t.Fatal("did not see approval_required within 2s")

	return ""
}
