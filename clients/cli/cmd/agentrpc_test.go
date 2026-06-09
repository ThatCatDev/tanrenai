package cmd

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/ThatCatDev/tanrenai/client/internal/chatctx"
	"github.com/ThatCatDev/tanrenai/shared/agent"
	"github.com/ThatCatDev/tanrenai/shared/pkg/api"
	"github.com/ThatCatDev/tanrenai/shared/tools"
)

// ── rpcServer.RequestTool ──────────────────────────────────────────────

func TestRPCServer_RequestToolWritesAndReturnsResult(t *testing.T) {
	var buf bytes.Buffer
	var bufMu sync.Mutex // json.Encoder isn't safe for concurrent reads of the buffer
	srv := newRPCServer(&lockedWriter{w: &buf, mu: &bufMu}, []string{"file_read"}, nil)

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
	srv := newRPCServer(&lockedWriter{w: &buf, mu: &bufMu}, nil, nil)

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
	srv := newRPCServer(&lockedWriter{w: &buf, mu: &bufMu}, nil, nil)

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
	srv := newRPCServer(&lockedWriter{w: &buf, mu: &bufMu}, nil, nil)

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

func TestRPCServer_AlwaysPersistsPermission(t *testing.T) {
	// Run in a tmp dir so .tanrenai/permissions.json is local to the test.
	tmp := t.TempDir()
	prevWd, _ := os.Getwd()
	if err := os.Chdir(tmp); err != nil {
		t.Fatalf("chdir: %v", err)
	}
	defer func() { _ = os.Chdir(prevWd) }()

	var buf bytes.Buffer
	var bufMu sync.Mutex
	perms := tools.LoadPermissions()
	srv := newRPCServer(&lockedWriter{w: &buf, mu: &bufMu}, nil, perms)

	// shell_exec for `ls -la`. After "always", we expect a rule scoped to
	// the command prefix (`ls *`) saved to .tanrenai/permissions.json.
	resCh := make(chan agent.ApprovalAction, 1)
	go func() {
		resCh <- srv.requestApproval(context.Background(), api.ToolCall{
			Function: api.ToolCallFunction{Name: "shell_exec", Arguments: `{"command":"ls -la"}`},
		})
	}()

	id := waitForApprovalID(t, &buf, &bufMu)
	srv.handleApprovalResponse(rpcApprovalResponseMsg{
		Type: "approval_response", ID: id, Action: "always",
	})

	select {
	case <-resCh:
	case <-time.After(2 * time.Second):
		t.Fatal("requestApproval did not return")
	}

	data, err := os.ReadFile(filepath.Join(".tanrenai", "permissions.json"))
	if err != nil {
		t.Fatalf("expected .tanrenai/permissions.json to exist: %v", err)
	}
	if !strings.Contains(string(data), "shell_exec") {
		t.Errorf("permissions file missing shell_exec rule: %s", data)
	}
	if !strings.Contains(string(data), "ls *") {
		t.Errorf("expected `ls *` prefix rule, got: %s", data)
	}

	// A second call with `ls -lh` should now be auto-allowed without
	// emitting a new approval_required event.
	bufMu.Lock()
	buf.Reset()
	bufMu.Unlock()
	// Reload permissions so the in-memory state matches what was saved.
	perms2 := tools.LoadPermissions()
	srv2 := newRPCServer(&lockedWriter{w: &buf, mu: &bufMu}, nil, perms2)
	action := srv2.requestApproval(context.Background(), api.ToolCall{
		Function: api.ToolCallFunction{Name: "shell_exec", Arguments: `{"command":"ls -lh"}`},
	})
	if action != agent.ApprovalAllow {
		t.Errorf("second `ls *` call should be auto-allowed, got %v", action)
	}
	if buf.Len() != 0 {
		t.Errorf("expected no approval_required for already-allowed call, got: %s", buf.String())
	}
}

// ── ready message contents ─────────────────────────────────────────────

// TestRPCServer_TokenRateThrottle pins the contract used by the agent
// pathways: recordRateAndMaybeEmit fires at most one token_rate message
// per tokenRateThrottle window during streaming, and flushTokenRate emits
// one final value at turn end (excluding short windows that the tracker
// suppresses).
func TestRPCServer_TokenRateThrottle(t *testing.T) {
	var buf bytes.Buffer
	var bufMu sync.Mutex
	srv := newRPCServer(&lockedWriter{w: &buf, mu: &bufMu}, nil, nil)

	// Hammer recordRateAndMaybeEmit far faster than the throttle. Even
	// though we record many tokens, only the FIRST eligible call emits —
	// subsequent calls within the window are throttled.
	for range 50 {
		srv.recordRateAndMaybeEmit()
		time.Sleep(5 * time.Millisecond) // stay well under tokenRateThrottle
	}

	bufMu.Lock()
	mid := buf.String()
	bufMu.Unlock()

	// The tracker requires ≥2 tokens AND ≥100ms before reporting a rate,
	// so the first ~20 records produce nothing, then one message lands.
	// Anywhere from 0–2 token_rate events is fine — we only insist we
	// haven't dumped 50.
	midCount := strings.Count(mid, `"type":"token_rate"`)
	if midCount > 2 {
		t.Errorf("token_rate emitted %d times during throttled window, want ≤2", midCount)
	}

	// Final flush always emits when there's a meaningful rate, regardless
	// of throttle. That's the value the webview lands on after turn_done.
	srv.flushTokenRate()
	bufMu.Lock()
	full := buf.String()
	bufMu.Unlock()
	if strings.Count(full, `"type":"token_rate"`) < midCount+1 {
		t.Errorf("flushTokenRate did not emit a final message; got %q", full)
	}
}

// TestRPCServer_TokenRateResetBetweenTurns is the regression test for
// per-turn isolation: after resetTokenRate, a subsequent stream's first
// delta starts a fresh window, so back-to-back fast turns don't show the
// previous turn's wall clock as elapsed.
func TestRPCServer_TokenRateResetBetweenTurns(t *testing.T) {
	var buf bytes.Buffer
	var bufMu sync.Mutex
	srv := newRPCServer(&lockedWriter{w: &buf, mu: &bufMu}, nil, nil)

	srv.recordRateAndMaybeEmit()
	srv.recordRateAndMaybeEmit()

	// Force the lastRateEmit window to be in the past so the next call
	// would otherwise emit. After reset, it should NOT emit until enough
	// time AND tokens accrue post-reset.
	srv.resetTokenRate()
	if tokens, _ := srv.genRate.Snapshot(); tokens != 0 {
		t.Errorf("after reset tokens = %d, want 0", tokens)
	}

	// flushTokenRate on a freshly-reset tracker must be a no-op (no
	// recorded tokens means no rate to report).
	bufMu.Lock()
	before := buf.Len()
	bufMu.Unlock()
	srv.flushTokenRate()
	bufMu.Lock()
	if buf.Len() != before {
		t.Errorf("flushTokenRate wrote %d bytes on empty tracker, want 0", buf.Len()-before)
	}
	bufMu.Unlock()
}

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

// ── context_usage + compaction ────────────────────────────────────────

func TestRPCServer_EmitContextUsage(t *testing.T) {
	var buf bytes.Buffer
	var bufMu sync.Mutex
	srv := newRPCServer(&lockedWriter{w: &buf, mu: &bufMu}, nil, nil)

	mgr := chatctx.NewManager(chatctx.Config{CtxSize: 4096, ResponseBudget: 256}, chatctx.NewTokenEstimator())
	mgr.SetSystemPrompt("you are helpful")
	mgr.Append(api.Message{Role: "user", Content: "hi"})
	mgr.Append(api.Message{Role: "assistant", Content: "hello!"})

	srv.emitContextUsage(mgr)

	bufMu.Lock()
	out := buf.String()
	bufMu.Unlock()

	var msg rpcContextUsageMsg
	if err := json.Unmarshal([]byte(strings.TrimSpace(out)), &msg); err != nil {
		t.Fatalf("decode: %v\nraw=%q", err, out)
	}
	if msg.Type != "context_usage" {
		t.Errorf("Type = %q", msg.Type)
	}
	if msg.Total != 4096 {
		t.Errorf("Total = %d, want 4096", msg.Total)
	}
	if msg.System <= 0 {
		t.Errorf("System should reflect the system prompt, got %d", msg.System)
	}
	if msg.HistoryCount != 2 {
		t.Errorf("HistoryCount = %d, want 2", msg.HistoryCount)
	}
	if msg.Available <= 0 {
		t.Errorf("Available should be positive, got %d", msg.Available)
	}
}

func TestRPCServer_EmitContextUsage_NilManagerIsSafe(t *testing.T) {
	var buf bytes.Buffer
	var bufMu sync.Mutex
	srv := newRPCServer(&lockedWriter{w: &buf, mu: &bufMu}, nil, nil)

	// Must not panic, and must not emit anything when there's no manager.
	srv.emitContextUsage(nil)

	bufMu.Lock()
	out := buf.String()
	bufMu.Unlock()
	if out != "" {
		t.Errorf("expected no output with nil manager, got %q", out)
	}
}

func TestRPCServer_SummariseWithEvents_EmitsStartAndDone(t *testing.T) {
	var buf bytes.Buffer
	var bufMu sync.Mutex
	srv := newRPCServer(&lockedWriter{w: &buf, mu: &bufMu}, nil, nil)

	// Tiny window — every Append forces Summarize to actually do work.
	mgr := chatctx.NewManager(chatctx.Config{CtxSize: 200, ResponseBudget: 20}, chatctx.NewTokenEstimator())
	mgr.SetSystemPrompt("system")
	for i := 0; i < 10; i++ {
		mgr.Append(api.Message{Role: "user", Content: strings.Repeat("aaaa ", 20)})
		mgr.Append(api.Message{Role: "assistant", Content: strings.Repeat("bbbb ", 20)})
	}

	complete := agent.CompletionFunc(func(_ context.Context, _ *api.ChatCompletionRequest) (*api.ChatCompletionResponse, error) {
		return &api.ChatCompletionResponse{
			Choices: []api.Choice{{
				Message: api.Message{Role: "assistant", Content: "summary of older turns"},
			}},
		}, nil
	})

	// NeedsSummary should be true given the small window — sanity-check
	// the precondition rather than asserting Summarize did nothing.
	if !mgr.NeedsSummary() {
		t.Fatal("test setup didn't fill the budget; adjust the seed messages")
	}
	if err := srv.summariseWithEvents(context.Background(), mgr, complete); err != nil {
		t.Fatalf("summariseWithEvents: %v", err)
	}

	bufMu.Lock()
	out := buf.String()
	bufMu.Unlock()

	lines := strings.Split(strings.TrimSpace(out), "\n")
	if len(lines) < 2 {
		t.Fatalf("expected at least 2 compaction events, got %d: %q", len(lines), out)
	}
	var phases []string
	for _, ln := range lines {
		var ev rpcCompactionMsg
		if err := json.Unmarshal([]byte(ln), &ev); err != nil {
			t.Fatalf("decode %q: %v", ln, err)
		}
		if ev.Type != "compaction" {
			t.Errorf("Type = %q", ev.Type)
		}
		phases = append(phases, ev.Phase)
	}
	if phases[0] != "start" {
		t.Errorf("first phase = %q, want start", phases[0])
	}
	if phases[len(phases)-1] != "done" {
		t.Errorf("last phase = %q, want done", phases[len(phases)-1])
	}
}

func TestRPCServer_SummariseWithEvents_EmitsNoopWhenNothingToCompact(t *testing.T) {
	var buf bytes.Buffer
	var bufMu sync.Mutex
	srv := newRPCServer(&lockedWriter{w: &buf, mu: &bufMu}, nil, nil)

	// Large window + tiny history → NeedsSummary() returns false, so
	// "Compact now" should report nothing-to-do, not a misleading
	// "Compacted 0 messages into summary".
	mgr := chatctx.NewManager(chatctx.Config{CtxSize: 8192, ResponseBudget: 256}, chatctx.NewTokenEstimator())
	mgr.SetSystemPrompt("system")
	mgr.Append(api.Message{Role: "user", Content: "hi"})
	mgr.Append(api.Message{Role: "assistant", Content: "hello"})

	if mgr.NeedsSummary() {
		t.Fatal("test setup is wrong — Manager already wants to compact")
	}

	called := false
	complete := agent.CompletionFunc(func(_ context.Context, _ *api.ChatCompletionRequest) (*api.ChatCompletionResponse, error) {
		called = true
		return &api.ChatCompletionResponse{}, nil
	})

	if err := srv.summariseWithEvents(context.Background(), mgr, complete); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if called {
		t.Error("complete() should not be called when NeedsSummary is false")
	}

	bufMu.Lock()
	out := strings.TrimSpace(buf.String())
	bufMu.Unlock()

	var ev rpcCompactionMsg
	if err := json.Unmarshal([]byte(out), &ev); err != nil {
		t.Fatalf("decode %q: %v", out, err)
	}
	if ev.Phase != "noop" {
		t.Errorf("Phase = %q, want noop", ev.Phase)
	}
}

func TestRPCServer_SummariseWithEvents_EmitsErrorPhase(t *testing.T) {
	var buf bytes.Buffer
	var bufMu sync.Mutex
	srv := newRPCServer(&lockedWriter{w: &buf, mu: &bufMu}, nil, nil)

	mgr := chatctx.NewManager(chatctx.Config{CtxSize: 200, ResponseBudget: 20}, chatctx.NewTokenEstimator())
	mgr.SetSystemPrompt("system")
	for i := 0; i < 10; i++ {
		mgr.Append(api.Message{Role: "user", Content: strings.Repeat("aaaa ", 20)})
		mgr.Append(api.Message{Role: "assistant", Content: strings.Repeat("bbbb ", 20)})
	}

	complete := agent.CompletionFunc(func(_ context.Context, _ *api.ChatCompletionRequest) (*api.ChatCompletionResponse, error) {
		return nil, errors.New("backend down")
	})

	err := srv.summariseWithEvents(context.Background(), mgr, complete)
	if err == nil {
		t.Fatal("expected error to propagate")
	}

	bufMu.Lock()
	out := buf.String()
	bufMu.Unlock()
	lines := strings.Split(strings.TrimSpace(out), "\n")
	var lastPhase, lastErr string
	for _, ln := range lines {
		var ev rpcCompactionMsg
		if jerr := json.Unmarshal([]byte(ln), &ev); jerr == nil {
			lastPhase = ev.Phase
			lastErr = ev.Error
		}
	}
	if lastPhase != "error" {
		t.Errorf("last phase = %q, want error", lastPhase)
	}
	if !strings.Contains(lastErr, "backend down") {
		t.Errorf("expected error string to include reason, got %q", lastErr)
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
