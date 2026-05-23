package cmd

import (
	"bufio"
	"encoding/json"
	"errors"
	"io"
	"os"
	"strings"
	"testing"

	"github.com/ThatCatDev/tanrenai/shared/apiclient"
	"github.com/ThatCatDev/tanrenai/shared/pkg/api"
	"github.com/ThatCatDev/tanrenai/shared/tools"
)

func TestReadPipeMessage_SingleLine(t *testing.T) {
	input := "hello world\n---END---\n"
	scanner := bufio.NewScanner(strings.NewReader(input))
	msg, eof := readPipeMessage(scanner)
	if msg != "hello world" {
		t.Errorf("got %q, want %q", msg, "hello world")
	}
	if eof {
		t.Error("expected eof=false")
	}
}

func TestReadPipeMessage_MultiLine(t *testing.T) {
	input := "line one\nline two\nline three\n---END---\n"
	scanner := bufio.NewScanner(strings.NewReader(input))
	msg, eof := readPipeMessage(scanner)
	want := "line one\nline two\nline three"
	if msg != want {
		t.Errorf("got %q, want %q", msg, want)
	}
	if eof {
		t.Error("expected eof=false")
	}
}

func TestReadPipeMessage_EOF(t *testing.T) {
	input := "hello world"
	scanner := bufio.NewScanner(strings.NewReader(input))
	msg, eof := readPipeMessage(scanner)
	if msg != "hello world" {
		t.Errorf("got %q, want %q", msg, "hello world")
	}
	if !eof {
		t.Error("expected eof=true")
	}
}

func TestReadPipeMessage_EmptyEOF(t *testing.T) {
	scanner := bufio.NewScanner(strings.NewReader(""))
	msg, eof := readPipeMessage(scanner)
	if msg != "" {
		t.Errorf("got %q, want empty", msg)
	}
	if !eof {
		t.Error("expected eof=true")
	}
}

func TestReadPipeMessage_MultipleTurns(t *testing.T) {
	input := "first message\n---END---\nsecond message\n---END---\n"
	scanner := bufio.NewScanner(strings.NewReader(input))

	msg1, eof1 := readPipeMessage(scanner)
	if msg1 != "first message" || eof1 {
		t.Errorf("turn 1: got %q eof=%v", msg1, eof1)
	}

	msg2, eof2 := readPipeMessage(scanner)
	if msg2 != "second message" || eof2 {
		t.Errorf("turn 2: got %q eof=%v", msg2, eof2)
	}
}

func TestReadPipeMessage_EmptyBetweenDelimiters(t *testing.T) {
	input := "---END---\nhello\n---END---\n"
	scanner := bufio.NewScanner(strings.NewReader(input))

	msg1, eof1 := readPipeMessage(scanner)
	if msg1 != "" || eof1 {
		t.Errorf("turn 1: got %q eof=%v", msg1, eof1)
	}

	msg2, eof2 := readPipeMessage(scanner)
	if msg2 != "hello" || eof2 {
		t.Errorf("turn 2: got %q eof=%v", msg2, eof2)
	}
}

// ── streamSimpleChat tests ────────────────────────────────────────────

// mockStreamEvents creates a channel that sends the given content deltas then DONE.
func mockStreamEvents(deltas ...string) <-chan apiclient.StreamEvent {
	ch := make(chan apiclient.StreamEvent, len(deltas)+1)
	go func() {
		defer close(ch)
		for _, d := range deltas {
			ch <- apiclient.StreamEvent{
				Chunk: &api.ChatCompletionChunk{
					Choices: []api.ChunkChoice{{
						Delta: api.MessageDelta{Content: d},
					}},
				},
			}
		}
		ch <- apiclient.StreamEvent{Done: true}
	}()
	return ch
}

func TestStreamSimpleChat_ContentOnly(t *testing.T) {
	events := mockStreamEvents("Hello", " World")

	var deltas []string
	content, err := streamSimpleChat(events, chatStreamHooks{
		OnContentDelta: func(delta string) {
			deltas = append(deltas, delta)
		},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if content != "Hello World" {
		t.Errorf("content = %q, want %q", content, "Hello World")
	}
	if len(deltas) != 2 {
		t.Errorf("got %d deltas, want 2", len(deltas))
	}
}

func TestStreamSimpleChat_ThinkingThenContent(t *testing.T) {
	ch := make(chan apiclient.StreamEvent, 5)
	go func() {
		defer close(ch)
		// Reasoning tokens
		ch <- apiclient.StreamEvent{
			Chunk: &api.ChatCompletionChunk{
				Choices: []api.ChunkChoice{{
					Delta: api.MessageDelta{ReasoningContent: "Let me think..."},
				}},
			},
		}
		// Content tokens
		ch <- apiclient.StreamEvent{
			Chunk: &api.ChatCompletionChunk{
				Choices: []api.ChunkChoice{{
					Delta: api.MessageDelta{Content: "Four"},
				}},
			},
		}
		ch <- apiclient.StreamEvent{Done: true}
	}()

	var thinkingCalled, thinkingDoneCalled bool
	content, err := streamSimpleChat(ch, chatStreamHooks{
		OnThinking:     func() { thinkingCalled = true },
		OnThinkingDone: func() { thinkingDoneCalled = true },
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if content != "Four" {
		t.Errorf("content = %q, want %q", content, "Four")
	}
	if !thinkingCalled {
		t.Error("OnThinking should have been called")
	}
	if !thinkingDoneCalled {
		t.Error("OnThinkingDone should have been called")
	}
}

func TestStreamSimpleChat_NoThinkingWhenNoReasoning(t *testing.T) {
	events := mockStreamEvents("direct answer")

	var thinkingCalled bool
	_, err := streamSimpleChat(events, chatStreamHooks{
		OnThinking: func() { thinkingCalled = true },
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if thinkingCalled {
		t.Error("OnThinking should not be called when there's no reasoning content")
	}
}

func TestStreamSimpleChat_Error(t *testing.T) {
	ch := make(chan apiclient.StreamEvent, 2)
	go func() {
		defer close(ch)
		ch <- apiclient.StreamEvent{
			Chunk: &api.ChatCompletionChunk{
				Choices: []api.ChunkChoice{{
					Delta: api.MessageDelta{Content: "partial"},
				}},
			},
		}
		ch <- apiclient.StreamEvent{Err: errStreamFailed}
	}()

	content, err := streamSimpleChat(ch, chatStreamHooks{})
	if err != errStreamFailed {
		t.Errorf("expected errStreamFailed, got %v", err)
	}
	if content != "partial" {
		t.Errorf("content = %q, want %q", content, "partial")
	}
}

var errStreamFailed = &testError{"stream failed"}

type testError struct{ msg string }

func (e *testError) Error() string { return e.msg }

func TestStreamSimpleChat_NilHooks(t *testing.T) {
	events := mockStreamEvents("hello")

	content, err := streamSimpleChat(events, chatStreamHooks{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if content != "hello" {
		t.Errorf("content = %q, want %q", content, "hello")
	}
}

func TestStreamSimpleChat_EmptyStream(t *testing.T) {
	ch := make(chan apiclient.StreamEvent, 1)
	go func() {
		defer close(ch)
		ch <- apiclient.StreamEvent{Done: true}
	}()

	content, err := streamSimpleChat(ch, chatStreamHooks{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if content != "" {
		t.Errorf("content = %q, want empty", content)
	}
}

// ── pipeEmitter / json mode tests ─────────────────────────────────────

// captureStdout redirects os.Stdout for the duration of fn and returns
// whatever was written. Pipe-mode emitters write directly to os.Stdout /
// os.Stderr so this is the cleanest way to assert their wire output without
// rewiring every emitter method to accept an io.Writer.
func captureStdout(t *testing.T, fn func()) string {
	t.Helper()
	old := os.Stdout
	r, w, err := os.Pipe()
	if err != nil {
		t.Fatalf("os.Pipe: %v", err)
	}
	os.Stdout = w
	done := make(chan string, 1)
	go func() {
		var buf strings.Builder
		_, _ = io.Copy(&buf, r)
		done <- buf.String()
	}()
	fn()
	_ = w.Close()
	os.Stdout = old
	return <-done
}

func captureStderr(t *testing.T, fn func()) string {
	t.Helper()
	old := os.Stderr
	r, w, err := os.Pipe()
	if err != nil {
		t.Fatalf("os.Pipe: %v", err)
	}
	os.Stderr = w
	done := make(chan string, 1)
	go func() {
		var buf strings.Builder
		_, _ = io.Copy(&buf, r)
		done <- buf.String()
	}()
	fn()
	_ = w.Close()
	os.Stderr = old
	return <-done
}

// parseEvents parses a JSONL stdout capture into a slice of decoded events,
// failing the test on any line that isn't valid JSON. The wire contract is
// "one event per line on stdout" so an unparseable line is a real bug, not
// something the host should have to tolerate.
func parseEvents(t *testing.T, captured string) []map[string]any {
	t.Helper()
	var out []map[string]any
	for i, line := range strings.Split(strings.TrimRight(captured, "\n"), "\n") {
		if line == "" {
			continue
		}
		var event map[string]any
		if err := json.Unmarshal([]byte(line), &event); err != nil {
			t.Fatalf("line %d not valid JSON: %q (err: %v)", i, line, err)
		}
		out = append(out, event)
	}
	return out
}

func TestNewPipeEmitter_FormatDispatch(t *testing.T) {
	cases := []struct {
		format   string
		wantJSON bool
	}{
		{"json", true},
		{"text", false},
		{"", false},
		{"garbage-value", false},
	}
	for _, tc := range cases {
		em := newPipeEmitter(tc.format)
		if tc.wantJSON {
			if _, ok := em.(*jsonEmitter); !ok {
				t.Errorf("format=%q got %T, want *jsonEmitter", tc.format, em)
			}
		} else {
			if _, ok := em.(*textEmitter); !ok {
				t.Errorf("format=%q got %T, want *textEmitter", tc.format, em)
			}
		}
	}
}

func TestJSONEmitter_TextDelta(t *testing.T) {
	em := &jsonEmitter{}
	out := captureStdout(t, func() {
		em.TextDelta("Hello, world!")
	})
	events := parseEvents(t, out)
	if len(events) != 1 {
		t.Fatalf("got %d events, want 1", len(events))
	}
	if events[0]["type"] != "text_delta" {
		t.Errorf("type = %v, want text_delta", events[0]["type"])
	}
	if events[0]["delta"] != "Hello, world!" {
		t.Errorf("delta = %v, want %q", events[0]["delta"], "Hello, world!")
	}
}

func TestJSONEmitter_ToolCall_ParsesStructuredArguments(t *testing.T) {
	em := &jsonEmitter{}
	call := api.ToolCall{
		ID:   "call_42",
		Type: "function",
		Function: api.ToolCallFunction{
			Name:      "file_read",
			Arguments: `{"path":"/tmp/x","limit":100}`,
		},
	}
	out := captureStdout(t, func() {
		em.ToolCall(call)
	})
	events := parseEvents(t, out)
	if len(events) != 1 {
		t.Fatalf("got %d events, want 1", len(events))
	}
	got := events[0]
	if got["type"] != "tool_call" {
		t.Errorf("type = %v, want tool_call", got["type"])
	}
	if got["id"] != "call_42" {
		t.Errorf("id = %v, want call_42", got["id"])
	}
	if got["name"] != "file_read" {
		t.Errorf("name = %v, want file_read", got["name"])
	}
	args, ok := got["arguments"].(map[string]any)
	if !ok {
		t.Fatalf("arguments = %v (%T), want parsed object", got["arguments"], got["arguments"])
	}
	if args["path"] != "/tmp/x" {
		t.Errorf("arguments.path = %v, want /tmp/x", args["path"])
	}
}

func TestJSONEmitter_ToolCall_FallsBackToRawStringForInvalidJSON(t *testing.T) {
	em := &jsonEmitter{}
	call := api.ToolCall{
		ID: "call_7",
		Function: api.ToolCallFunction{
			Name:      "file_write",
			Arguments: "this is not json{",
		},
	}
	out := captureStdout(t, func() {
		em.ToolCall(call)
	})
	events := parseEvents(t, out)
	if len(events) != 1 {
		t.Fatalf("got %d events, want 1", len(events))
	}
	if events[0]["arguments"] != "this is not json{" {
		t.Errorf("arguments = %v, want raw string fallback", events[0]["arguments"])
	}
}

func TestJSONEmitter_ToolResult(t *testing.T) {
	em := &jsonEmitter{}
	call := api.ToolCall{ID: "call_x", Function: api.ToolCallFunction{Name: "shell_exec"}}
	result := &tools.ToolResult{Output: "command failed", IsError: true}
	out := captureStdout(t, func() {
		em.ToolResult(call, result)
	})
	events := parseEvents(t, out)
	if len(events) != 1 {
		t.Fatalf("got %d events, want 1", len(events))
	}
	got := events[0]
	if got["type"] != "tool_result" {
		t.Errorf("type = %v, want tool_result", got["type"])
	}
	if got["tool_call_id"] != "call_x" {
		t.Errorf("tool_call_id = %v, want call_x", got["tool_call_id"])
	}
	if got["output"] != "command failed" {
		t.Errorf("output = %v, want command failed", got["output"])
	}
	if got["is_error"] != true {
		t.Errorf("is_error = %v, want true", got["is_error"])
	}
}

func TestJSONEmitter_TurnEnd_DefaultsReasonToStop(t *testing.T) {
	em := &jsonEmitter{}
	out := captureStdout(t, func() {
		em.TurnEnd("")
	})
	events := parseEvents(t, out)
	if len(events) != 1 {
		t.Fatalf("got %d events, want 1", len(events))
	}
	if events[0]["reason"] != "stop" {
		t.Errorf("reason = %v, want stop (default)", events[0]["reason"])
	}
}

func TestJSONEmitter_TurnEnd_HonoursExplicitReason(t *testing.T) {
	em := &jsonEmitter{}
	out := captureStdout(t, func() {
		em.TurnEnd("error")
	})
	events := parseEvents(t, out)
	if events[0]["reason"] != "error" {
		t.Errorf("reason = %v, want error", events[0]["reason"])
	}
}

func TestJSONEmitter_Error(t *testing.T) {
	em := &jsonEmitter{}
	out := captureStdout(t, func() {
		em.Error(errors.New("model not loaded"))
	})
	events := parseEvents(t, out)
	if events[0]["type"] != "error" {
		t.Errorf("type = %v, want error", events[0]["type"])
	}
	if events[0]["message"] != "model not loaded" {
		t.Errorf("message = %v, want %q", events[0]["message"], "model not loaded")
	}
}

func TestJSONEmitter_GenRate_SkipsWhenNoSamples(t *testing.T) {
	em := &jsonEmitter{}
	out := captureStdout(t, func() {
		em.GenRate(0, 0)
		em.GenRate(1, -1)
	})
	if out != "" {
		t.Errorf("gen_rate emitted output for zero/negative tps: %q", out)
	}
}

func TestJSONEmitter_StatusGoesToStdout(t *testing.T) {
	// Status events ARE part of the JSON stream (consumers may surface them
	// as UI hints) — they go on stdout, not stderr. Stderr is reserved for
	// startup log lines that pre-date the agent loop (model loading, GPU
	// readiness) and for slog-driven debug output.
	em := &jsonEmitter{}
	stdout := captureStdout(t, func() {
		stderr := captureStderr(t, func() {
			em.Status("thinking")
		})
		if stderr != "" {
			t.Errorf("status leaked to stderr: %q", stderr)
		}
	})
	events := parseEvents(t, stdout)
	if events[0]["type"] != "status" {
		t.Errorf("type = %v, want status", events[0]["type"])
	}
	if events[0]["label"] != "thinking" {
		t.Errorf("label = %v, want thinking", events[0]["label"])
	}
}

func TestJSONEmitter_StdoutIsParseableJSONLAcrossManyEvents(t *testing.T) {
	// Smoke test that a realistic mid-turn event sequence produces clean
	// one-event-per-line JSONL with no interleaving. Regression target for
	// any future change that adds stdout writes outside of `emit()`.
	em := &jsonEmitter{}
	out := captureStdout(t, func() {
		em.Status("thinking")
		em.TextDelta("Looking at the ")
		em.TextDelta("repo... ")
		em.ToolCall(api.ToolCall{ID: "c1", Function: api.ToolCallFunction{Name: "list_dir", Arguments: `{"path":"."}`}})
		em.ToolResult(api.ToolCall{ID: "c1", Function: api.ToolCallFunction{Name: "list_dir"}}, &tools.ToolResult{Output: "ok"})
		em.TextDelta("Done.")
		em.TurnEnd("stop")
	})
	events := parseEvents(t, out)
	if len(events) != 7 {
		t.Fatalf("got %d events, want 7", len(events))
	}
	wantTypes := []string{"status", "text_delta", "text_delta", "tool_call", "tool_result", "text_delta", "turn_end"}
	for i, w := range wantTypes {
		if events[i]["type"] != w {
			t.Errorf("event %d type = %v, want %s", i, events[i]["type"], w)
		}
	}
}

func TestTextEmitter_TurnEndPrintsDelimiterToStdout(t *testing.T) {
	em := &textEmitter{}
	out := captureStdout(t, func() {
		em.TurnEnd("anything")
	})
	if strings.TrimSpace(out) != pipeDelimiter {
		t.Errorf("got %q, want %q", strings.TrimSpace(out), pipeDelimiter)
	}
}

func TestTextEmitter_StatusGoesToStderrNotStdout(t *testing.T) {
	// Regression guard for the original pipe-mode bug where status messages
	// (model loading, iteration banners) leaked into stdout and corrupted
	// downstream consumers that treat stdout as assistant content.
	em := &textEmitter{}
	stdout := captureStdout(t, func() {
		stderr := captureStderr(t, func() {
			em.Status("loading model")
		})
		if !strings.Contains(stderr, "[loading model]") {
			t.Errorf("stderr = %q, want bracketed status line", stderr)
		}
	})
	if stdout != "" {
		t.Errorf("status leaked to stdout: %q", stdout)
	}
}

func TestStreamSimpleChat_ThinkingCalledOnce(t *testing.T) {
	ch := make(chan apiclient.StreamEvent, 5)
	go func() {
		defer close(ch)
		// Multiple reasoning tokens
		for _, r := range []string{"think", "think more", "think again"} {
			ch <- apiclient.StreamEvent{
				Chunk: &api.ChatCompletionChunk{
					Choices: []api.ChunkChoice{{
						Delta: api.MessageDelta{ReasoningContent: r},
					}},
				},
			}
		}
		ch <- apiclient.StreamEvent{
			Chunk: &api.ChatCompletionChunk{
				Choices: []api.ChunkChoice{{
					Delta: api.MessageDelta{Content: "answer"},
				}},
			},
		}
		ch <- apiclient.StreamEvent{Done: true}
	}()

	var thinkingCount int
	_, err := streamSimpleChat(ch, chatStreamHooks{
		OnThinking: func() { thinkingCount++ },
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if thinkingCount != 1 {
		t.Errorf("OnThinking called %d times, want 1", thinkingCount)
	}
}
