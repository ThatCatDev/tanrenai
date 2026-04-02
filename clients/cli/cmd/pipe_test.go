package cmd

import (
	"bufio"
	"strings"
	"testing"

	"github.com/ThatCatDev/tanrenai/shared/apiclient"
	"github.com/ThatCatDev/tanrenai/shared/pkg/api"
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
