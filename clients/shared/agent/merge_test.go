package agent

import (
	"testing"

	"github.com/ThatCatDev/tanrenai/shared/pkg/api"
)

func TestMergeSystemMessages_NoSystem(t *testing.T) {
	msgs := []api.Message{
		{Role: "user", Content: "hello"},
		{Role: "assistant", Content: "hi"},
	}
	result := mergeSystemMessages(msgs)
	if len(result) != 2 {
		t.Fatalf("expected 2 messages, got %d", len(result))
	}
	if result[0].Role != "user" {
		t.Errorf("first message role = %q, want user", result[0].Role)
	}
}

func TestMergeSystemMessages_SingleSystem(t *testing.T) {
	msgs := []api.Message{
		{Role: "system", Content: "You are helpful."},
		{Role: "user", Content: "hello"},
	}
	result := mergeSystemMessages(msgs)
	if len(result) != 2 {
		t.Fatalf("expected 2 messages, got %d", len(result))
	}
	if result[0].Role != "system" || result[0].Content != "You are helpful." {
		t.Errorf("system message = %q", result[0].Content)
	}
}

func TestMergeSystemMessages_MultipleSystem(t *testing.T) {
	msgs := []api.Message{
		{Role: "system", Content: "You are helpful."},
		{Role: "user", Content: "hello"},
		{Role: "assistant", Content: "hi"},
		{Role: "system", Content: "Remember: be concise."},
		{Role: "user", Content: "tell me more"},
	}
	result := mergeSystemMessages(msgs)
	if len(result) != 4 { // 1 merged system + user + assistant + user
		t.Fatalf("expected 4 messages, got %d", len(result))
	}
	if result[0].Role != "system" {
		t.Fatalf("first message should be system, got %q", result[0].Role)
	}
	if result[0].Content != "You are helpful.\n\nRemember: be concise." {
		t.Errorf("merged content = %q", result[0].Content)
	}
	if result[1].Role != "user" || result[1].Content != "hello" {
		t.Errorf("second message = %v", result[1])
	}
}

func TestMergeSystemMessages_SystemAfterToolCalls(t *testing.T) {
	msgs := []api.Message{
		{Role: "system", Content: "prompt"},
		{Role: "user", Content: "do something"},
		{Role: "assistant", Content: "", ToolCalls: []api.ToolCall{{ID: "1"}}},
		{Role: "tool", Content: "result", ToolCallID: "1"},
		{Role: "system", Content: "scroll content"},
	}
	result := mergeSystemMessages(msgs)
	// System messages should be at position 0 only
	if result[0].Role != "system" {
		t.Fatalf("first message should be system")
	}
	if result[0].Content != "prompt\n\nscroll content" {
		t.Errorf("merged = %q", result[0].Content)
	}
	// No system messages after position 0
	for i := 1; i < len(result); i++ {
		if result[i].Role == "system" {
			t.Errorf("found system message at position %d", i)
		}
	}
}

func TestMergeSystemMessages_EmptySystem(t *testing.T) {
	msgs := []api.Message{
		{Role: "system", Content: ""},
		{Role: "system", Content: "real content"},
		{Role: "user", Content: "hello"},
	}
	result := mergeSystemMessages(msgs)
	if result[0].Content != "real content" {
		t.Errorf("should skip empty system, got %q", result[0].Content)
	}
}
