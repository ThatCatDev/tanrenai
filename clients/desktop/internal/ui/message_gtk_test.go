package ui

import (
	"testing"

	"github.com/diamondburned/gotk4/pkg/gtk/v4"

	"github.com/ThatCatDev/tanrenai/shared/pkg/api"
)

func TestMessageWidgetUser(t *testing.T) {
	w := messageWidget("user", "hello")
	if w == nil {
		t.Fatal("messageWidget returned nil")
	}
	if !w.HasCSSClass("user-message") {
		t.Fatal("expected CSS class 'user-message'")
	}
	if w.HAlign() != gtk.AlignEnd {
		t.Fatal("expected user message to align end")
	}
}

func TestMessageWidgetAssistant(t *testing.T) {
	w := messageWidget("assistant", "hi there")
	if w == nil {
		t.Fatal("messageWidget returned nil")
	}
	if !w.HasCSSClass("assistant-message") {
		t.Fatal("expected CSS class 'assistant-message'")
	}
	if w.HAlign() != gtk.AlignStart {
		t.Fatal("expected assistant message to align start")
	}
}

func TestMessageWidgetDefault(t *testing.T) {
	w := messageWidget("system", "system message")
	if w == nil {
		t.Fatal("messageWidget returned nil for unknown role")
	}
	// Default has no role-specific CSS class
	if w.HasCSSClass("user-message") || w.HasCSSClass("assistant-message") {
		t.Fatal("unexpected CSS class for default role")
	}
}

func TestToolCallWidget(t *testing.T) {
	call := api.ToolCall{
		ID:   "tc1",
		Type: "function",
		Function: api.ToolCallFunction{
			Name:      "file_read",
			Arguments: `{"path": "/tmp/test.go"}`,
		},
	}
	w := toolCallWidget(call)
	if w == nil {
		t.Fatal("toolCallWidget returned nil")
	}
	if !w.HasCSSClass("tool-call") {
		t.Fatal("expected CSS class 'tool-call'")
	}

	// Count children: icon, name, arg
	count := 0
	for child := w.FirstChild(); child != nil; child = gtk.BaseWidget(child).NextSibling() {
		count++
	}
	if count != 3 {
		t.Fatalf("expected 3 children (icon, name, arg), got %d", count)
	}
}

func TestToolCallWidgetNoArg(t *testing.T) {
	call := api.ToolCall{
		ID:   "tc2",
		Type: "function",
		Function: api.ToolCallFunction{
			Name:      "custom_tool",
			Arguments: `{"foo": "bar"}`,
		},
	}
	w := toolCallWidget(call)

	// extractKeyArg returns "" for unknown keys, so only 2 children (icon, name)
	count := 0
	for child := w.FirstChild(); child != nil; child = gtk.BaseWidget(child).NextSibling() {
		count++
	}
	if count != 2 {
		t.Fatalf("expected 2 children (icon, name) when no key arg, got %d", count)
	}
}

func TestToolResultWidget(t *testing.T) {
	call := api.ToolCall{
		ID:   "tc1",
		Type: "function",
		Function: api.ToolCallFunction{
			Name:      "file_read",
			Arguments: `{"path": "/tmp/test.go"}`,
		},
	}
	w := toolResultWidget(call, "file contents here\nline 2")
	if w == nil {
		t.Fatal("toolResultWidget returned nil")
	}
	if !w.HasCSSClass("tool-result") {
		t.Fatal("expected CSS class 'tool-result'")
	}
}

func TestThinkingWidget(t *testing.T) {
	w := thinkingWidget()
	if w == nil {
		t.Fatal("thinkingWidget returned nil")
	}
	if !w.HasCSSClass("thinking-indicator") {
		t.Fatal("expected CSS class 'thinking-indicator'")
	}

	// Should have spinner + label = 2 children
	count := 0
	for child := w.FirstChild(); child != nil; child = gtk.BaseWidget(child).NextSibling() {
		count++
	}
	if count != 2 {
		t.Fatalf("expected 2 children (spinner + label), got %d", count)
	}
}

func TestStreamingWidget(t *testing.T) {
	w := streamingWidget()
	if w == nil {
		t.Fatal("streamingWidget returned nil")
	}
	if !w.HasCSSClass("assistant-message") {
		t.Fatal("expected CSS class 'assistant-message'")
	}
	if !w.Selectable() {
		t.Fatal("expected streaming widget to be selectable")
	}
}
