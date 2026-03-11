package ui

import (
	"testing"

	"github.com/diamondburned/gotk4/pkg/gtk/v4"

	"github.com/ThatCatDev/tanrenai/shared/pkg/api"
)

func TestBuildChat(t *testing.T) {
	a := newTestApp()
	// buildChat was already called by newTestApp; verify the returned widget type
	// by calling it again
	tv := a.buildChat()
	if tv == nil {
		t.Fatal("buildChat returned nil")
	}
}

func TestUpdateChatHeader(t *testing.T) {
	a := newTestApp()
	a.activeSession.Title = "My Chat"
	a.updateChatHeader()

	if got := a.chatTitle.Label(); got != "My Chat" {
		t.Fatalf("expected chat title 'My Chat', got %q", got)
	}
}

func TestUpdateModelBadgeVisible(t *testing.T) {
	a := newTestApp()
	a.activeSession.Model = "llama-7b"
	a.updateModelBadge()

	if !a.modelBadge.Visible() {
		t.Fatal("expected model badge to be visible")
	}
	if got := a.modelBadge.Label(); got != "llama-7b" {
		t.Fatalf("expected badge 'llama-7b', got %q", got)
	}
}

func TestUpdateModelBadgeHidden(t *testing.T) {
	a := newTestApp()
	a.activeSession.Model = ""
	a.selectedModel = ""
	a.updateModelBadge()

	if a.modelBadge.Visible() {
		t.Fatal("expected model badge to be hidden when no model set")
	}
}

func TestFinishGenerating(t *testing.T) {
	a := newTestApp()

	// Simulate generating state
	a.generating = true
	a.sendButton.SetIconName("media-playback-stop-symbolic")
	a.sendButton.RemoveCSSClass("send-button")
	a.sendButton.AddCSSClass("stop-button")

	a.finishGenerating()

	if a.generating {
		t.Fatal("expected generating to be false")
	}
	if a.sendButton.IconName() != "go-up-symbolic" {
		t.Fatalf("expected icon 'go-up-symbolic', got %q", a.sendButton.IconName())
	}
	if a.cancelGenerate != nil {
		t.Fatal("expected cancelGenerate to be nil")
	}
}

func TestClearChat(t *testing.T) {
	a := newTestApp()

	// Add some messages
	a.messageList.Append(gtk.NewLabel("msg1"))
	a.messageList.Append(gtk.NewLabel("msg2"))
	a.activeSession.Messages = []api.Message{
		{Role: "user", Content: "hi"},
		{Role: "assistant", Content: "hello"},
	}
	a.activeSession.Title = "Some Title"

	a.clearChat()

	if a.messageList.FirstChild() != nil {
		t.Fatal("expected messageList to be empty after clear")
	}
	if a.activeSession.Messages != nil {
		t.Fatal("expected session messages to be nil after clear")
	}
	if a.activeSession.Title != "New Chat" {
		t.Fatalf("expected title reset to 'New Chat', got %q", a.activeSession.Title)
	}
}

func TestLoadSessionMessages(t *testing.T) {
	a := newTestApp()

	session := NewSession("test-model")
	session.Messages = []api.Message{
		{Role: "user", Content: "hello"},
		{Role: "assistant", Content: "hi there"},
		{Role: "user", Content: "how are you?"},
		{Role: "assistant", Content: "great!"},
	}

	a.loadSessionMessages(session)

	count := countChildren(a.messageList)
	if count != 4 {
		t.Fatalf("expected 4 message widgets, got %d", count)
	}
}

func TestLoadSessionMessagesWithToolCalls(t *testing.T) {
	a := newTestApp()

	session := NewSession("test-model")
	session.Messages = []api.Message{
		{Role: "user", Content: "read my file"},
		{
			Role: "assistant",
			ToolCalls: []api.ToolCall{
				{
					ID:   "tc1",
					Type: "function",
					Function: api.ToolCallFunction{
						Name:      "file_read",
						Arguments: `{"path": "/tmp/test.txt"}`,
					},
				},
			},
		},
		{Role: "tool", Content: "file contents here"},
		{Role: "assistant", Content: "Here's the file content."},
	}

	a.loadSessionMessages(session)

	// user message + tool call + assistant message = 3 widgets
	// (tool role messages are skipped, assistant with no content but tool calls shows tool call widget)
	count := countChildren(a.messageList)
	if count != 3 {
		t.Fatalf("expected 3 widgets (user + tool_call + assistant), got %d", count)
	}
}

func TestLoadSessionMessagesEmpty(t *testing.T) {
	a := newTestApp()

	// Pre-populate with some content
	a.messageList.Append(gtk.NewLabel("old"))

	session := NewSession("test-model")
	a.loadSessionMessages(session)

	if a.messageList.FirstChild() != nil {
		t.Fatal("expected empty messageList after loading empty session")
	}
}
