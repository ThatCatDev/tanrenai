package ui

import (
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/ThatCatDev/tanrenai/shared/pkg/api"
)

func TestNewSession(t *testing.T) {
	s := NewSession("test-model")
	if s.ID == "" {
		t.Fatal("expected non-empty ID")
	}
	if s.Title != "New Chat" {
		t.Fatalf("expected title 'New Chat', got %q", s.Title)
	}
	if s.Model != "test-model" {
		t.Fatalf("expected model 'test-model', got %q", s.Model)
	}
	if s.CreatedAt.IsZero() || s.UpdatedAt.IsZero() {
		t.Fatal("expected non-zero timestamps")
	}
}

func TestAutoTitle(t *testing.T) {
	s := NewSession("")
	s.Messages = []api.Message{
		{Role: "user", Content: "Hello, how are you today?"},
		{Role: "assistant", Content: "I'm fine, thanks!"},
	}
	s.AutoTitle()
	if s.Title != "Hello, how are you today?" {
		t.Fatalf("expected auto-title from first user message, got %q", s.Title)
	}
}

func TestAutoTitleTruncates(t *testing.T) {
	s := NewSession("")
	long := "This is a very long message that should be truncated because it exceeds fifty characters in length"
	s.Messages = []api.Message{
		{Role: "user", Content: long},
	}
	s.AutoTitle()
	if len(s.Title) > 54 { // 50 + "..."
		t.Fatalf("expected truncated title, got length %d: %q", len(s.Title), s.Title)
	}
	if s.Title[len(s.Title)-3:] != "..." {
		t.Fatalf("expected title to end with '...', got %q", s.Title)
	}
}

func TestAutoTitleSkipsNonUser(t *testing.T) {
	s := NewSession("")
	s.Title = "New Chat"
	s.Messages = []api.Message{
		{Role: "system", Content: "You are a helper"},
		{Role: "assistant", Content: "Hello!"},
	}
	s.AutoTitle()
	if s.Title != "New Chat" {
		t.Fatalf("expected title unchanged with no user messages, got %q", s.Title)
	}
}

func TestSaveLoadSessions(t *testing.T) {
	tmpDir := t.TempDir()
	t.Setenv("TANRENAI_DATA_DIR", tmpDir)

	sessions := []*ChatSession{
		{
			ID:        "1",
			Title:     "First Chat",
			Model:     "model-a",
			CreatedAt: time.Now().Add(-time.Hour),
			UpdatedAt: time.Now().Add(-time.Minute),
			Messages: []api.Message{
				{Role: "user", Content: "hi"},
				{Role: "assistant", Content: "hello"},
			},
		},
		{
			ID:        "2",
			Title:     "Second Chat",
			Model:     "model-b",
			CreatedAt: time.Now().Add(-30 * time.Minute),
			UpdatedAt: time.Now(),
			Messages:  nil,
		},
	}

	if err := SaveSessions(sessions); err != nil {
		t.Fatalf("SaveSessions: %v", err)
	}

	// Verify file exists
	path := filepath.Join(tmpDir, "chats", "sessions.json")
	if _, err := os.Stat(path); err != nil {
		t.Fatalf("sessions file not created: %v", err)
	}

	loaded, err := LoadSessions()
	if err != nil {
		t.Fatalf("LoadSessions: %v", err)
	}

	if len(loaded) != 2 {
		t.Fatalf("expected 2 sessions, got %d", len(loaded))
	}

	// Should be sorted by UpdatedAt descending — session "2" is more recent
	if loaded[0].ID != "2" {
		t.Fatalf("expected most recent session first, got ID %q", loaded[0].ID)
	}
	if loaded[1].ID != "1" {
		t.Fatalf("expected older session second, got ID %q", loaded[1].ID)
	}

	// Verify message content preserved
	if len(loaded[1].Messages) != 2 {
		t.Fatalf("expected 2 messages in session 1, got %d", len(loaded[1].Messages))
	}
	if loaded[1].Messages[0].Content != "hi" {
		t.Fatalf("expected first message 'hi', got %q", loaded[1].Messages[0].Content)
	}
}

func TestLoadSessionsEmpty(t *testing.T) {
	tmpDir := t.TempDir()
	t.Setenv("TANRENAI_DATA_DIR", tmpDir)

	loaded, err := LoadSessions()
	if err != nil {
		t.Fatalf("LoadSessions on missing file: %v", err)
	}
	if loaded != nil {
		t.Fatalf("expected nil sessions, got %d", len(loaded))
	}
}

func TestSaveSessionsCreatesDir(t *testing.T) {
	tmpDir := t.TempDir()
	t.Setenv("TANRENAI_DATA_DIR", filepath.Join(tmpDir, "nested", "deep"))

	err := SaveSessions([]*ChatSession{NewSession("m")})
	if err != nil {
		t.Fatalf("SaveSessions should create dirs: %v", err)
	}
}
