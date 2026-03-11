package ui

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"time"

	"github.com/ThatCatDev/tanrenai/shared/pkg/api"
)

// ChatSession represents a single conversation.
type ChatSession struct {
	ID        string        `json:"id"`
	Title     string        `json:"title"`
	Messages  []api.Message `json:"messages"`
	Model     string        `json:"model"`
	CreatedAt time.Time     `json:"created_at"`
	UpdatedAt time.Time     `json:"updated_at"`
}

// NewSession creates a new chat session with a unique ID.
func NewSession(model string) *ChatSession {
	now := time.Now()
	return &ChatSession{
		ID:        fmt.Sprintf("%d", now.UnixNano()),
		Title:     "New Chat",
		Model:     model,
		CreatedAt: now,
		UpdatedAt: now,
	}
}

// AutoTitle sets the title from the first user message (up to 50 chars).
func (s *ChatSession) AutoTitle() {
	for _, m := range s.Messages {
		if m.Role == "user" && m.Content != "" {
			title := m.Content
			if len(title) > 50 {
				title = title[:50] + "..."
			}
			s.Title = title
			return
		}
	}
}

// sessionsDir returns the directory for session storage.
func sessionsDir() string {
	dataDir := os.Getenv("TANRENAI_DATA_DIR")
	if dataDir == "" {
		home, _ := os.UserHomeDir()
		dataDir = filepath.Join(home, ".local", "share", "tanrenai")
	}
	return filepath.Join(dataDir, "chats")
}

// sessionsPath returns the full path to the sessions JSON file.
func sessionsPath() string {
	return filepath.Join(sessionsDir(), "sessions.json")
}

// SaveSessions persists all sessions to disk.
func SaveSessions(sessions []*ChatSession) error {
	dir := sessionsDir()
	if err := os.MkdirAll(dir, 0755); err != nil {
		return err
	}
	data, err := json.MarshalIndent(sessions, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(sessionsPath(), data, 0644)
}

// LoadSessions reads sessions from disk.
func LoadSessions() ([]*ChatSession, error) {
	data, err := os.ReadFile(sessionsPath())
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, err
	}
	var sessions []*ChatSession
	if err := json.Unmarshal(data, &sessions); err != nil {
		return nil, err
	}
	// Sort by UpdatedAt descending
	sort.Slice(sessions, func(i, j int) bool {
		return sessions[i].UpdatedAt.After(sessions[j].UpdatedAt)
	})
	return sessions, nil
}
