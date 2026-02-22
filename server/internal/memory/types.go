package memory

import (
	"context"
	"time"
)

// Entry represents a single memory entry (a completed user+assistant turn).
type Entry struct {
	ID        string
	UserMsg   string
	AssistMsg string
	Timestamp time.Time
	SessionID string
	Metadata  map[string]string
}

// Content returns the combined text of the entry for embedding and search.
func (e *Entry) Content() string {
	return "User: " + e.UserMsg + "\nAssistant: " + e.AssistMsg
}

// SearchResult is a memory entry with associated scores from hybrid search.
type SearchResult struct {
	Entry         Entry
	SemanticScore float32
	KeywordScore  float32
	CombinedScore float32
}

// Store is the interface for persistent memory storage with hybrid search.
type Store interface {
	Add(ctx context.Context, entry Entry) error
	Search(ctx context.Context, query string, limit int) ([]SearchResult, error)
	List(ctx context.Context, limit int) ([]Entry, error)
	Delete(ctx context.Context, id string) error
	Clear(ctx context.Context) error
	Count() int
	Close() error
}
