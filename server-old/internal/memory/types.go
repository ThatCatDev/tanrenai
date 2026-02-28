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
// The result is capped to keep it within typical embedding model context limits.
func (e *Entry) Content() string {
	content := "User: " + e.UserMsg + "\nAssistant: " + e.AssistMsg
	// Cap at ~1200 chars (~400 tokens at 3.0 c/t) to stay safely within
	// small embedding models (512-token context like MiniLM).
	if len(content) > 1200 {
		content = content[:1200]
	}
	return content
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
