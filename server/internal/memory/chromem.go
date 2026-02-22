package memory

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/google/uuid"
	"github.com/philippgille/chromem-go"
)

// ChromemStore implements Store using chromem-go for vector storage with hybrid search.
type ChromemStore struct {
	db         *chromem.DB
	collection *chromem.Collection
	entries    map[string]Entry
	mu         sync.RWMutex
	persistDir string // empty for in-memory
}

// NewChromemStore creates a persistent ChromemStore backed by chromem-go.
func NewChromemStore(persistDir string, embedFunc EmbedFunc) (*ChromemStore, error) {
	db, err := chromem.NewPersistentDB(persistDir, false)
	if err != nil {
		return nil, fmt.Errorf("create persistent DB: %w", err)
	}

	col, err := db.GetOrCreateCollection("memories", nil, chromem.EmbeddingFunc(embedFunc))
	if err != nil {
		return nil, fmt.Errorf("get or create collection: %w", err)
	}

	s := &ChromemStore{
		db:         db,
		collection: col,
		entries:    make(map[string]Entry),
		persistDir: persistDir,
	}

	// Load entry index from disk
	if err := s.loadIndex(); err != nil {
		// Not fatal — index may not exist yet
		_ = err
	}

	return s, nil
}

// NewChromemStoreInMemory creates an in-memory ChromemStore for testing.
func NewChromemStoreInMemory(embedFunc EmbedFunc) (*ChromemStore, error) {
	db := chromem.NewDB()
	col, err := db.GetOrCreateCollection("memories", nil, chromem.EmbeddingFunc(embedFunc))
	if err != nil {
		return nil, fmt.Errorf("get or create collection: %w", err)
	}

	return &ChromemStore{
		db:         db,
		collection: col,
		entries:    make(map[string]Entry),
	}, nil
}

func (s *ChromemStore) Add(ctx context.Context, entry Entry) error {
	if entry.ID == "" {
		entry.ID = uuid.New().String()
	}
	if entry.Timestamp.IsZero() {
		entry.Timestamp = time.Now()
	}

	doc := chromem.Document{
		ID:      entry.ID,
		Content: entry.Content(),
		Metadata: map[string]string{
			"user_msg":   entry.UserMsg,
			"assist_msg": entry.AssistMsg,
			"timestamp":  entry.Timestamp.Format(time.RFC3339),
			"session_id": entry.SessionID,
		},
	}

	if err := s.collection.AddDocument(ctx, doc); err != nil {
		return fmt.Errorf("add document: %w", err)
	}

	s.mu.Lock()
	s.entries[entry.ID] = entry
	s.mu.Unlock()

	s.saveIndex()
	return nil
}

func (s *ChromemStore) Search(ctx context.Context, query string, limit int) ([]SearchResult, error) {
	if limit <= 0 {
		limit = 5
	}

	count := s.collection.Count()
	if count == 0 {
		return nil, nil
	}

	// Query up to limit results (but not more than count)
	nResults := limit
	if nResults > count {
		nResults = count
	}

	results, err := s.collection.Query(ctx, query, nResults, nil, nil)
	if err != nil {
		return nil, fmt.Errorf("query collection: %w", err)
	}

	// Compute hybrid scores: 70% semantic + 30% keyword
	queryWords := extractWords(query)
	searchResults := make([]SearchResult, 0, len(results))

	for _, r := range results {
		entry := s.entryFromResult(r)
		kwScore := keywordScore(queryWords, r.Content)

		sr := SearchResult{
			Entry:         entry,
			SemanticScore: r.Similarity,
			KeywordScore:  kwScore,
			CombinedScore: 0.7*r.Similarity + 0.3*kwScore,
		}
		searchResults = append(searchResults, sr)
	}

	// Sort by combined score descending
	sort.Slice(searchResults, func(i, j int) bool {
		return searchResults[i].CombinedScore > searchResults[j].CombinedScore
	})

	return searchResults, nil
}

func (s *ChromemStore) List(ctx context.Context, limit int) ([]Entry, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	entries := make([]Entry, 0, len(s.entries))
	for _, e := range s.entries {
		entries = append(entries, e)
	}

	// Sort by timestamp descending (newest first)
	sort.Slice(entries, func(i, j int) bool {
		return entries[i].Timestamp.After(entries[j].Timestamp)
	})

	if limit > 0 && len(entries) > limit {
		entries = entries[:limit]
	}

	return entries, nil
}

func (s *ChromemStore) Delete(ctx context.Context, id string) error {
	if err := s.collection.Delete(ctx, nil, nil, id); err != nil {
		return fmt.Errorf("delete document: %w", err)
	}

	s.mu.Lock()
	delete(s.entries, id)
	s.mu.Unlock()

	s.saveIndex()
	return nil
}

func (s *ChromemStore) Clear(ctx context.Context) error {
	s.mu.Lock()
	ids := make([]string, 0, len(s.entries))
	for id := range s.entries {
		ids = append(ids, id)
	}
	s.entries = make(map[string]Entry)
	s.mu.Unlock()

	if len(ids) > 0 {
		if err := s.collection.Delete(ctx, nil, nil, ids...); err != nil {
			return fmt.Errorf("clear collection: %w", err)
		}
	}

	s.saveIndex()
	return nil
}

func (s *ChromemStore) Count() int {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return len(s.entries)
}

func (s *ChromemStore) Close() error {
	return nil
}

// entryFromResult reconstructs an Entry from a chromem-go Result.
func (s *ChromemStore) entryFromResult(r chromem.Result) Entry {
	s.mu.RLock()
	if e, ok := s.entries[r.ID]; ok {
		s.mu.RUnlock()
		return e
	}
	s.mu.RUnlock()

	// Fallback: reconstruct from metadata
	ts, _ := time.Parse(time.RFC3339, r.Metadata["timestamp"])
	return Entry{
		ID:        r.ID,
		UserMsg:   r.Metadata["user_msg"],
		AssistMsg: r.Metadata["assist_msg"],
		Timestamp: ts,
		SessionID: r.Metadata["session_id"],
	}
}

// Index persistence — simple JSON file alongside chromem data.

func (s *ChromemStore) indexPath() string {
	if s.persistDir == "" {
		return ""
	}
	return filepath.Join(s.persistDir, "entries_index.json")
}

func (s *ChromemStore) saveIndex() {
	path := s.indexPath()
	if path == "" {
		return
	}

	s.mu.RLock()
	data, err := json.Marshal(s.entries)
	s.mu.RUnlock()

	if err != nil {
		return
	}
	os.WriteFile(path, data, 0644)
}

func (s *ChromemStore) loadIndex() error {
	path := s.indexPath()
	if path == "" {
		return nil
	}

	data, err := os.ReadFile(path)
	if err != nil {
		return err
	}

	s.mu.Lock()
	defer s.mu.Unlock()
	return json.Unmarshal(data, &s.entries)
}

// Keyword scoring helpers.

// extractWords returns lowercased words from text with length >= 3.
func extractWords(text string) []string {
	fields := strings.Fields(strings.ToLower(text))
	words := make([]string, 0, len(fields))
	for _, w := range fields {
		if len(w) >= 3 {
			words = append(words, w)
		}
	}
	return words
}

// keywordScore computes the fraction of query words found in the content.
func keywordScore(queryWords []string, content string) float32 {
	if len(queryWords) == 0 {
		return 0
	}
	lower := strings.ToLower(content)
	matches := 0
	for _, w := range queryWords {
		if strings.Contains(lower, w) {
			matches++
		}
	}
	return float32(matches) / float32(len(queryWords))
}
