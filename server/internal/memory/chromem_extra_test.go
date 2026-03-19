package memory

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"testing"
	"time"

	chromem "github.com/philippgille/chromem-go"
)

// ---------------------------------------------------------------------------
// NewChromemStore — persistent store (uses a temp dir)
// ---------------------------------------------------------------------------

func newTestPersistentStore(t *testing.T) (*ChromemStore, string) {
	t.Helper()
	dir := t.TempDir()
	store, err := NewChromemStore(dir, deterministicEmbed)
	if err != nil {
		t.Fatalf("NewChromemStore: %v", err)
	}

	return store, dir
}

func TestNewChromemStorePersistent(t *testing.T) {
	store, _ := newTestPersistentStore(t)
	if store == nil {
		t.Fatal("NewChromemStore returned nil")
	}
	if store.persistDir == "" {
		t.Error("expected persistDir to be set on persistent store")
	}
}

func TestNewChromemStorePersistentAddAndReload(t *testing.T) {
	dir := t.TempDir()

	// Create store, add an entry, then close it.
	store, err := NewChromemStore(dir, deterministicEmbed)
	if err != nil {
		t.Fatalf("NewChromemStore: %v", err)
	}
	ctx := context.Background()
	entry := &Entry{UserMsg: "persist me", AssistMsg: "ok"}
	if err := store.Add(ctx, entry); err != nil {
		t.Fatalf("Add: %v", err)
	}
	savedID := entry.ID
	if err := store.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}

	// Reopen the store — loadIndex should restore the entry from the JSON file.
	store2, err := NewChromemStore(dir, deterministicEmbed)
	if err != nil {
		t.Fatalf("NewChromemStore reload: %v", err)
	}

	store2.mu.RLock()
	_, ok := store2.entries[savedID]
	store2.mu.RUnlock()

	if !ok {
		t.Errorf("entry %s not found after reload", savedID)
	}
}

// ---------------------------------------------------------------------------
// NewChromemStoreInMemory — collection creation error via bad embed func
// ---------------------------------------------------------------------------

// errEmbedFunc is a chromem.EmbeddingFunc that always returns an error.
// We can't trigger GetOrCreateCollection to fail easily, so we test that a
// valid in-memory store is created with a nil embed function path as well.
// This test just ensures the happy path sets persistDir correctly.
func TestNewChromemStoreInMemoryPersistDirEmpty(t *testing.T) {
	store, err := NewChromemStoreInMemory(deterministicEmbed)
	if err != nil {
		t.Fatalf("NewChromemStoreInMemory: %v", err)
	}
	if store.persistDir != "" {
		t.Errorf("expected empty persistDir for in-memory store, got %q", store.persistDir)
	}
}

// ---------------------------------------------------------------------------
// saveIndex — persistent store writes JSON file
// ---------------------------------------------------------------------------

func TestSaveIndexWritesFile(t *testing.T) {
	store, dir := newTestPersistentStore(t)
	ctx := context.Background()

	if err := store.Add(ctx, &Entry{UserMsg: "save test", AssistMsg: "yes"}); err != nil {
		t.Fatalf("Add: %v", err)
	}

	indexFile := filepath.Join(dir, "entries_index.json")
	if _, err := os.Stat(indexFile); os.IsNotExist(err) {
		t.Error("saveIndex did not create entries_index.json")
	}
}

func TestSaveIndexInMemoryNoOp(t *testing.T) {
	// For an in-memory store, saveIndex should silently return nil.
	store, err := NewChromemStoreInMemory(deterministicEmbed)
	if err != nil {
		t.Fatalf("NewChromemStoreInMemory: %v", err)
	}
	if err := store.saveIndex(); err != nil {
		t.Errorf("saveIndex on in-memory store returned error: %v", err)
	}
}

// ---------------------------------------------------------------------------
// loadIndex — covers the full load path
// ---------------------------------------------------------------------------

func TestLoadIndexInMemoryNoOp(t *testing.T) {
	store, err := NewChromemStoreInMemory(deterministicEmbed)
	if err != nil {
		t.Fatalf("NewChromemStoreInMemory: %v", err)
	}
	// loadIndex with empty persistDir should return nil immediately.
	if err := store.loadIndex(); err != nil {
		t.Errorf("loadIndex on in-memory store returned error: %v", err)
	}
}

func TestLoadIndexFileNotExist(t *testing.T) {
	dir := t.TempDir()
	store := &ChromemStore{persistDir: dir, entries: make(map[string]Entry)}

	// No index file exists yet — loadIndex should return an error (os.ErrNotExist).
	err := store.loadIndex()
	if err == nil {
		t.Fatal("expected error when index file does not exist, got nil")
	}
	if !errors.Is(err, os.ErrNotExist) {
		t.Errorf("expected os.ErrNotExist, got: %v", err)
	}
}

func TestLoadIndexRoundTrip(t *testing.T) {
	dir := t.TempDir()

	// Write a known entry to the index file by using a store.
	store := &ChromemStore{persistDir: dir, entries: make(map[string]Entry)}
	store.entries["id-1"] = Entry{
		ID:        "id-1",
		UserMsg:   "round trip",
		AssistMsg: "yes",
		Timestamp: time.Date(2024, 1, 15, 10, 0, 0, 0, time.UTC),
	}
	if err := store.saveIndex(); err != nil {
		t.Fatalf("saveIndex: %v", err)
	}

	// Load into a fresh store.
	store2 := &ChromemStore{persistDir: dir, entries: make(map[string]Entry)}
	if err := store2.loadIndex(); err != nil {
		t.Fatalf("loadIndex: %v", err)
	}

	store2.mu.RLock()
	e, ok := store2.entries["id-1"]
	store2.mu.RUnlock()

	if !ok {
		t.Fatal("entry id-1 not found after loadIndex")
	}
	if e.UserMsg != "round trip" {
		t.Errorf("UserMsg = %q, want %q", e.UserMsg, "round trip")
	}
}

// ---------------------------------------------------------------------------
// indexPath — verify both persistent and in-memory paths
// ---------------------------------------------------------------------------

func TestIndexPathPersistent(t *testing.T) {
	store := &ChromemStore{persistDir: "/some/dir"}
	got := store.indexPath()
	want := filepath.Join("/some/dir", "entries_index.json")
	if got != want {
		t.Errorf("indexPath() = %q, want %q", got, want)
	}
}

func TestIndexPathInMemory(t *testing.T) {
	store := &ChromemStore{persistDir: ""}
	got := store.indexPath()
	if got != "" {
		t.Errorf("indexPath() = %q, want empty string for in-memory store", got)
	}
}

// ---------------------------------------------------------------------------
// entryFromResult — fallback path (entry not in s.entries map)
// ---------------------------------------------------------------------------

func TestEntryFromResultFallback(t *testing.T) {
	store, err := NewChromemStoreInMemory(deterministicEmbed)
	if err != nil {
		t.Fatalf("NewChromemStoreInMemory: %v", err)
	}

	// Inject a chromem.Result that is NOT in the entries map to trigger the
	// fallback reconstruction from metadata.
	ts := time.Date(2024, 6, 1, 12, 0, 0, 0, time.UTC)
	result := fakeResult{
		id:      "missing-id",
		content: "User: test\nAssistant: yes",
		metadata: map[string]string{
			"user_msg":   "test",
			"assist_msg": "yes",
			"timestamp":  ts.Format(time.RFC3339),
			"session_id": "sess-1",
		},
	}

	entry := store.entryFromResult(result.toChromemResult())

	if entry.ID != "missing-id" {
		t.Errorf("ID = %q, want missing-id", entry.ID)
	}
	if entry.UserMsg != "test" {
		t.Errorf("UserMsg = %q, want test", entry.UserMsg)
	}
	if entry.AssistMsg != "yes" {
		t.Errorf("AssistMsg = %q, want yes", entry.AssistMsg)
	}
	if entry.SessionID != "sess-1" {
		t.Errorf("SessionID = %q, want sess-1", entry.SessionID)
	}
	if !entry.Timestamp.Equal(ts) {
		t.Errorf("Timestamp = %v, want %v", entry.Timestamp, ts)
	}
}

// ---------------------------------------------------------------------------
// entryFromResult — cache hit path (entry IS in the map)
// ---------------------------------------------------------------------------

func TestEntryFromResultCacheHit(t *testing.T) {
	store, err := NewChromemStoreInMemory(deterministicEmbed)
	if err != nil {
		t.Fatalf("NewChromemStoreInMemory: %v", err)
	}

	cached := Entry{
		ID:        "cache-id",
		UserMsg:   "cached user",
		AssistMsg: "cached assist",
		Timestamp: time.Now(),
		SessionID: "sess-cache",
	}
	store.mu.Lock()
	store.entries["cache-id"] = cached
	store.mu.Unlock()

	result := fakeResult{
		id:       "cache-id",
		content:  "irrelevant",
		metadata: map[string]string{},
	}

	entry := store.entryFromResult(result.toChromemResult())
	if entry.UserMsg != cached.UserMsg {
		t.Errorf("UserMsg = %q, want %q", entry.UserMsg, cached.UserMsg)
	}
}

// ---------------------------------------------------------------------------
// Add — with existing ID and Timestamp (no auto-generation)
// ---------------------------------------------------------------------------

func TestAddPresetIDAndTimestamp(t *testing.T) {
	store := newTestStore(t)
	ctx := context.Background()

	fixedTime := time.Date(2024, 3, 1, 9, 0, 0, 0, time.UTC)
	entry := &Entry{
		ID:        "preset-id",
		UserMsg:   "preset",
		AssistMsg: "yes",
		Timestamp: fixedTime,
	}
	if err := store.Add(ctx, entry); err != nil {
		t.Fatalf("Add: %v", err)
	}
	if entry.ID != "preset-id" {
		t.Errorf("ID changed: got %q", entry.ID)
	}
	if !entry.Timestamp.Equal(fixedTime) {
		t.Errorf("Timestamp changed: got %v", entry.Timestamp)
	}
}

// ---------------------------------------------------------------------------
// Delete — persistent store triggers saveIndex
// ---------------------------------------------------------------------------

func TestDeletePersistentSavesIndex(t *testing.T) {
	store, dir := newTestPersistentStore(t)
	ctx := context.Background()

	entry := &Entry{UserMsg: "to delete", AssistMsg: "ok"}
	if err := store.Add(ctx, entry); err != nil {
		t.Fatalf("Add: %v", err)
	}
	if err := store.Delete(ctx, entry.ID); err != nil {
		t.Fatalf("Delete: %v", err)
	}

	// Reload and verify the entry is gone from the index file.
	store2 := &ChromemStore{persistDir: dir, entries: make(map[string]Entry)}
	if err := store2.loadIndex(); err != nil && !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("loadIndex after delete: %v", err)
	}
	store2.mu.RLock()
	_, ok := store2.entries[entry.ID]
	store2.mu.RUnlock()
	if ok {
		t.Error("entry should not be in index after Delete + reload")
	}
}

// ---------------------------------------------------------------------------
// Clear — persistent store triggers saveIndex
// ---------------------------------------------------------------------------

func TestClearPersistentSavesIndex(t *testing.T) {
	store, dir := newTestPersistentStore(t)
	ctx := context.Background()

	for i := 0; i < 3; i++ {
		if err := store.Add(ctx, &Entry{UserMsg: "msg", AssistMsg: "reply"}); err != nil {
			t.Fatalf("Add: %v", err)
		}
	}
	if err := store.Clear(ctx); err != nil {
		t.Fatalf("Clear: %v", err)
	}

	// Reload: the index file should exist but contain an empty map.
	store2 := &ChromemStore{persistDir: dir, entries: make(map[string]Entry)}
	if err := store2.loadIndex(); err != nil {
		t.Fatalf("loadIndex after clear: %v", err)
	}
	store2.mu.RLock()
	count := len(store2.entries)
	store2.mu.RUnlock()
	if count != 0 {
		t.Errorf("expected 0 entries after Clear + reload, got %d", count)
	}
}

// ---------------------------------------------------------------------------
// Clear — empty store (no IDs to delete, still calls saveIndex)
// ---------------------------------------------------------------------------

func TestClearEmptyStore(t *testing.T) {
	store, _ := newTestPersistentStore(t)
	ctx := context.Background()

	// Clear an already-empty persistent store.
	if err := store.Clear(ctx); err != nil {
		t.Fatalf("Clear on empty store: %v", err)
	}
}

// ---------------------------------------------------------------------------
// Search — limit greater than count clamps to count
// ---------------------------------------------------------------------------

func TestSearchLimitGreaterThanCount(t *testing.T) {
	store := newTestStore(t)
	ctx := context.Background()

	if err := store.Add(ctx, &Entry{UserMsg: "only entry", AssistMsg: "yes"}); err != nil {
		t.Fatalf("Add: %v", err)
	}

	// Request limit=10 but there is only 1 document.
	results, err := store.Search(ctx, "only entry", 10)
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(results) != 1 {
		t.Errorf("expected 1 result, got %d", len(results))
	}
}

// ---------------------------------------------------------------------------
// Close — persistent store (still returns nil)
// ---------------------------------------------------------------------------

func TestClosePersistentStore(t *testing.T) {
	store, _ := newTestPersistentStore(t)
	if err := store.Close(); err != nil {
		t.Errorf("Close() on persistent store returned error: %v", err)
	}
}

// ---------------------------------------------------------------------------
// Helper: fakeResult wraps the chromem.Result fields we need.
// chromem.Result is a struct with exported fields, so we can construct it directly.
// ---------------------------------------------------------------------------

type fakeResult struct {
	id       string
	content  string
	metadata map[string]string
}

func (f fakeResult) toChromemResult() chromem.Result {
	return chromem.Result{
		ID:         f.id,
		Content:    f.content,
		Metadata:   f.metadata,
		Similarity: 0.9,
	}
}
