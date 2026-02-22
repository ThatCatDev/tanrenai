package memory

import (
	"context"
	"encoding/json"
	"hash/fnv"
	"math"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"testing"
	"time"
)

// mockEmbedFunc creates deterministic embedding vectors from text hashing.
// Produces a 64-dimensional unit vector based on FNV hash.
func mockEmbedFunc(ctx context.Context, text string) ([]float32, error) {
	const dims = 64
	vec := make([]float32, dims)
	h := fnv.New64a()
	h.Write([]byte(text))
	seed := h.Sum64()

	for i := range vec {
		// Use bits of the hash to generate pseudo-random components
		bits := seed ^ (uint64(i) * 0x9E3779B97F4A7C15)
		vec[i] = float32(bits%1000) / 1000.0
	}

	normalizeVector(vec)
	return vec, nil
}

func TestEntryContent(t *testing.T) {
	e := Entry{
		UserMsg:   "hello",
		AssistMsg: "world",
	}
	expected := "User: hello\nAssistant: world"
	if got := e.Content(); got != expected {
		t.Errorf("Content() = %q, want %q", got, expected)
	}
}

func TestNormalizeVector(t *testing.T) {
	v := []float32{3, 4}
	normalizeVector(v)

	// Should be unit length
	var sum float64
	for _, x := range v {
		sum += float64(x) * float64(x)
	}
	norm := math.Sqrt(sum)
	if math.Abs(norm-1.0) > 1e-5 {
		t.Errorf("normalizeVector: norm = %f, want 1.0", norm)
	}

	// Check values: 3/5 and 4/5
	if math.Abs(float64(v[0])-0.6) > 1e-5 || math.Abs(float64(v[1])-0.8) > 1e-5 {
		t.Errorf("normalizeVector: got [%f, %f], want [0.6, 0.8]", v[0], v[1])
	}
}

func TestNormalizeZeroVector(t *testing.T) {
	v := []float32{0, 0, 0}
	normalizeVector(v) // Should not panic
	for _, x := range v {
		if x != 0 {
			t.Errorf("normalizeVector of zero vector: got %f, want 0", x)
		}
	}
}

func TestAddAndSearch(t *testing.T) {
	store, err := NewChromemStoreInMemory(mockEmbedFunc)
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()

	ctx := context.Background()

	entries := []Entry{
		{UserMsg: "how do I compile Go code", AssistMsg: "use go build ./..."},
		{UserMsg: "what is the capital of France", AssistMsg: "Paris is the capital of France"},
		{UserMsg: "explain goroutines in Go", AssistMsg: "goroutines are lightweight threads managed by the Go runtime"},
	}

	for _, e := range entries {
		e.Timestamp = time.Now()
		if err := store.Add(ctx, e); err != nil {
			t.Fatalf("Add failed: %v", err)
		}
	}

	if got := store.Count(); got != 3 {
		t.Fatalf("Count() = %d, want 3", got)
	}

	// Search should return results
	results, err := store.Search(ctx, "Go programming", 3)
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}
	if len(results) == 0 {
		t.Fatal("Search returned no results")
	}
	if len(results) > 3 {
		t.Fatalf("Search returned %d results, want <= 3", len(results))
	}
}

func TestKeywordScoring(t *testing.T) {
	store, err := NewChromemStoreInMemory(mockEmbedFunc)
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()

	ctx := context.Background()

	// Add entries with distinct keywords
	e1 := Entry{UserMsg: "kubernetes deployment yaml", AssistMsg: "use kubectl apply -f deployment.yaml", Timestamp: time.Now()}
	e2 := Entry{UserMsg: "hello world program", AssistMsg: "print hello world", Timestamp: time.Now()}

	store.Add(ctx, e1)
	store.Add(ctx, e2)

	results, err := store.Search(ctx, "kubernetes deployment", 2)
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}

	if len(results) < 2 {
		t.Fatalf("expected 2 results, got %d", len(results))
	}

	// The entry with "kubernetes" should have a higher keyword score
	for _, r := range results {
		if r.Entry.UserMsg == "kubernetes deployment yaml" {
			if r.KeywordScore == 0 {
				t.Error("expected non-zero keyword score for matching entry")
			}
		}
	}
}

func TestExtractWords(t *testing.T) {
	words := extractWords("Go is a great language for backend")
	// "Go" and "is" and "a" are < 3 chars, should be excluded
	for _, w := range words {
		if len(w) < 3 {
			t.Errorf("extractWords returned word %q with len < 3", w)
		}
	}
	// "great", "language", "for", "backend" should be present (for is 3 chars)
	found := map[string]bool{}
	for _, w := range words {
		found[w] = true
	}
	if !found["great"] || !found["language"] || !found["backend"] {
		t.Errorf("extractWords missing expected words, got %v", words)
	}
}

func TestKeywordScore(t *testing.T) {
	words := []string{"kubernetes", "deployment"}
	score := keywordScore(words, "kubernetes deployment yaml file")
	if score != 1.0 {
		t.Errorf("keywordScore = %f, want 1.0 (all words match)", score)
	}

	score = keywordScore(words, "hello world program")
	if score != 0.0 {
		t.Errorf("keywordScore = %f, want 0.0 (no words match)", score)
	}

	score = keywordScore(words, "kubernetes cluster setup")
	if score != 0.5 {
		t.Errorf("keywordScore = %f, want 0.5 (1 of 2 words match)", score)
	}

	// Empty query words
	score = keywordScore(nil, "anything")
	if score != 0.0 {
		t.Errorf("keywordScore with nil words = %f, want 0.0", score)
	}
}

func TestDeleteAndClear(t *testing.T) {
	store, err := NewChromemStoreInMemory(mockEmbedFunc)
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()

	ctx := context.Background()

	e1 := Entry{ID: "test-id-1", UserMsg: "first", AssistMsg: "response1", Timestamp: time.Now()}
	e2 := Entry{ID: "test-id-2", UserMsg: "second", AssistMsg: "response2", Timestamp: time.Now()}

	store.Add(ctx, e1)
	store.Add(ctx, e2)

	if store.Count() != 2 {
		t.Fatalf("Count() = %d, want 2", store.Count())
	}

	// Delete one
	if err := store.Delete(ctx, "test-id-1"); err != nil {
		t.Fatalf("Delete failed: %v", err)
	}
	if store.Count() != 1 {
		t.Fatalf("Count() after delete = %d, want 1", store.Count())
	}

	// Clear all
	if err := store.Clear(ctx); err != nil {
		t.Fatalf("Clear failed: %v", err)
	}
	if store.Count() != 0 {
		t.Fatalf("Count() after clear = %d, want 0", store.Count())
	}
}

func TestListOrdering(t *testing.T) {
	store, err := NewChromemStoreInMemory(mockEmbedFunc)
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()

	ctx := context.Background()

	now := time.Now()
	entries := []Entry{
		{UserMsg: "oldest", AssistMsg: "old reply", Timestamp: now.Add(-2 * time.Hour)},
		{UserMsg: "newest", AssistMsg: "new reply", Timestamp: now},
		{UserMsg: "middle", AssistMsg: "mid reply", Timestamp: now.Add(-1 * time.Hour)},
	}

	for _, e := range entries {
		store.Add(ctx, e)
	}

	listed, err := store.List(ctx, 0)
	if err != nil {
		t.Fatalf("List failed: %v", err)
	}

	if len(listed) != 3 {
		t.Fatalf("List returned %d entries, want 3", len(listed))
	}

	// Should be sorted by timestamp descending (newest first)
	if listed[0].UserMsg != "newest" {
		t.Errorf("first entry should be 'newest', got %q", listed[0].UserMsg)
	}
	if listed[1].UserMsg != "middle" {
		t.Errorf("second entry should be 'middle', got %q", listed[1].UserMsg)
	}
	if listed[2].UserMsg != "oldest" {
		t.Errorf("third entry should be 'oldest', got %q", listed[2].UserMsg)
	}

	// Test with limit
	limited, err := store.List(ctx, 2)
	if err != nil {
		t.Fatalf("List with limit failed: %v", err)
	}
	if len(limited) != 2 {
		t.Fatalf("List with limit=2 returned %d entries, want 2", len(limited))
	}
}

func TestSearchEmpty(t *testing.T) {
	store, err := NewChromemStoreInMemory(mockEmbedFunc)
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()

	results, err := store.Search(context.Background(), "anything", 5)
	if err != nil {
		t.Fatalf("Search on empty store failed: %v", err)
	}
	if len(results) != 0 {
		t.Fatalf("Search on empty store returned %d results, want 0", len(results))
	}
}

func TestPersistentStore(t *testing.T) {
	dir := t.TempDir()
	persistDir := filepath.Join(dir, "chromem")

	ctx := context.Background()

	// Create store and add entries
	store, err := NewChromemStore(persistDir, mockEmbedFunc)
	if err != nil {
		t.Fatal(err)
	}

	e := Entry{
		UserMsg:   "persistent test",
		AssistMsg: "this should survive restart",
		Timestamp: time.Now(),
		SessionID: "session-1",
	}
	if err := store.Add(ctx, e); err != nil {
		t.Fatalf("Add failed: %v", err)
	}

	if store.Count() != 1 {
		t.Fatalf("Count() = %d, want 1", store.Count())
	}

	store.Close()

	// Re-open and verify index was persisted
	store2, err := NewChromemStore(persistDir, mockEmbedFunc)
	if err != nil {
		t.Fatal(err)
	}
	defer store2.Close()

	if store2.Count() != 1 {
		t.Fatalf("Count() after reopen = %d, want 1", store2.Count())
	}

	entries, err := store2.List(ctx, 0)
	if err != nil {
		t.Fatalf("List after reopen failed: %v", err)
	}
	if len(entries) != 1 {
		t.Fatalf("List returned %d entries, want 1", len(entries))
	}
	if entries[0].UserMsg != "persistent test" {
		t.Errorf("UserMsg = %q, want %q", entries[0].UserMsg, "persistent test")
	}
	if entries[0].SessionID != "session-1" {
		t.Errorf("SessionID = %q, want %q", entries[0].SessionID, "session-1")
	}
}

func TestNewLlamaEmbedFunc(t *testing.T) {
	// Mock embedding server returning OpenAI-compatible response
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/embeddings" {
			t.Errorf("unexpected path: %s", r.URL.Path)
			http.Error(w, "not found", http.StatusNotFound)
			return
		}

		var req embeddingRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}

		// Return a simple embedding
		resp := embeddingResponse{
			Data: []embeddingData{
				{
					Embedding: []float32{0.1, 0.2, 0.3, 0.4},
					Index:     0,
				},
			},
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	embedFunc := NewLlamaEmbedFunc(server.URL, 0)
	vec, err := embedFunc(context.Background(), "test text")
	if err != nil {
		t.Fatalf("NewLlamaEmbedFunc failed: %v", err)
	}

	if len(vec) != 4 {
		t.Fatalf("expected 4-dimensional vector, got %d", len(vec))
	}

	// Verify normalization (should be unit length)
	var sum float64
	for _, x := range vec {
		sum += float64(x) * float64(x)
	}
	norm := math.Sqrt(sum)
	if math.Abs(norm-1.0) > 1e-5 {
		t.Errorf("vector not normalized: norm = %f", norm)
	}
}

func TestNewLlamaEmbedFuncErrors(t *testing.T) {
	// Test server returning error
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "internal error", http.StatusInternalServerError)
	}))
	defer server.Close()

	embedFunc := NewLlamaEmbedFunc(server.URL, 0)
	_, err := embedFunc(context.Background(), "test")
	if err == nil {
		t.Fatal("expected error for 500 response")
	}

	// Test empty data response
	server2 := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		resp := embeddingResponse{Data: []embeddingData{}}
		json.NewEncoder(w).Encode(resp)
	}))
	defer server2.Close()

	embedFunc2 := NewLlamaEmbedFunc(server2.URL, 0)
	_, err = embedFunc2(context.Background(), "test")
	if err == nil {
		t.Fatal("expected error for empty data response")
	}
}

func TestEmbeddingRunnerHealthCheck(t *testing.T) {
	// Mock a healthy server
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/health" {
			w.WriteHeader(http.StatusOK)
			return
		}
		http.Error(w, "not found", http.StatusNotFound)
	}))
	defer server.Close()

	r := &EmbeddingRunner{
		baseURL: server.URL,
	}

	if err := r.healthCheck(context.Background()); err != nil {
		t.Errorf("healthCheck failed on healthy server: %v", err)
	}

	if r.BaseURL() != server.URL {
		t.Errorf("BaseURL() = %q, want %q", r.BaseURL(), server.URL)
	}
}

func TestEmbeddingRunnerHealthCheckUnhealthy(t *testing.T) {
	// Mock an unhealthy server
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusServiceUnavailable)
	}))
	defer server.Close()

	r := &EmbeddingRunner{
		baseURL: server.URL,
	}

	if err := r.healthCheck(context.Background()); err == nil {
		t.Error("expected error for unhealthy server")
	}
}

func TestEmbeddingRunnerClose(t *testing.T) {
	// Close on nil process should not panic
	r := &EmbeddingRunner{}
	if err := r.Close(); err != nil {
		t.Errorf("Close on nil runner: %v", err)
	}
}

func TestPersistentStoreIndexFile(t *testing.T) {
	dir := t.TempDir()
	persistDir := filepath.Join(dir, "chromem")

	store, err := NewChromemStore(persistDir, mockEmbedFunc)
	if err != nil {
		t.Fatal(err)
	}

	ctx := context.Background()
	store.Add(ctx, Entry{
		UserMsg:   "indexed entry",
		AssistMsg: "response",
		Timestamp: time.Now(),
	})
	store.Close()

	// Verify index file exists
	indexPath := filepath.Join(persistDir, "entries_index.json")
	if _, err := os.Stat(indexPath); os.IsNotExist(err) {
		t.Fatal("entries_index.json not created")
	}

	// Verify it's valid JSON
	data, err := os.ReadFile(indexPath)
	if err != nil {
		t.Fatal(err)
	}
	var entries map[string]Entry
	if err := json.Unmarshal(data, &entries); err != nil {
		t.Fatalf("index file is not valid JSON: %v", err)
	}
	if len(entries) != 1 {
		t.Fatalf("index has %d entries, want 1", len(entries))
	}
}

func TestEntryFromResultFallback(t *testing.T) {
	// Test the fallback path where entry is not in the local map
	store, err := NewChromemStoreInMemory(mockEmbedFunc)
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()

	ctx := context.Background()

	// Add an entry, then remove it from the local map to test fallback
	e := Entry{
		ID:        "fallback-test",
		UserMsg:   "fallback user",
		AssistMsg: "fallback assist",
		Timestamp: time.Now(),
		SessionID: "sess-1",
	}
	store.Add(ctx, e)

	// Remove from local entries map to force fallback
	store.mu.Lock()
	delete(store.entries, "fallback-test")
	store.mu.Unlock()

	// Search should still work using metadata fallback
	results, err := store.Search(ctx, "fallback", 1)
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}
	if len(results) == 0 {
		t.Fatal("expected search results")
	}
	if results[0].Entry.UserMsg != "fallback user" {
		t.Errorf("fallback UserMsg = %q, want %q", results[0].Entry.UserMsg, "fallback user")
	}
}

func TestClearEmptyStore(t *testing.T) {
	store, err := NewChromemStoreInMemory(mockEmbedFunc)
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()

	// Clear on empty store should not error
	if err := store.Clear(context.Background()); err != nil {
		t.Fatalf("Clear on empty store failed: %v", err)
	}
}

func TestWaitForHealth(t *testing.T) {
	callCount := 0
	// Server that becomes healthy after 2 requests
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		callCount++
		if callCount < 3 {
			w.WriteHeader(http.StatusServiceUnavailable)
			return
		}
		w.WriteHeader(http.StatusOK)
	}))
	defer server.Close()

	r := &EmbeddingRunner{
		baseURL: server.URL,
	}

	ctx := context.Background()
	err := r.waitForHealth(ctx, 5*time.Second)
	if err != nil {
		t.Fatalf("waitForHealth failed: %v", err)
	}
	if callCount < 3 {
		t.Errorf("expected at least 3 health checks, got %d", callCount)
	}
}

func TestWaitForHealthTimeout(t *testing.T) {
	// Server that is always unhealthy
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusServiceUnavailable)
	}))
	defer server.Close()

	r := &EmbeddingRunner{
		baseURL: server.URL,
	}

	ctx := context.Background()
	err := r.waitForHealth(ctx, 1*time.Second)
	if err == nil {
		t.Fatal("expected timeout error")
	}
}

func TestWaitForHealthCancelled(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusServiceUnavailable)
	}))
	defer server.Close()

	r := &EmbeddingRunner{
		baseURL: server.URL,
	}

	ctx, cancel := context.WithCancel(context.Background())
	cancel() // Cancel immediately

	err := r.waitForHealth(ctx, 30*time.Second)
	if err == nil {
		t.Fatal("expected context cancelled error")
	}
}

func TestCloseWithProcess(t *testing.T) {
	// Start a real process (sleep) and then close it
	cmd := exec.Command("sleep", "60")
	if err := cmd.Start(); err != nil {
		t.Skipf("cannot start sleep process: %v", err)
	}

	r := &EmbeddingRunner{
		cmd: cmd,
	}

	if err := r.Close(); err != nil {
		t.Fatalf("Close failed: %v", err)
	}
}

func TestNewEmbeddingRunnerMissingBinary(t *testing.T) {
	dir := t.TempDir()
	ctx := context.Background()

	_, err := NewEmbeddingRunner(ctx, EmbeddingRunnerConfig{
		ModelPath: "/nonexistent/model.gguf",
		BinDir:    dir,
		Port:      19999,
	})
	if err == nil {
		t.Fatal("expected error for missing binary")
	}
}
