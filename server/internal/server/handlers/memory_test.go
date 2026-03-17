package handlers

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/ThatCatDev/tanrenai/server/internal/memory"
	"github.com/ThatCatDev/tanrenai/server/pkg/api"
)

const embedDim = 384 // MiniLM dimension

// deterministicEmbed returns a fixed vector for any input text.
func deterministicEmbed(_ context.Context, _ string) ([]float32, error) {
	vec := make([]float32, embedDim)
	for i := range vec {
		vec[i] = 0.1
	}

	return vec, nil
}

func newTestHandler(t *testing.T) *MemoryHandler {
	t.Helper()
	store, err := memory.NewChromemStoreInMemory(deterministicEmbed)
	if err != nil {
		t.Fatalf("NewChromemStoreInMemory: %v", err)
	}

	return &MemoryHandler{MemStore: store}
}

// storeEntry is a helper that stores a memory entry and returns the ID.
func storeEntry(t *testing.T, h *MemoryHandler, userMsg, assistMsg string) string {
	t.Helper()
	body, _ := json.Marshal(api.MemoryStoreRequest{
		UserMsg:   userMsg,
		AssistMsg: assistMsg,
	})
	req := httptest.NewRequest(http.MethodPost, "/v1/memory/store", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	h.Store(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("store returned status %d: %s", w.Code, w.Body.String())
	}

	var resp api.MemoryStoreResponse
	if err := json.NewDecoder(w.Body).Decode(&resp); err != nil {
		t.Fatalf("decode store response: %v", err)
	}

	return resp.ID
}

func TestStoreMemory(t *testing.T) {
	h := newTestHandler(t)
	id := storeEntry(t, h, "hello", "world")
	if id == "" {
		t.Fatal("expected non-empty ID from store")
	}
}

func TestSearchMemory(t *testing.T) {
	h := newTestHandler(t)

	storeEntry(t, h, "the weather is sunny", "glad to hear it")
	storeEntry(t, h, "tell me about Go programming", "Go is great")

	body, _ := json.Marshal(api.MemorySearchRequest{Query: "weather", Limit: 5})
	req := httptest.NewRequest(http.MethodPost, "/v1/memory/search", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	h.Search(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("search returned status %d: %s", w.Code, w.Body.String())
	}

	var resp api.MemorySearchResponse
	if err := json.NewDecoder(w.Body).Decode(&resp); err != nil {
		t.Fatalf("decode search response: %v", err)
	}

	if len(resp.Results) == 0 {
		t.Fatal("expected at least one search result")
	}

	// With a deterministic embedding, all semantic scores are identical,
	// so keyword score should boost the weather entry to the top.
	top := resp.Results[0]
	if !strings.Contains(top.Entry.UserMsg, "weather") {
		t.Errorf("expected top result to contain 'weather', got user_msg=%q", top.Entry.UserMsg)
	}
}

func TestSearchMemoryEmptyQuery(t *testing.T) {
	h := newTestHandler(t)

	body, _ := json.Marshal(api.MemorySearchRequest{Query: ""})
	req := httptest.NewRequest(http.MethodPost, "/v1/memory/search", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	h.Search(w, req)

	if w.Code != http.StatusBadRequest {
		t.Fatalf("expected 400 for empty query, got %d", w.Code)
	}
}

func TestListMemory(t *testing.T) {
	h := newTestHandler(t)

	storeEntry(t, h, "msg1", "resp1")
	storeEntry(t, h, "msg2", "resp2")
	storeEntry(t, h, "msg3", "resp3")

	req := httptest.NewRequest(http.MethodGet, "/v1/memory/list", nil)
	w := httptest.NewRecorder()
	h.List(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("list returned status %d: %s", w.Code, w.Body.String())
	}

	var resp api.MemoryListResponse
	if err := json.NewDecoder(w.Body).Decode(&resp); err != nil {
		t.Fatalf("decode list response: %v", err)
	}

	if len(resp.Entries) != 3 {
		t.Errorf("expected 3 entries, got %d", len(resp.Entries))
	}
	if resp.Total != 3 {
		t.Errorf("expected total 3, got %d", resp.Total)
	}
}

func TestDeleteMemory(t *testing.T) {
	h := newTestHandler(t)

	id := storeEntry(t, h, "to be deleted", "bye")

	// Delete via a mux that extracts {id} from the path.
	mux := http.NewServeMux()
	mux.HandleFunc("DELETE /v1/memory/{id}", h.Delete)

	req := httptest.NewRequest(http.MethodDelete, "/v1/memory/"+id, nil)
	w := httptest.NewRecorder()
	mux.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("delete returned status %d: %s", w.Code, w.Body.String())
	}

	// Verify count is now 0.
	if h.MemStore.Count() != 0 {
		t.Errorf("expected count 0 after delete, got %d", h.MemStore.Count())
	}
}

func TestDeleteMemoryMissingID(t *testing.T) {
	h := newTestHandler(t)

	// Call Delete directly without a path value, simulating missing ID.
	req := httptest.NewRequest(http.MethodDelete, "/v1/memory/", nil)
	w := httptest.NewRecorder()
	h.Delete(w, req)

	if w.Code != http.StatusBadRequest {
		t.Fatalf("expected 400 for missing ID, got %d", w.Code)
	}
}

func TestClearMemory(t *testing.T) {
	h := newTestHandler(t)

	storeEntry(t, h, "a", "b")
	storeEntry(t, h, "c", "d")

	if h.MemStore.Count() != 2 {
		t.Fatalf("expected 2 entries before clear, got %d", h.MemStore.Count())
	}

	req := httptest.NewRequest(http.MethodDelete, "/v1/memory", nil)
	w := httptest.NewRecorder()
	h.Clear(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("clear returned status %d: %s", w.Code, w.Body.String())
	}

	if h.MemStore.Count() != 0 {
		t.Errorf("expected count 0 after clear, got %d", h.MemStore.Count())
	}
}

func TestCountMemory(t *testing.T) {
	h := newTestHandler(t)

	// Count starts at 0.
	req := httptest.NewRequest(http.MethodGet, "/v1/memory/count", nil)
	w := httptest.NewRecorder()
	h.Count(w, req)

	var resp api.MemoryCountResponse
	if err := json.NewDecoder(w.Body).Decode(&resp); err != nil {
		t.Fatalf("decode count response: %v", err)
	}
	if resp.Count != 0 {
		t.Errorf("expected count 0, got %d", resp.Count)
	}

	// Store 2 entries and check again.
	storeEntry(t, h, "x", "y")
	storeEntry(t, h, "a", "b")

	w = httptest.NewRecorder()
	req = httptest.NewRequest(http.MethodGet, "/v1/memory/count", nil)
	h.Count(w, req)

	if err := json.NewDecoder(w.Body).Decode(&resp); err != nil {
		t.Fatalf("decode count response: %v", err)
	}
	if resp.Count != 2 {
		t.Errorf("expected count 2, got %d", resp.Count)
	}
}

func TestStoreMemoryOversizedBody(t *testing.T) {
	h := newTestHandler(t)

	// Create a body larger than the 1MB limit.
	oversized := strings.Repeat("x", 1<<20+1)
	body, _ := json.Marshal(api.MemoryStoreRequest{
		UserMsg:   oversized,
		AssistMsg: "resp",
	})

	req := httptest.NewRequest(http.MethodPost, "/v1/memory/store", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	h.Store(w, req)

	// MaxBytesReader causes a decode error → 400.
	if w.Code != http.StatusBadRequest {
		t.Fatalf("expected 400 for oversized body, got %d", w.Code)
	}
}

func TestSearchMemoryOversizedBody(t *testing.T) {
	h := newTestHandler(t)

	oversized := strings.Repeat("x", 1<<20+1)
	body, _ := json.Marshal(api.MemorySearchRequest{
		Query: oversized,
	})

	req := httptest.NewRequest(http.MethodPost, "/v1/memory/search", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	h.Search(w, req)

	if w.Code != http.StatusBadRequest {
		t.Fatalf("expected 400 for oversized body, got %d", w.Code)
	}
}

func TestDeleteMemoryNotFound(t *testing.T) {
	h := newTestHandler(t)

	mux := http.NewServeMux()
	mux.HandleFunc("DELETE /v1/memory/{id}", h.Delete)

	req := httptest.NewRequest(http.MethodDelete, "/v1/memory/nonexistent-id", nil)
	w := httptest.NewRecorder()
	mux.ServeHTTP(w, req)

	if w.Code != http.StatusNotFound {
		t.Fatalf("expected 404 for nonexistent ID, got %d: %s", w.Code, w.Body.String())
	}
}

func TestListMemoryWithLimit(t *testing.T) {
	h := newTestHandler(t)

	for i := 0; i < 5; i++ {
		storeEntry(t, h, "msg", "resp")
	}

	req := httptest.NewRequest(http.MethodGet, "/v1/memory/list?limit=3", nil)
	w := httptest.NewRecorder()
	h.List(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("list returned status %d: %s", w.Code, w.Body.String())
	}

	var resp api.MemoryListResponse
	if err := json.NewDecoder(w.Body).Decode(&resp); err != nil {
		t.Fatalf("decode list response: %v", err)
	}
	if len(resp.Entries) != 3 {
		t.Errorf("expected 3 entries with limit=3, got %d", len(resp.Entries))
	}
}

func TestStoreMemoryBadJSON(t *testing.T) {
	h := newTestHandler(t)

	req := httptest.NewRequest(http.MethodPost, "/v1/memory/store", bytes.NewReader([]byte(`{bad json`)))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	h.Store(w, req)

	if w.Code != http.StatusBadRequest {
		t.Fatalf("expected 400 for bad JSON, got %d", w.Code)
	}
}

func TestSearchMemoryBadJSON(t *testing.T) {
	h := newTestHandler(t)

	req := httptest.NewRequest(http.MethodPost, "/v1/memory/search", bytes.NewReader([]byte(`{bad json`)))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	h.Search(w, req)

	if w.Code != http.StatusBadRequest {
		t.Fatalf("expected 400 for bad JSON, got %d", w.Code)
	}
}

// errStore is a Store implementation that returns errors for testing writeMemoryError.
type errStore struct {
	err error
}

func (s *errStore) Add(_ context.Context, _ *memory.Entry) error { return s.err }
func (s *errStore) Search(_ context.Context, _ string, _ int) ([]memory.SearchResult, error) {
	return nil, s.err
}
func (s *errStore) List(_ context.Context, _ int) ([]memory.Entry, error) { return nil, s.err }
func (s *errStore) Delete(_ context.Context, _ string) error              { return s.err }
func (s *errStore) Clear(_ context.Context) error                         { return s.err }
func (s *errStore) Count() int                                            { return 0 }
func (s *errStore) Close() error                                          { return nil }

func TestWriteMemoryErrorNotFound(t *testing.T) {
	h := &MemoryHandler{MemStore: &errStore{err: memory.ErrNotFound}}

	mux := http.NewServeMux()
	mux.HandleFunc("DELETE /v1/memory/{id}", h.Delete)

	req := httptest.NewRequest(http.MethodDelete, "/v1/memory/any-id", nil)
	w := httptest.NewRecorder()
	mux.ServeHTTP(w, req)

	if w.Code != http.StatusNotFound {
		t.Fatalf("expected 404 for ErrNotFound, got %d: %s", w.Code, w.Body.String())
	}
}

func TestWriteMemoryErrorEmpty(t *testing.T) {
	h := &MemoryHandler{MemStore: &errStore{err: memory.ErrEmpty}}

	body, _ := json.Marshal(api.MemorySearchRequest{Query: "test"})
	req := httptest.NewRequest(http.MethodPost, "/v1/memory/search", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	h.Search(w, req)

	if w.Code != http.StatusBadRequest {
		t.Fatalf("expected 400 for ErrEmpty, got %d: %s", w.Code, w.Body.String())
	}
}

func TestWriteMemoryErrorGeneric(t *testing.T) {
	h := &MemoryHandler{MemStore: &errStore{err: errors.New("some internal error")}}

	req := httptest.NewRequest(http.MethodDelete, "/v1/memory", nil)
	w := httptest.NewRecorder()
	h.Clear(w, req)

	if w.Code != http.StatusInternalServerError {
		t.Fatalf("expected 500 for generic error, got %d: %s", w.Code, w.Body.String())
	}
}
