package memory

import (
	"context"
	"testing"
)

// deterministicEmbed returns a fixed 384-dimensional vector for any input.
// This makes tests deterministic and avoids needing a real embedding model.
func deterministicEmbed(_ context.Context, _ string) ([]float32, error) {
	vec := make([]float32, 384)
	for i := range vec {
		vec[i] = 0.1
	}
	return vec, nil
}

func newTestStore(t *testing.T) *ChromemStore {
	t.Helper()
	store, err := NewChromemStoreInMemory(deterministicEmbed)
	if err != nil {
		t.Fatalf("NewChromemStoreInMemory: %v", err)
	}
	return store
}

func TestAdd(t *testing.T) {
	store := newTestStore(t)
	ctx := context.Background()

	entry := &Entry{
		UserMsg:   "hello",
		AssistMsg: "hi there",
		SessionID: "s1",
	}

	if err := store.Add(ctx, entry); err != nil {
		t.Fatalf("Add: %v", err)
	}

	if entry.ID == "" {
		t.Error("expected entry.ID to be set after Add, got empty string")
	}

	if entry.Timestamp.IsZero() {
		t.Error("expected entry.Timestamp to be set after Add")
	}
}

func TestSearch(t *testing.T) {
	store := newTestStore(t)
	ctx := context.Background()

	entries := []*Entry{
		{UserMsg: "how do I cook pasta", AssistMsg: "boil water and add pasta"},
		{UserMsg: "what is the weather today", AssistMsg: "it is sunny"},
		{UserMsg: "tell me about Go programming", AssistMsg: "Go is a statically typed language"},
	}

	for _, e := range entries {
		if err := store.Add(ctx, e); err != nil {
			t.Fatalf("Add: %v", err)
		}
	}

	results, err := store.Search(ctx, "pasta cooking", 5)
	if err != nil {
		t.Fatalf("Search: %v", err)
	}

	if len(results) == 0 {
		t.Fatal("expected at least one search result, got 0")
	}

	// With a deterministic embed function all semantic scores are equal,
	// so keyword score should push the pasta entry to the top.
	top := results[0]
	if top.Entry.UserMsg != "how do I cook pasta" {
		t.Errorf("expected top result to be the pasta entry, got UserMsg=%q", top.Entry.UserMsg)
	}
}

func TestList(t *testing.T) {
	store := newTestStore(t)
	ctx := context.Background()

	for _, msg := range []string{"first", "second", "third"} {
		if err := store.Add(ctx, &Entry{UserMsg: msg, AssistMsg: "reply"}); err != nil {
			t.Fatalf("Add: %v", err)
		}
	}

	entries, err := store.List(ctx, 0)
	if err != nil {
		t.Fatalf("List: %v", err)
	}

	if len(entries) != 3 {
		t.Errorf("expected 3 entries, got %d", len(entries))
	}

	// Test with limit
	entries, err = store.List(ctx, 2)
	if err != nil {
		t.Fatalf("List with limit: %v", err)
	}

	if len(entries) != 2 {
		t.Errorf("expected 2 entries with limit, got %d", len(entries))
	}
}

func TestDelete(t *testing.T) {
	store := newTestStore(t)
	ctx := context.Background()

	entry := &Entry{UserMsg: "delete me", AssistMsg: "ok"}
	if err := store.Add(ctx, entry); err != nil {
		t.Fatalf("Add: %v", err)
	}

	id := entry.ID
	if err := store.Delete(ctx, id); err != nil {
		t.Fatalf("Delete: %v", err)
	}

	entries, err := store.List(ctx, 0)
	if err != nil {
		t.Fatalf("List after delete: %v", err)
	}

	for _, e := range entries {
		if e.ID == id {
			t.Error("entry still present after Delete")
		}
	}

	if store.Count() != 0 {
		t.Errorf("expected count 0 after delete, got %d", store.Count())
	}
}

func TestClear(t *testing.T) {
	store := newTestStore(t)
	ctx := context.Background()

	for i := 0; i < 5; i++ {
		if err := store.Add(ctx, &Entry{UserMsg: "msg", AssistMsg: "reply"}); err != nil {
			t.Fatalf("Add: %v", err)
		}
	}

	if store.Count() != 5 {
		t.Fatalf("expected count 5 before clear, got %d", store.Count())
	}

	if err := store.Clear(ctx); err != nil {
		t.Fatalf("Clear: %v", err)
	}

	if store.Count() != 0 {
		t.Errorf("expected count 0 after clear, got %d", store.Count())
	}
}

func TestCount(t *testing.T) {
	store := newTestStore(t)
	ctx := context.Background()

	if store.Count() != 0 {
		t.Errorf("expected count 0 for empty store, got %d", store.Count())
	}

	for i := 0; i < 3; i++ {
		if err := store.Add(ctx, &Entry{UserMsg: "msg", AssistMsg: "reply"}); err != nil {
			t.Fatalf("Add: %v", err)
		}
	}

	if store.Count() != 3 {
		t.Errorf("expected count 3, got %d", store.Count())
	}
}
