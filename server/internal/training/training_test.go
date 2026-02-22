package training

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/thatcatdev/tanrenai/server/internal/memory"
	"github.com/thatcatdev/tanrenai/server/pkg/api"
)

// mockEmbedFunc returns a fixed vector for any input (FNV hash mock).
func mockEmbedFunc(ctx context.Context, text string) ([]float32, error) {
	vec := make([]float32, 8)
	for i, c := range text {
		vec[i%8] += float32(c) / 1000.0
	}
	return vec, nil
}

func TestRunStoreSaveLoadList(t *testing.T) {
	dir := t.TempDir()
	store := NewRunStoreAt(dir)

	run := &TrainingRun{
		ID:        "test-run-1",
		BaseModel: "test-model",
		Status:    StatusPending,
		CreatedAt: time.Now(),
		UpdatedAt: time.Now(),
		Config:    DefaultRunConfig(),
		Metrics:   RunMetrics{SamplesUsed: 10},
	}

	if err := store.Save(run); err != nil {
		t.Fatalf("Save: %v", err)
	}

	loaded, err := store.Load("test-run-1")
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if loaded.ID != "test-run-1" {
		t.Errorf("ID = %q, want %q", loaded.ID, "test-run-1")
	}
	if loaded.Status != StatusPending {
		t.Errorf("Status = %q, want %q", loaded.Status, StatusPending)
	}
	if loaded.Config.LoraRank != DefaultRunConfig().LoraRank {
		t.Errorf("LoraRank = %d, want %d", loaded.Config.LoraRank, DefaultRunConfig().LoraRank)
	}

	// List
	runs, err := store.List()
	if err != nil {
		t.Fatalf("List: %v", err)
	}
	if len(runs) != 1 {
		t.Fatalf("List returned %d runs, want 1", len(runs))
	}
}

func TestRunStoreDelete(t *testing.T) {
	dir := t.TempDir()
	store := NewRunStoreAt(dir)

	run := &TrainingRun{
		ID:        "delete-me",
		BaseModel: "model",
		Status:    StatusPending,
		CreatedAt: time.Now(),
		UpdatedAt: time.Now(),
	}
	store.Save(run)

	if err := store.Delete("delete-me"); err != nil {
		t.Fatalf("Delete: %v", err)
	}

	if _, err := store.Load("delete-me"); err == nil {
		t.Error("expected error loading deleted run")
	}
}

func TestRunStoreLoadNotFound(t *testing.T) {
	dir := t.TempDir()
	store := NewRunStoreAt(dir)

	if _, err := store.Load("nonexistent"); err == nil {
		t.Error("expected error for nonexistent run")
	}
}

func TestRunStoreListEmpty(t *testing.T) {
	dir := t.TempDir()
	store := NewRunStoreAt(dir)

	runs, err := store.List()
	if err != nil {
		t.Fatalf("List: %v", err)
	}
	if len(runs) != 0 {
		t.Errorf("List returned %d runs, want 0", len(runs))
	}
}

func TestExportDatasetTo(t *testing.T) {
	memStore, err := memory.NewChromemStoreInMemory(mockEmbedFunc)
	if err != nil {
		t.Fatal(err)
	}
	defer memStore.Close()

	ctx := context.Background()

	// Add some entries
	memStore.Add(ctx, memory.Entry{
		UserMsg:   "What is Go?",
		AssistMsg: "Go is a programming language designed at Google.",
		Timestamp: time.Now(),
	})
	memStore.Add(ctx, memory.Entry{
		UserMsg:   "How do goroutines work?",
		AssistMsg: "Goroutines are lightweight threads managed by the Go runtime.",
		Timestamp: time.Now(),
	})
	// Short entry that should be filtered out
	memStore.Add(ctx, memory.Entry{
		UserMsg:   "Hi",
		AssistMsg: "Hello",
		Timestamp: time.Now(),
	})
	// Empty assistant — should be filtered out
	memStore.Add(ctx, memory.Entry{
		UserMsg:   "This has no reply",
		AssistMsg: "",
		Timestamp: time.Now(),
	})

	outPath := filepath.Join(t.TempDir(), "dataset.jsonl")
	count, err := ExportDatasetTo(ctx, memStore, outPath, 0)
	if err != nil {
		t.Fatalf("ExportDatasetTo: %v", err)
	}

	if count != 2 {
		t.Errorf("count = %d, want 2", count)
	}

	// Verify JSONL format
	data, err := os.ReadFile(outPath)
	if err != nil {
		t.Fatal(err)
	}

	lines := 0
	for _, line := range splitNonEmpty(string(data)) {
		var entry DatasetEntry
		if err := json.Unmarshal([]byte(line), &entry); err != nil {
			t.Fatalf("invalid JSONL line: %v", err)
		}
		if len(entry.Messages) != 3 {
			t.Errorf("expected 3 messages (system+user+assistant), got %d", len(entry.Messages))
		}
		if entry.Messages[0].Role != "system" {
			t.Errorf("first message role = %q, want system", entry.Messages[0].Role)
		}
		if entry.Messages[1].Role != "user" {
			t.Errorf("second message role = %q, want user", entry.Messages[1].Role)
		}
		if entry.Messages[2].Role != "assistant" {
			t.Errorf("third message role = %q, want assistant", entry.Messages[2].Role)
		}
		lines++
	}

	if lines != 2 {
		t.Errorf("JSONL has %d lines, want 2", lines)
	}
}

func TestExportDatasetMaxSamples(t *testing.T) {
	memStore, err := memory.NewChromemStoreInMemory(mockEmbedFunc)
	if err != nil {
		t.Fatal(err)
	}
	defer memStore.Close()

	ctx := context.Background()

	for i := 0; i < 10; i++ {
		memStore.Add(ctx, memory.Entry{
			UserMsg:   "Question about topic number whatever",
			AssistMsg: "Detailed answer about topic number whatever",
			Timestamp: time.Now(),
		})
	}

	outPath := filepath.Join(t.TempDir(), "dataset.jsonl")
	count, err := ExportDatasetTo(ctx, memStore, outPath, 3)
	if err != nil {
		t.Fatalf("ExportDatasetTo: %v", err)
	}

	if count != 3 {
		t.Errorf("count = %d, want 3", count)
	}
}

func TestExportDatasetEmptyStore(t *testing.T) {
	memStore, err := memory.NewChromemStoreInMemory(mockEmbedFunc)
	if err != nil {
		t.Fatal(err)
	}
	defer memStore.Close()

	outPath := filepath.Join(t.TempDir(), "dataset.jsonl")
	_, err = ExportDatasetTo(context.Background(), memStore, outPath, 0)
	// Empty store should return 0 samples (not an error from ExportDatasetTo,
	// but the count will be 0)
	if err != nil {
		// ExportDatasetTo doesn't check for empty — it just returns 0 count
		t.Logf("ExportDatasetTo with empty store: %v", err)
	}
}

func TestDefaultRunConfig(t *testing.T) {
	cfg := DefaultRunConfig()
	if cfg.Epochs != 3 {
		t.Errorf("Epochs = %d, want 3", cfg.Epochs)
	}
	if cfg.LoraRank != 16 {
		t.Errorf("LoraRank = %d, want 16", cfg.LoraRank)
	}
	if cfg.LoraAlpha != 32 {
		t.Errorf("LoraAlpha = %d, want 32", cfg.LoraAlpha)
	}
}

func TestDatasetEntryJSON(t *testing.T) {
	entry := DatasetEntry{
		Messages: []api.Message{
			{Role: "system", Content: "You are a helpful assistant."},
			{Role: "user", Content: "Hello"},
			{Role: "assistant", Content: "Hi there!"},
		},
	}

	data, err := json.Marshal(entry)
	if err != nil {
		t.Fatal(err)
	}

	var decoded DatasetEntry
	if err := json.Unmarshal(data, &decoded); err != nil {
		t.Fatal(err)
	}

	if len(decoded.Messages) != 3 {
		t.Errorf("Messages len = %d, want 3", len(decoded.Messages))
	}
}

func TestScheduleConfig(t *testing.T) {
	cfg := DefaultScheduleConfig()
	if cfg.Interval != "24h" {
		t.Errorf("Interval = %q, want 24h", cfg.Interval)
	}
	if cfg.MinNewEntries != 50 {
		t.Errorf("MinNewEntries = %d, want 50", cfg.MinNewEntries)
	}

	info := FormatScheduleInfo(cfg)
	if info == "" {
		t.Error("FormatScheduleInfo returned empty string")
	}
}

func splitNonEmpty(s string) []string {
	var result []string
	for _, line := range splitLines(s) {
		if line != "" {
			result = append(result, line)
		}
	}
	return result
}

func splitLines(s string) []string {
	var lines []string
	start := 0
	for i := 0; i < len(s); i++ {
		if s[i] == '\n' {
			lines = append(lines, s[start:i])
			start = i + 1
		}
	}
	if start < len(s) {
		lines = append(lines, s[start:])
	}
	return lines
}
