package training

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"

	"github.com/ThatCatDev/tanrenai/server/internal/config"
	"github.com/ThatCatDev/tanrenai/server/internal/memory"
	"github.com/ThatCatDev/tanrenai/server/pkg/api"
)

const minEntryLength = 20

// ExportDataset extracts training data from memory and writes it as JSONL.
// Returns the dataset path and the number of samples written.
func ExportDataset(ctx context.Context, memStore memory.Store, runID string, maxSamples int) (string, int, error) {
	entries, err := memStore.List(ctx, 0) // 0 = all entries
	if err != nil {
		return "", 0, fmt.Errorf("list memories: %w", err)
	}

	if len(entries) == 0 {
		return "", 0, fmt.Errorf("no memory entries to export")
	}

	dir := config.TrainingDatasetsDir()
	if err := os.MkdirAll(dir, 0755); err != nil {
		return "", 0, fmt.Errorf("create datasets dir: %w", err)
	}

	datasetPath := filepath.Join(dir, runID+".jsonl")
	f, err := os.Create(datasetPath)
	if err != nil {
		return "", 0, fmt.Errorf("create dataset file: %w", err)
	}
	defer f.Close()

	enc := json.NewEncoder(f)
	count := 0

	for _, entry := range entries {
		if maxSamples > 0 && count >= maxSamples {
			break
		}

		// Filter: skip entries with empty assistant responses
		if entry.AssistMsg == "" {
			continue
		}

		// Filter: skip very short exchanges
		if len(entry.UserMsg)+len(entry.AssistMsg) < minEntryLength {
			continue
		}

		sample := DatasetEntry{
			Messages: []api.Message{
				{Role: "system", Content: "You are a helpful assistant."},
				{Role: "user", Content: entry.UserMsg},
				{Role: "assistant", Content: entry.AssistMsg},
			},
		}

		if err := enc.Encode(sample); err != nil {
			return "", 0, fmt.Errorf("encode sample: %w", err)
		}
		count++
	}

	if count == 0 {
		os.Remove(datasetPath)
		return "", 0, fmt.Errorf("no valid samples after filtering")
	}

	return datasetPath, count, nil
}

// ExportDatasetTo exports to a specific path (for testing).
func ExportDatasetTo(ctx context.Context, memStore memory.Store, datasetPath string, maxSamples int) (int, error) {
	entries, err := memStore.List(ctx, 0)
	if err != nil {
		return 0, fmt.Errorf("list memories: %w", err)
	}

	f, err := os.Create(datasetPath)
	if err != nil {
		return 0, fmt.Errorf("create dataset file: %w", err)
	}
	defer f.Close()

	enc := json.NewEncoder(f)
	count := 0

	for _, entry := range entries {
		if maxSamples > 0 && count >= maxSamples {
			break
		}
		if entry.AssistMsg == "" {
			continue
		}
		if len(entry.UserMsg)+len(entry.AssistMsg) < minEntryLength {
			continue
		}

		sample := DatasetEntry{
			Messages: []api.Message{
				{Role: "system", Content: "You are a helpful assistant."},
				{Role: "user", Content: entry.UserMsg},
				{Role: "assistant", Content: entry.AssistMsg},
			},
		}

		if err := enc.Encode(sample); err != nil {
			return 0, fmt.Errorf("encode sample: %w", err)
		}
		count++
	}

	return count, nil
}
