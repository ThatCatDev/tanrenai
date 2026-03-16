package tools

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func setupFindDir(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()

	files := map[string]string{
		"main.go":          "package main",
		"utils.go":         "package main",
		"README.md":        "# readme",
		"sub/handler.go":   "package sub",
		"sub/handler_test.go": "package sub",
	}

	for rel, content := range files {
		full := filepath.Join(dir, rel)
		if err := os.MkdirAll(filepath.Dir(full), 0755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(full, []byte(content), 0644); err != nil {
			t.Fatal(err)
		}
	}
	return dir
}

func TestFindFilesBasic(t *testing.T) {
	dir := setupFindDir(t)

	tool := &FindFilesTool{}
	args, _ := json.Marshal(map[string]interface{}{"pattern": "*.go", "path": dir})
	result, err := tool.Execute(context.Background(), string(args))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.IsError {
		t.Fatalf("expected success, got tool error: %s", result.Output)
	}

	if !strings.Contains(result.Output, "main.go") {
		t.Errorf("expected 'main.go' in results, got: %s", result.Output)
	}
	if !strings.Contains(result.Output, "utils.go") {
		t.Errorf("expected 'utils.go' in results, got: %s", result.Output)
	}
	if !strings.Contains(result.Output, "handler.go") {
		t.Errorf("expected 'handler.go' in results, got: %s", result.Output)
	}
	// README.md should not match *.go
	if strings.Contains(result.Output, "README.md") {
		t.Errorf("expected 'README.md' not to appear in *.go results, got: %s", result.Output)
	}
}

func TestFindFilesNoMatch(t *testing.T) {
	dir := setupFindDir(t)

	tool := &FindFilesTool{}
	args, _ := json.Marshal(map[string]interface{}{"pattern": "*.rb", "path": dir})
	result, err := tool.Execute(context.Background(), string(args))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.IsError {
		t.Fatalf("expected success (no matches), got tool error: %s", result.Output)
	}
	if !strings.Contains(result.Output, "No files found") {
		t.Errorf("expected 'No files found' message, got: %s", result.Output)
	}
}

func TestFindFilesEmptyPattern(t *testing.T) {
	tool := &FindFilesTool{}
	args, _ := json.Marshal(map[string]interface{}{"pattern": "", "path": "."})
	result, err := tool.Execute(context.Background(), string(args))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !result.IsError {
		t.Fatal("expected tool error for empty pattern, got success")
	}
	if !strings.Contains(result.Output, "pattern is required") {
		t.Errorf("expected 'pattern is required' error, got: %s", result.Output)
	}
}

func TestFindFilesMaxResults(t *testing.T) {
	dir := t.TempDir()
	// Create more files than the cap.
	for i := 0; i < 10; i++ {
		name := filepath.Join(dir, strings.Repeat("x", i+1)+".txt")
		if err := os.WriteFile(name, []byte("content"), 0644); err != nil {
			t.Fatal(err)
		}
	}

	tool := &FindFilesTool{}
	// Set max_results to 3, verify only 3 are returned.
	args, _ := json.Marshal(map[string]interface{}{"pattern": "*.txt", "path": dir, "max_results": 3})
	result, err := tool.Execute(context.Background(), string(args))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.IsError {
		t.Fatalf("expected success, got tool error: %s", result.Output)
	}

	lines := strings.Split(strings.TrimSpace(result.Output), "\n")
	// Count only the .txt file lines (exclude the "[showing first N results]" line).
	fileCount := 0
	for _, line := range lines {
		if strings.HasSuffix(strings.TrimSpace(line), ".txt") {
			fileCount++
		}
	}
	if fileCount != 3 {
		t.Errorf("expected 3 file results with max_results=3, got %d; output: %s", fileCount, result.Output)
	}
	if !strings.Contains(result.Output, "showing first 3 results") {
		t.Errorf("expected truncation notice, got: %s", result.Output)
	}
}

func TestFindFilesDefaultPath(t *testing.T) {
	// When path is empty, defaults to "."; just verify no crash.
	tool := &FindFilesTool{}
	args, _ := json.Marshal(map[string]interface{}{"pattern": "*.go"})
	result, err := tool.Execute(context.Background(), string(args))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	// Result may or may not find files; we only care there is no infrastructure error.
	_ = result
}

func TestFindFilesInvalidArgs(t *testing.T) {
	tool := &FindFilesTool{}
	result, err := tool.Execute(context.Background(), "{{bad json")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !result.IsError {
		t.Fatal("expected tool error for invalid JSON args, got success")
	}
	if !strings.Contains(result.Output, "invalid arguments") {
		t.Errorf("expected 'invalid arguments' error, got: %s", result.Output)
	}
}
