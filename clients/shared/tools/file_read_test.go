package tools

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestFileReadExistingFile(t *testing.T) {
	dir := t.TempDir()
	fpath := filepath.Join(dir, "hello.txt")
	content := "hello, world\nsecond line\n"
	if err := os.WriteFile(fpath, []byte(content), 0644); err != nil {
		t.Fatal(err)
	}

	tool := &FileReadTool{}
	args, _ := json.Marshal(fileReadArgs{Path: fpath})
	result, err := tool.Execute(context.Background(), string(args))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.IsError {
		t.Fatalf("expected success, got tool error: %s", result.Output)
	}
	if result.Output != content {
		t.Errorf("expected %q, got %q", content, result.Output)
	}
}

func TestFileReadMissingFile(t *testing.T) {
	tool := &FileReadTool{}
	args, _ := json.Marshal(fileReadArgs{Path: "/tmp/nonexistent_file_abc123.txt"})
	result, err := tool.Execute(context.Background(), string(args))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !result.IsError {
		t.Fatal("expected tool error for missing file, got success")
	}
	if !strings.Contains(result.Output, "failed to read file") {
		t.Errorf("expected error message about failed read, got: %s", result.Output)
	}
}

func TestFileReadEmptyPath(t *testing.T) {
	tool := &FileReadTool{}
	args, _ := json.Marshal(fileReadArgs{Path: ""})
	result, err := tool.Execute(context.Background(), string(args))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !result.IsError {
		t.Fatal("expected tool error for empty path, got success")
	}
	if !strings.Contains(result.Output, "path is required") {
		t.Errorf("expected 'path is required' error, got: %s", result.Output)
	}
}
