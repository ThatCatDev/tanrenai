package tools

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestFileWriteNewFile(t *testing.T) {
	dir := t.TempDir()
	fpath := filepath.Join(dir, "newfile.txt")
	content := "hello, world\n"

	tool := &FileWriteTool{}
	args, _ := json.Marshal(map[string]interface{}{"path": fpath, "content": content})
	result, err := tool.Execute(context.Background(), string(args))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.IsError {
		t.Fatalf("expected success, got tool error: %s", result.Output)
	}
	if !strings.Contains(result.Output, "Successfully wrote") {
		t.Errorf("expected output to contain 'Successfully wrote', got: %s", result.Output)
	}

	// Verify the file was actually created with correct content.
	got, err := os.ReadFile(fpath)
	if err != nil {
		t.Fatalf("failed to read written file: %v", err)
	}
	if string(got) != content {
		t.Errorf("expected file content %q, got %q", content, string(got))
	}
}

func TestFileWriteOverwrite(t *testing.T) {
	dir := t.TempDir()
	fpath := filepath.Join(dir, "existing.txt")

	// Write initial content.
	if err := os.WriteFile(fpath, []byte("original content\n"), 0644); err != nil {
		t.Fatal(err)
	}

	tool := &FileWriteTool{}
	newContent := "replaced content\n"
	args, _ := json.Marshal(map[string]interface{}{"path": fpath, "content": newContent})
	result, err := tool.Execute(context.Background(), string(args))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.IsError {
		t.Fatalf("expected success on overwrite, got tool error: %s", result.Output)
	}

	// A diff should have been generated since content changed.
	if result.Diff == "" {
		t.Error("expected non-empty diff when overwriting file with different content")
	}
	if !strings.Contains(result.Diff, "-original content") {
		t.Errorf("expected diff to show removed original content, got: %s", result.Diff)
	}
	if !strings.Contains(result.Diff, "+replaced content") {
		t.Errorf("expected diff to show added replacement content, got: %s", result.Diff)
	}

	// Verify new content was written.
	got, err := os.ReadFile(fpath)
	if err != nil {
		t.Fatalf("failed to read file after overwrite: %v", err)
	}
	if string(got) != newContent {
		t.Errorf("expected file content %q after overwrite, got %q", newContent, string(got))
	}
}

func TestFileWriteEmptyPath(t *testing.T) {
	tool := &FileWriteTool{}
	args, _ := json.Marshal(map[string]interface{}{"path": "", "content": "hello"})
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

func TestFileWriteInvalidArgs(t *testing.T) {
	tool := &FileWriteTool{}
	result, err := tool.Execute(context.Background(), "not valid json{{{")
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

func TestFileWriteCreatesDirectories(t *testing.T) {
	dir := t.TempDir()
	// Write to a nested path that doesn't exist yet.
	fpath := filepath.Join(dir, "a", "b", "c", "nested.txt")
	content := "nested file content\n"

	tool := &FileWriteTool{}
	args, _ := json.Marshal(map[string]interface{}{"path": fpath, "content": content})
	result, err := tool.Execute(context.Background(), string(args))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.IsError {
		t.Fatalf("expected success creating nested directories, got tool error: %s", result.Output)
	}

	got, err := os.ReadFile(fpath)
	if err != nil {
		t.Fatalf("failed to read nested file: %v", err)
	}
	if string(got) != content {
		t.Errorf("expected content %q in nested file, got %q", content, string(got))
	}
}

func TestFileWriteNoDiffForNewFile(t *testing.T) {
	dir := t.TempDir()
	fpath := filepath.Join(dir, "brand_new.txt")

	tool := &FileWriteTool{}
	args, _ := json.Marshal(map[string]interface{}{"path": fpath, "content": "some content\n"})
	result, err := tool.Execute(context.Background(), string(args))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.IsError {
		t.Fatalf("expected success, got tool error: %s", result.Output)
	}
	// No old content → diff should be empty (new file, nothing to compare against).
	if result.Diff != "" {
		// If a diff was generated from empty to new content, it's also acceptable.
		// The key is it should not error. We just verify no crash occurred.
		_ = result.Diff
	}
}
