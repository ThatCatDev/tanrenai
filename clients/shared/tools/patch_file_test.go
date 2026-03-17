package tools

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestPatchFileSuccess(t *testing.T) {
	dir := t.TempDir()
	fpath := filepath.Join(dir, "patch_me.txt")
	original := "Hello world\nThis is a test\nGoodbye world\n"
	if err := os.WriteFile(fpath, []byte(original), 0644); err != nil {
		t.Fatal(err)
	}

	tool := &PatchFileTool{}
	args, _ := json.Marshal(map[string]interface{}{
		"path":       fpath,
		"old_string": "This is a test",
		"new_string": "This is a replacement",
	})
	result, err := tool.Execute(context.Background(), string(args))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.IsError {
		t.Fatalf("expected success, got tool error: %s", result.Output)
	}
	if !strings.Contains(result.Output, "Replaced") {
		t.Errorf("expected output to contain 'Replaced', got: %s", result.Output)
	}

	got, err := os.ReadFile(fpath)
	if err != nil {
		t.Fatalf("failed to read patched file: %v", err)
	}
	if strings.Contains(string(got), "This is a test") {
		t.Error("old_string still present in file after patch")
	}
	if !strings.Contains(string(got), "This is a replacement") {
		t.Error("new_string not found in file after patch")
	}

	// Diff should have been produced.
	if result.Diff == "" {
		t.Error("expected non-empty diff after successful patch")
	}
}

func TestPatchFileNotFound(t *testing.T) {
	tool := &PatchFileTool{}
	args, _ := json.Marshal(map[string]interface{}{
		"path":       "/tmp/does_not_exist_xyz987.txt",
		"old_string": "anything",
		"new_string": "something",
	})
	result, err := tool.Execute(context.Background(), string(args))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !result.IsError {
		t.Fatal("expected tool error for missing file, got success")
	}
	if !strings.Contains(result.Output, "file not found") {
		t.Errorf("expected 'file not found' error, got: %s", result.Output)
	}
}

func TestPatchFileNoMatch(t *testing.T) {
	dir := t.TempDir()
	fpath := filepath.Join(dir, "no_match.txt")
	if err := os.WriteFile(fpath, []byte("actual content here\n"), 0644); err != nil {
		t.Fatal(err)
	}

	tool := &PatchFileTool{}
	args, _ := json.Marshal(map[string]interface{}{
		"path":       fpath,
		"old_string": "this string does not exist",
		"new_string": "replacement",
	})
	result, err := tool.Execute(context.Background(), string(args))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !result.IsError {
		t.Fatal("expected tool error when old_string not found, got success")
	}
	if !strings.Contains(result.Output, "not found") {
		t.Errorf("expected 'not found' in error message, got: %s", result.Output)
	}
}

func TestPatchFileMultipleMatches(t *testing.T) {
	dir := t.TempDir()
	fpath := filepath.Join(dir, "multi_match.txt")
	content := "duplicate line\nduplicate line\nother content\n"
	if err := os.WriteFile(fpath, []byte(content), 0644); err != nil {
		t.Fatal(err)
	}

	tool := &PatchFileTool{}
	args, _ := json.Marshal(map[string]interface{}{
		"path":       fpath,
		"old_string": "duplicate line",
		"new_string": "unique line",
	})
	result, err := tool.Execute(context.Background(), string(args))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !result.IsError {
		t.Fatal("expected tool error when old_string matches multiple locations, got success")
	}
	if !strings.Contains(result.Output, "matches 2 locations") {
		t.Errorf("expected error about multiple matches, got: %s", result.Output)
	}
}

func TestPatchFileIdentical(t *testing.T) {
	dir := t.TempDir()
	fpath := filepath.Join(dir, "identical.txt")
	if err := os.WriteFile(fpath, []byte("some content\n"), 0644); err != nil {
		t.Fatal(err)
	}

	tool := &PatchFileTool{}
	args, _ := json.Marshal(map[string]interface{}{
		"path":       fpath,
		"old_string": "same text",
		"new_string": "same text",
	})
	result, err := tool.Execute(context.Background(), string(args))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !result.IsError {
		t.Fatal("expected tool error when old_string == new_string, got success")
	}
	if !strings.Contains(result.Output, "identical") {
		t.Errorf("expected 'identical' in error message, got: %s", result.Output)
	}
}

func TestPatchFileEmptyPath(t *testing.T) {
	tool := &PatchFileTool{}
	args, _ := json.Marshal(map[string]interface{}{
		"path":       "",
		"old_string": "old",
		"new_string": "new",
	})
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

func TestPatchFileEmptyOldString(t *testing.T) {
	dir := t.TempDir()
	fpath := filepath.Join(dir, "file.txt")
	if err := os.WriteFile(fpath, []byte("content\n"), 0644); err != nil {
		t.Fatal(err)
	}

	tool := &PatchFileTool{}
	args, _ := json.Marshal(map[string]interface{}{
		"path":       fpath,
		"old_string": "",
		"new_string": "something",
	})
	result, err := tool.Execute(context.Background(), string(args))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !result.IsError {
		t.Fatal("expected tool error for empty old_string, got success")
	}
	if !strings.Contains(result.Output, "old_string is required") {
		t.Errorf("expected 'old_string is required' error, got: %s", result.Output)
	}
}

func TestPatchFileInvalidArgs(t *testing.T) {
	tool := &PatchFileTool{}
	result, err := tool.Execute(context.Background(), "{{invalid json")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !result.IsError {
		t.Fatal("expected tool error for invalid JSON, got success")
	}
	if !strings.Contains(result.Output, "invalid arguments") {
		t.Errorf("expected 'invalid arguments' error, got: %s", result.Output)
	}
}
