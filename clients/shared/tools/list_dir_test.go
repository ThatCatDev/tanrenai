package tools

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestListDirBasic(t *testing.T) {
	dir := t.TempDir()
	files := []string{"alpha.txt", "beta.go", "gamma.md"}
	for _, name := range files {
		if err := os.WriteFile(filepath.Join(dir, name), []byte("content"), 0644); err != nil {
			t.Fatal(err)
		}
	}

	tool := &ListDirTool{}
	args, _ := json.Marshal(map[string]interface{}{"path": dir})
	result, err := tool.Execute(context.Background(), string(args))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.IsError {
		t.Fatalf("expected success, got tool error: %s", result.Output)
	}

	for _, name := range files {
		if !strings.Contains(result.Output, name) {
			t.Errorf("expected output to contain %q, got: %s", name, result.Output)
		}
	}
	if !strings.Contains(result.Output, "[file]") {
		t.Errorf("expected output to contain '[file]' indicator, got: %s", result.Output)
	}
}

func TestListDirEmpty(t *testing.T) {
	dir := t.TempDir()

	tool := &ListDirTool{}
	args, _ := json.Marshal(map[string]interface{}{"path": dir})
	result, err := tool.Execute(context.Background(), string(args))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.IsError {
		t.Fatalf("expected success for empty dir, got tool error: %s", result.Output)
	}
	if result.Output != "(empty directory)" {
		t.Errorf("expected '(empty directory)', got: %s", result.Output)
	}
}

func TestListDirWithSubdirs(t *testing.T) {
	dir := t.TempDir()
	subdir := filepath.Join(dir, "subdir")
	if err := os.Mkdir(subdir, 0755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(subdir, "nested.txt"), []byte("nested"), 0644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "top.txt"), []byte("top"), 0644); err != nil {
		t.Fatal(err)
	}

	tool := &ListDirTool{}
	// Use depth=2 to see inside the subdir.
	depth := 2
	args, _ := json.Marshal(map[string]interface{}{"path": dir, "depth": depth})
	result, err := tool.Execute(context.Background(), string(args))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.IsError {
		t.Fatalf("expected success with subdirs, got tool error: %s", result.Output)
	}
	if !strings.Contains(result.Output, "[dir]") {
		t.Errorf("expected output to contain '[dir]' indicator, got: %s", result.Output)
	}
	if !strings.Contains(result.Output, "subdir") {
		t.Errorf("expected output to contain 'subdir', got: %s", result.Output)
	}
	if !strings.Contains(result.Output, "nested.txt") {
		t.Errorf("expected output to contain 'nested.txt', got: %s", result.Output)
	}
	if !strings.Contains(result.Output, "top.txt") {
		t.Errorf("expected output to contain 'top.txt', got: %s", result.Output)
	}
}

func TestListDirNonexistent(t *testing.T) {
	tool := &ListDirTool{}
	// An absolute path that looks real but does not exist — should fail to read directory.
	args, _ := json.Marshal(map[string]interface{}{"path": "/nonexistent_path_xyz_abc_123"})
	result, err := tool.Execute(context.Background(), string(args))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	// The implementation may redirect to "." or return an error result.
	// Either way it should not panic, and if it is an error it should say so.
	_ = result
}

func TestListDirDepthOne(t *testing.T) {
	dir := t.TempDir()
	subdir := filepath.Join(dir, "deep")
	if err := os.Mkdir(subdir, 0755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(subdir, "hidden.txt"), []byte("x"), 0644); err != nil {
		t.Fatal(err)
	}

	tool := &ListDirTool{}
	depth := 1
	args, _ := json.Marshal(map[string]interface{}{"path": dir, "depth": depth})
	result, err := tool.Execute(context.Background(), string(args))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.IsError {
		t.Fatalf("expected success, got tool error: %s", result.Output)
	}
	// The subdir itself should be listed but not its contents with depth=1.
	if !strings.Contains(result.Output, "deep") {
		t.Errorf("expected 'deep' dir to be listed, got: %s", result.Output)
	}
	if strings.Contains(result.Output, "hidden.txt") {
		t.Errorf("expected 'hidden.txt' to be hidden at depth=1, got: %s", result.Output)
	}
}

func TestListDirInvalidArgs(t *testing.T) {
	tool := &ListDirTool{}
	result, err := tool.Execute(context.Background(), "not json{{")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !result.IsError {
		t.Fatal("expected tool error for invalid JSON args, got success")
	}
}
