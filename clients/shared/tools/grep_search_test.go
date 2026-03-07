package tools

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func setupGrepDir(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()

	files := map[string]string{
		"main.go":      "package main\n\nimport \"fmt\"\n\nfunc main() {\n\tfmt.Println(\"hello\")\n}\n",
		"utils.go":     "package main\n\nfunc helper() string {\n\treturn \"helper\"\n}\n",
		"notes.txt":    "TODO: fix the bug\nDone: refactor code\nTODO: add tests\n",
	}

	for name, content := range files {
		if err := os.WriteFile(filepath.Join(dir, name), []byte(content), 0644); err != nil {
			t.Fatal(err)
		}
	}
	return dir
}

func TestGrepSearchMatchFound(t *testing.T) {
	dir := setupGrepDir(t)

	tool := &GrepSearchTool{}
	args, _ := json.Marshal(grepSearchArgs{Pattern: "TODO", Path: dir})
	result, err := tool.Execute(context.Background(), string(args))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.IsError {
		t.Fatalf("expected success, got tool error: %s", result.Output)
	}

	// Should find 2 TODO lines in notes.txt.
	if !strings.Contains(result.Output, "TODO") {
		t.Errorf("expected output to contain 'TODO', got: %s", result.Output)
	}
	if !strings.Contains(result.Output, "notes.txt") {
		t.Errorf("expected output to reference 'notes.txt', got: %s", result.Output)
	}
}

func TestGrepSearchNoMatch(t *testing.T) {
	dir := setupGrepDir(t)

	tool := &GrepSearchTool{}
	args, _ := json.Marshal(grepSearchArgs{Pattern: "ZZZZNOTFOUND", Path: dir})
	result, err := tool.Execute(context.Background(), string(args))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.IsError {
		t.Fatalf("expected success (no matches), got tool error: %s", result.Output)
	}
	if !strings.Contains(result.Output, "No matches found") {
		t.Errorf("expected 'No matches found' message, got: %s", result.Output)
	}
}

func TestGrepSearchWithFileGlob(t *testing.T) {
	dir := setupGrepDir(t)

	tool := &GrepSearchTool{}
	// Search for "func" but only in .go files.
	args, _ := json.Marshal(grepSearchArgs{Pattern: "func", Path: dir, FileGlob: "*.go"})
	result, err := tool.Execute(context.Background(), string(args))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.IsError {
		t.Fatalf("expected success, got tool error: %s", result.Output)
	}
	if !strings.Contains(result.Output, "func") {
		t.Errorf("expected matches containing 'func', got: %s", result.Output)
	}
	// Should not include notes.txt matches.
	if strings.Contains(result.Output, "notes.txt") {
		t.Errorf("expected no matches from notes.txt with *.go glob, got: %s", result.Output)
	}
}

func TestGrepSearchInvalidRegex(t *testing.T) {
	tool := &GrepSearchTool{}
	args, _ := json.Marshal(grepSearchArgs{Pattern: "[invalid", Path: "."})
	result, err := tool.Execute(context.Background(), string(args))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !result.IsError {
		t.Fatal("expected tool error for invalid regex, got success")
	}
	if !strings.Contains(result.Output, "invalid regex") {
		t.Errorf("expected 'invalid regex' error, got: %s", result.Output)
	}
}
