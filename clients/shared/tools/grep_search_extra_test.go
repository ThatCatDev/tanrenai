package tools

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// TestIsBinaryFilename verifies isBinaryFilename returns true for binary extensions
// and false for source code / text extensions.
func TestIsBinaryFilename(t *testing.T) {
	binaries := []string{
		"program.exe", "libfoo.so", "libfoo.dylib", "foo.dll", "foo.o", "libfoo.a",
		"image.png", "photo.jpg", "photo.jpeg", "anim.gif", "icon.bmp", "icon.ico",
		"logo.svg", "document.pdf",
		"archive.zip", "tarball.tar", "compressed.gz", "bzip.bz2", "lzma.xz", "pack.7z",
		"model.gguf", "weights.safetensors", "checkpoint.pt", "model.onnx",
		"module.wasm", "cache.pyc", "Foo.class",
		"binary.bin",
	}
	for _, name := range binaries {
		if !isBinaryFilename(name) {
			t.Errorf("expected isBinaryFilename(%q) == true", name)
		}
	}

	text := []string{
		"main.go", "main.py", "index.js", "style.css", "README.md",
		"data.json", "config.yaml", "Makefile", "notes.txt", ".gitignore",
		"script.sh", "index.html",
	}
	for _, name := range text {
		if isBinaryFilename(name) {
			t.Errorf("expected isBinaryFilename(%q) == false", name)
		}
	}
}

// TestGrepSearch_MaxResultsCap tests that max_results is capped at maxGrepMatches.
func TestGrepSearch_MaxResultsCap(t *testing.T) {
	dir := t.TempDir()
	// Create a file with many matching lines.
	var sb strings.Builder
	for i := 0; i < 200; i++ {
		fmt.Fprintf(&sb, "match line %d\n", i)
	}
	if err := os.WriteFile(filepath.Join(dir, "many.txt"), []byte(sb.String()), 0644); err != nil {
		t.Fatal(err)
	}

	tool := &GrepSearchTool{}
	// Request more than maxGrepMatches (100).
	args, _ := json.Marshal(grepSearchArgs{Pattern: "match", Path: dir, MaxResults: 200})
	result, err := tool.Execute(context.Background(), string(args))
	if err != nil {
		t.Fatal(err)
	}
	if result.IsError {
		t.Fatalf("unexpected error: %s", result.Output)
	}
	// Output should contain the truncation note.
	if !strings.Contains(result.Output, "showing first") {
		t.Errorf("expected truncation note, got: %s", result.Output)
	}
}

// TestGrepSearch_SkipsBinaryFiles verifies that binary-named files are skipped.
func TestGrepSearch_SkipsBinaryFiles(t *testing.T) {
	dir := t.TempDir()
	// Write a "binary" file (by name) that contains matching text.
	if err := os.WriteFile(filepath.Join(dir, "model.gguf"), []byte("match this"), 0644); err != nil {
		t.Fatal(err)
	}
	// Write a text file that also matches so the search itself succeeds.
	if err := os.WriteFile(filepath.Join(dir, "main.go"), []byte("match this too\n"), 0644); err != nil {
		t.Fatal(err)
	}

	tool := &GrepSearchTool{}
	args, _ := json.Marshal(grepSearchArgs{Pattern: "match", Path: dir})
	result, err := tool.Execute(context.Background(), string(args))
	if err != nil {
		t.Fatal(err)
	}
	if result.IsError {
		t.Fatalf("unexpected error: %s", result.Output)
	}
	if strings.Contains(result.Output, "model.gguf") {
		t.Errorf("expected binary file to be skipped, but it appeared in output: %s", result.Output)
	}
	if !strings.Contains(result.Output, "main.go") {
		t.Errorf("expected main.go to appear in output, got: %s", result.Output)
	}
}

// TestGrepSearch_SkipsHiddenDirectories verifies that .git and similar dirs are skipped.
func TestGrepSearch_SkipsHiddenDirectories(t *testing.T) {
	dir := t.TempDir()
	gitDir := filepath.Join(dir, ".git")
	if err := os.MkdirAll(gitDir, 0755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(gitDir, "config"), []byte("match here\n"), 0644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "code.go"), []byte("match here\n"), 0644); err != nil {
		t.Fatal(err)
	}

	tool := &GrepSearchTool{}
	args, _ := json.Marshal(grepSearchArgs{Pattern: "match", Path: dir})
	result, err := tool.Execute(context.Background(), string(args))
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(result.Output, ".git") {
		t.Errorf("expected .git directory to be skipped, got: %s", result.Output)
	}
}

// TestGrepSearch_EmptyPattern tests the "pattern is required" guard.
func TestGrepSearch_EmptyPattern(t *testing.T) {
	tool := &GrepSearchTool{}
	args, _ := json.Marshal(grepSearchArgs{Pattern: "", Path: "."})
	result, err := tool.Execute(context.Background(), string(args))
	if err != nil {
		t.Fatal(err)
	}
	if !result.IsError {
		t.Fatal("expected error for empty pattern")
	}
	if !strings.Contains(result.Output, "pattern is required") {
		t.Errorf("expected 'pattern is required', got: %s", result.Output)
	}
}

// TestGrepSearch_InvalidJSONArgs tests the invalid-arguments guard.
func TestGrepSearch_InvalidJSONArgs(t *testing.T) {
	tool := &GrepSearchTool{}
	result, err := tool.Execute(context.Background(), `{not valid json}`)
	if err != nil {
		t.Fatal(err)
	}
	if !result.IsError {
		t.Fatal("expected error for invalid JSON")
	}
}

// TestGrepSearch_DefaultPath tests that an empty path defaults to ".".
func TestGrepSearch_DefaultPath(t *testing.T) {
	tool := &GrepSearchTool{}
	// Pattern that is unlikely to match anything but still exercises the path.
	args, _ := json.Marshal(map[string]string{"pattern": "ZZZZ_unlikely_42"})
	result, err := tool.Execute(context.Background(), string(args))
	if err != nil {
		t.Fatal(err)
	}
	// Should succeed (no matches is fine).
	_ = result
}
