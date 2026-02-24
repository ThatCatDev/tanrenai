package tools

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestFileReadValid(t *testing.T) {
	tmp := t.TempDir()
	path := filepath.Join(tmp, "test.txt")
	os.WriteFile(path, []byte("hello world"), 0644)

	tool := &FileReadTool{}
	result, err := tool.Execute(context.Background(), `{"path":"`+path+`"}`)
	if err != nil {
		t.Fatal(err)
	}
	if result.IsError {
		t.Fatalf("unexpected error result: %s", result.Output)
	}
	if result.Output != "hello world" {
		t.Errorf("got %q, want %q", result.Output, "hello world")
	}
}

func TestFileReadNotFound(t *testing.T) {
	tool := &FileReadTool{}
	result, err := tool.Execute(context.Background(), `{"path":"/nonexistent/file.txt"}`)
	if err != nil {
		t.Fatal(err)
	}
	if !result.IsError {
		t.Error("expected error result for missing file")
	}
}

func TestFileReadTruncation(t *testing.T) {
	tmp := t.TempDir()
	path := filepath.Join(tmp, "big.txt")
	data := make([]byte, 64*1024) // 64KB, exceeds 32KB limit
	for i := range data {
		data[i] = 'A'
	}
	os.WriteFile(path, data, 0644)

	tool := &FileReadTool{}
	result, err := tool.Execute(context.Background(), `{"path":"`+path+`"}`)
	if err != nil {
		t.Fatal(err)
	}
	if result.IsError {
		t.Fatalf("unexpected error: %s", result.Output)
	}
	if !strings.Contains(result.Output, "[truncated") {
		t.Error("expected truncation notice in output")
	}
}

func TestFileReadEmptyPath(t *testing.T) {
	tool := &FileReadTool{}
	result, err := tool.Execute(context.Background(), `{"path":""}`)
	if err != nil {
		t.Fatal(err)
	}
	if !result.IsError {
		t.Error("expected error result for empty path")
	}
}

func TestFileReadInvalidJSON(t *testing.T) {
	tool := &FileReadTool{}
	result, err := tool.Execute(context.Background(), `not json`)
	if err != nil {
		t.Fatal(err)
	}
	if !result.IsError {
		t.Error("expected error result for invalid JSON")
	}
}

func TestFileWriteValid(t *testing.T) {
	tmp := t.TempDir()
	path := filepath.Join(tmp, "out.txt")

	tool := &FileWriteTool{}
	result, err := tool.Execute(context.Background(), `{"path":"`+path+`","content":"written"}`)
	if err != nil {
		t.Fatal(err)
	}
	if result.IsError {
		t.Fatalf("unexpected error: %s", result.Output)
	}

	data, _ := os.ReadFile(path)
	if string(data) != "written" {
		t.Errorf("file content = %q, want %q", string(data), "written")
	}
}

func TestFileWriteCreatesParentDirs(t *testing.T) {
	tmp := t.TempDir()
	path := filepath.Join(tmp, "a", "b", "c", "file.txt")

	tool := &FileWriteTool{}
	result, err := tool.Execute(context.Background(), `{"path":"`+path+`","content":"deep"}`)
	if err != nil {
		t.Fatal(err)
	}
	if result.IsError {
		t.Fatalf("unexpected error: %s", result.Output)
	}

	data, _ := os.ReadFile(path)
	if string(data) != "deep" {
		t.Errorf("file content = %q, want %q", string(data), "deep")
	}
}

func TestFileWriteEmptyPath(t *testing.T) {
	tool := &FileWriteTool{}
	result, err := tool.Execute(context.Background(), `{"path":"","content":"x"}`)
	if err != nil {
		t.Fatal(err)
	}
	if !result.IsError {
		t.Error("expected error for empty path")
	}
}

func TestListDirValid(t *testing.T) {
	tmp := t.TempDir()
	os.WriteFile(filepath.Join(tmp, "a.txt"), []byte(""), 0644)
	os.Mkdir(filepath.Join(tmp, "subdir"), 0755)

	tool := &ListDirTool{}
	result, err := tool.Execute(context.Background(), `{"path":"`+tmp+`"}`)
	if err != nil {
		t.Fatal(err)
	}
	if result.IsError {
		t.Fatalf("unexpected error: %s", result.Output)
	}
	if !strings.Contains(result.Output, "a.txt") {
		t.Error("expected a.txt in output")
	}
	if !strings.Contains(result.Output, "subdir") {
		t.Error("expected subdir in output")
	}
	if !strings.Contains(result.Output, "[dir]") {
		t.Error("expected [dir] indicator")
	}
	if !strings.Contains(result.Output, "[file]") {
		t.Error("expected [file] indicator")
	}
}

func TestListDirEmpty(t *testing.T) {
	tmp := t.TempDir()

	tool := &ListDirTool{}
	result, err := tool.Execute(context.Background(), `{"path":"`+tmp+`"}`)
	if err != nil {
		t.Fatal(err)
	}
	if result.IsError {
		t.Fatalf("unexpected error: %s", result.Output)
	}
	if result.Output != "(empty directory)" {
		t.Errorf("got %q, want %q", result.Output, "(empty directory)")
	}
}

func TestListDirDefaultsToDot(t *testing.T) {
	tool := &ListDirTool{}
	result, err := tool.Execute(context.Background(), `{"path":""}`)
	if err != nil {
		t.Fatal(err)
	}
	// Should not error — defaults to "."
	if result.IsError {
		t.Fatalf("unexpected error: %s", result.Output)
	}
}

func TestListDirPlaceholderFallback(t *testing.T) {
	tool := &ListDirTool{}
	result, err := tool.Execute(context.Background(), `{"path":"current_directory"}`)
	if err != nil {
		t.Fatal(err)
	}
	// Should fall back to "." instead of erroring
	if result.IsError {
		t.Fatalf("expected fallback to '.', got error: %s", result.Output)
	}
}

func TestListDirNotFound(t *testing.T) {
	tool := &ListDirTool{}
	result, err := tool.Execute(context.Background(), `{"path":"/nonexistent/path/that/surely/doesnt/exist"}`)
	if err != nil {
		t.Fatal(err)
	}
	if !result.IsError {
		t.Error("expected error for nonexistent absolute path")
	}
}

func TestShellExecValid(t *testing.T) {
	tool := &ShellExecTool{}
	result, err := tool.Execute(context.Background(), `{"command":"echo hello"}`)
	if err != nil {
		t.Fatal(err)
	}
	if result.IsError {
		t.Fatalf("unexpected error: %s", result.Output)
	}
	if strings.TrimSpace(result.Output) != "hello" {
		t.Errorf("got %q, want %q", strings.TrimSpace(result.Output), "hello")
	}
}

func TestShellExecFailure(t *testing.T) {
	tool := &ShellExecTool{}
	result, err := tool.Execute(context.Background(), `{"command":"exit 1"}`)
	if err != nil {
		t.Fatal(err)
	}
	if !result.IsError {
		t.Error("expected error result for failed command")
	}
}

func TestShellExecTimeout(t *testing.T) {
	tool := &ShellExecTool{}
	result, err := tool.Execute(context.Background(), `{"command":"sleep 10","timeout_seconds":1}`)
	if err != nil {
		t.Fatal(err)
	}
	if !result.IsError {
		t.Error("expected error result for timed out command")
	}
	if !strings.Contains(result.Output, "timed out") {
		t.Errorf("expected timeout message, got: %s", result.Output)
	}
}

func TestShellExecEmptyCommand(t *testing.T) {
	tool := &ShellExecTool{}
	result, err := tool.Execute(context.Background(), `{"command":""}`)
	if err != nil {
		t.Fatal(err)
	}
	if !result.IsError {
		t.Error("expected error for empty command")
	}
}

func TestShellExecCapturesStderr(t *testing.T) {
	tool := &ShellExecTool{}
	result, err := tool.Execute(context.Background(), `{"command":"echo err >&2 && exit 1"}`)
	if err != nil {
		t.Fatal(err)
	}
	if !result.IsError {
		t.Error("expected error result")
	}
	if !strings.Contains(result.Output, "err") {
		t.Error("expected stderr content in output")
	}
}

func TestFileWriteInvalidJSON(t *testing.T) {
	tool := &FileWriteTool{}
	result, err := tool.Execute(context.Background(), `not json`)
	if err != nil {
		t.Fatal(err)
	}
	if !result.IsError {
		t.Error("expected error for invalid JSON")
	}
}

func TestShellExecInvalidJSON(t *testing.T) {
	tool := &ShellExecTool{}
	result, err := tool.Execute(context.Background(), `not json`)
	if err != nil {
		t.Fatal(err)
	}
	if !result.IsError {
		t.Error("expected error for invalid JSON")
	}
}

func TestShellExecNoOutput(t *testing.T) {
	tool := &ShellExecTool{}
	result, err := tool.Execute(context.Background(), `{"command":"true"}`)
	if err != nil {
		t.Fatal(err)
	}
	if result.IsError {
		t.Fatalf("unexpected error: %s", result.Output)
	}
	if result.Output != "(no output)" {
		t.Errorf("got %q, want %q", result.Output, "(no output)")
	}
}

func TestShellExecCustomTimeout(t *testing.T) {
	tool := &ShellExecTool{}
	// Valid custom timeout that succeeds
	result, err := tool.Execute(context.Background(), `{"command":"echo ok","timeout_seconds":60}`)
	if err != nil {
		t.Fatal(err)
	}
	if result.IsError {
		t.Fatalf("unexpected error: %s", result.Output)
	}
	if strings.TrimSpace(result.Output) != "ok" {
		t.Errorf("got %q, want %q", strings.TrimSpace(result.Output), "ok")
	}
}

func TestListDirInvalidJSON(t *testing.T) {
	tool := &ListDirTool{}
	result, err := tool.Execute(context.Background(), `{bad`)
	if err != nil {
		t.Fatal(err)
	}
	if !result.IsError {
		t.Error("expected error for invalid JSON")
	}
}

func TestPatchFileValid(t *testing.T) {
	tmp := t.TempDir()
	path := filepath.Join(tmp, "test.go")
	os.WriteFile(path, []byte("func hello() {\n\treturn \"hello\"\n}\n"), 0644)

	tool := &PatchFileTool{}
	result, err := tool.Execute(context.Background(), `{"path":"`+path+`","old_string":"\"hello\"","new_string":"\"world\""}`)
	if err != nil {
		t.Fatal(err)
	}
	if result.IsError {
		t.Fatalf("unexpected error: %s", result.Output)
	}
	if !strings.Contains(result.Output, "Replaced") {
		t.Errorf("expected success message, got %q", result.Output)
	}

	data, _ := os.ReadFile(path)
	if !strings.Contains(string(data), "\"world\"") {
		t.Errorf("expected replaced content, got %q", string(data))
	}
	if strings.Contains(string(data), "\"hello\"") {
		t.Error("old string should be gone")
	}
}

func TestPatchFileNotFound(t *testing.T) {
	tool := &PatchFileTool{}
	result, err := tool.Execute(context.Background(), `{"path":"/nonexistent/file.go","old_string":"a","new_string":"b"}`)
	if err != nil {
		t.Fatal(err)
	}
	if !result.IsError {
		t.Error("expected error for missing file")
	}
	if !strings.Contains(result.Output, "not found") {
		t.Errorf("expected 'not found' message, got %q", result.Output)
	}
}

func TestPatchFileOldStringNotFound(t *testing.T) {
	tmp := t.TempDir()
	path := filepath.Join(tmp, "test.txt")
	os.WriteFile(path, []byte("some content here"), 0644)

	tool := &PatchFileTool{}
	result, err := tool.Execute(context.Background(), `{"path":"`+path+`","old_string":"nonexistent","new_string":"replacement"}`)
	if err != nil {
		t.Fatal(err)
	}
	if !result.IsError {
		t.Error("expected error when old_string not found")
	}
	if !strings.Contains(result.Output, "not found") {
		t.Errorf("expected 'not found' message, got %q", result.Output)
	}
	if !strings.Contains(result.Output, "some content") {
		t.Error("expected file snippet in error output")
	}
}

func TestPatchFileMultipleMatches(t *testing.T) {
	tmp := t.TempDir()
	path := filepath.Join(tmp, "test.txt")
	os.WriteFile(path, []byte("foo bar foo baz foo"), 0644)

	tool := &PatchFileTool{}
	result, err := tool.Execute(context.Background(), `{"path":"`+path+`","old_string":"foo","new_string":"qux"}`)
	if err != nil {
		t.Fatal(err)
	}
	if !result.IsError {
		t.Error("expected error for multiple matches")
	}
	if !strings.Contains(result.Output, "3 locations") {
		t.Errorf("expected match count in error, got %q", result.Output)
	}

	// File should be unchanged
	data, _ := os.ReadFile(path)
	if string(data) != "foo bar foo baz foo" {
		t.Error("file should not be modified on multiple matches")
	}
}

func TestPatchFileSameStrings(t *testing.T) {
	tool := &PatchFileTool{}
	result, err := tool.Execute(context.Background(), `{"path":"any.txt","old_string":"same","new_string":"same"}`)
	if err != nil {
		t.Fatal(err)
	}
	if !result.IsError {
		t.Error("expected error when old_string == new_string")
	}
	if !strings.Contains(result.Output, "identical") {
		t.Errorf("expected 'identical' message, got %q", result.Output)
	}
}

func TestPatchFileInvalidJSON(t *testing.T) {
	tool := &PatchFileTool{}
	result, err := tool.Execute(context.Background(), `not json`)
	if err != nil {
		t.Fatal(err)
	}
	if !result.IsError {
		t.Error("expected error for invalid JSON")
	}
}

func TestPatchFileEmptyPath(t *testing.T) {
	tool := &PatchFileTool{}
	result, err := tool.Execute(context.Background(), `{"path":"","old_string":"a","new_string":"b"}`)
	if err != nil {
		t.Fatal(err)
	}
	if !result.IsError {
		t.Error("expected error for empty path")
	}
}

func TestToolDescriptionsAndParameters(t *testing.T) {
	// Exercise Description() and Parameters() for all tools
	allTools := []Tool{&FileReadTool{}, &FileWriteTool{}, &PatchFileTool{}, &ListDirTool{}, &ShellExecTool{}}
	for _, tool := range allTools {
		if tool.Description() == "" {
			t.Errorf("%s: empty description", tool.Name())
		}
		params := tool.Parameters()
		if len(params) == 0 {
			t.Errorf("%s: empty parameters", tool.Name())
		}
		// Verify it's valid JSON
		var m map[string]any
		if err := json.Unmarshal(params, &m); err != nil {
			t.Errorf("%s: parameters not valid JSON: %v", tool.Name(), err)
		}
	}
}

func TestSchemaMustMarshal(t *testing.T) {
	s := Schema{
		Type: "object",
		Properties: map[string]SchemaProperty{
			"name": {Type: "string", Description: "a name"},
		},
		Required: []string{"name"},
	}
	raw := s.MustMarshal()
	if len(raw) == 0 {
		t.Error("expected non-empty JSON")
	}
	var m map[string]any
	if err := json.Unmarshal(raw, &m); err != nil {
		t.Errorf("not valid JSON: %v", err)
	}
}

func TestIsRealPath(t *testing.T) {
	tests := []struct {
		input string
		want  bool
	}{
		{".", true},
		{"..", true},
		{"./foo", true},
		{"../bar", true},
		{"/absolute/path", true},
		{"~/home", true},
		{"relative/with/slashes", true},
		{"current_directory", false},
		{"my_files", false},
		{"somefile", false},
	}
	for _, tt := range tests {
		got := isRealPath(tt.input)
		if got != tt.want {
			t.Errorf("isRealPath(%q) = %v, want %v", tt.input, got, tt.want)
		}
	}
}
