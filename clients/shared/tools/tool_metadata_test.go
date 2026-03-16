package tools

import (
	"context"
	"encoding/json"
	"strings"
	"testing"
)

// toolMetaCheck verifies that a Tool's Name, Description, and Parameters
// are non-empty and that Parameters is valid JSON.
func toolMetaCheck(t *testing.T, tool Tool) {
	t.Helper()
	name := tool.Name()
	if name == "" {
		t.Errorf("%T.Name() returned empty string", tool)
	}
	desc := tool.Description()
	if desc == "" {
		t.Errorf("%T.Description() returned empty string", tool)
	}
	params := tool.Parameters()
	if params == nil || len(params) == 0 {
		t.Errorf("%T.Parameters() returned nil/empty", tool)
	}
	var m interface{}
	if err := json.Unmarshal(params, &m); err != nil {
		t.Errorf("%T.Parameters() returned invalid JSON: %v", tool, err)
	}
}

func TestFileReadToolMeta(t *testing.T)  { toolMetaCheck(t, &FileReadTool{}) }
func TestFileWriteToolMeta(t *testing.T) { toolMetaCheck(t, &FileWriteTool{}) }
func TestPatchFileToolMeta(t *testing.T) { toolMetaCheck(t, &PatchFileTool{}) }
func TestListDirToolMeta(t *testing.T)   { toolMetaCheck(t, &ListDirTool{}) }
func TestFindFilesToolMeta(t *testing.T) { toolMetaCheck(t, &FindFilesTool{}) }
func TestGrepSearchToolMeta(t *testing.T) { toolMetaCheck(t, &GrepSearchTool{}) }
func TestGitInfoToolMeta(t *testing.T)   { toolMetaCheck(t, &GitInfoTool{}) }
func TestShellExecToolMeta(t *testing.T) { toolMetaCheck(t, &ShellExecTool{}) }
func TestWebSearchToolMeta(t *testing.T) { toolMetaCheck(t, &WebSearchTool{}) }

func TestDefaultRegistry(t *testing.T) {
	r := DefaultRegistry()
	if r == nil {
		t.Fatal("expected non-nil registry from DefaultRegistry()")
	}

	expectedTools := []string{
		"file_read", "file_write", "patch_file", "list_dir",
		"find_files", "grep_search", "git_info", "shell_exec", "web_search",
	}
	for _, name := range expectedTools {
		if r.Get(name) == nil {
			t.Errorf("DefaultRegistry missing tool %q", name)
		}
	}

	apiTools := r.APITools()
	if len(apiTools) != len(expectedTools) {
		t.Errorf("DefaultRegistry().APITools() returned %d tools, want %d", len(apiTools), len(expectedTools))
	}
}

func TestGitInfoInvalidArgs(t *testing.T) {
	tool := &GitInfoTool{}
	result, err := tool.Execute(context.Background(), "{{bad")
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

func TestGitInfoUnknownCommand(t *testing.T) {
	tool := &GitInfoTool{}
	args, _ := json.Marshal(map[string]interface{}{"command": "push_force"})
	result, err := tool.Execute(context.Background(), string(args))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !result.IsError {
		t.Fatal("expected tool error for unknown git command, got success")
	}
	if !strings.Contains(result.Output, "unknown git command") {
		t.Errorf("expected 'unknown git command' error, got: %s", result.Output)
	}
}

func TestGitInfoShowMissingArgs(t *testing.T) {
	tool := &GitInfoTool{}
	args, _ := json.Marshal(map[string]interface{}{"command": "show"})
	result, err := tool.Execute(context.Background(), string(args))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !result.IsError {
		t.Fatal("expected tool error for show without args, got success")
	}
	if !strings.Contains(result.Output, "show requires a commit hash") {
		t.Errorf("expected 'show requires a commit hash' error, got: %s", result.Output)
	}
}

func TestGitInfoBlameMissingArgs(t *testing.T) {
	tool := &GitInfoTool{}
	args, _ := json.Marshal(map[string]interface{}{"command": "blame"})
	result, err := tool.Execute(context.Background(), string(args))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !result.IsError {
		t.Fatal("expected tool error for blame without args, got success")
	}
	if !strings.Contains(result.Output, "blame requires a file path") {
		t.Errorf("expected 'blame requires a file path' error, got: %s", result.Output)
	}
}

func TestGitInfoStatus(t *testing.T) {
	// git status should work in any git repo (this repo is a git repo).
	tool := &GitInfoTool{}
	args, _ := json.Marshal(map[string]interface{}{"command": "status"})
	result, err := tool.Execute(context.Background(), string(args))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	// Should succeed or return a meaningful tool error (e.g., not a git repo).
	_ = result
}

func TestGitInfoLog(t *testing.T) {
	tool := &GitInfoTool{}
	args, _ := json.Marshal(map[string]interface{}{"command": "log", "args": "5"})
	result, err := tool.Execute(context.Background(), string(args))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	_ = result
}

func TestIsRealPathVariants(t *testing.T) {
	tests := []struct {
		input string
		want  bool
	}{
		{".", true},
		{"..", true},
		{"/absolute/path", true},
		{"./relative", true},
		{"../parent", true},
		{"~/home", true},
		{"/foo/bar/baz", true},
		{"just-a-name", false},   // no slash, no space, not . or ..
		{"name with space", false}, // has spaces
	}

	for _, tt := range tests {
		got := isRealPath(tt.input)
		if got != tt.want {
			t.Errorf("isRealPath(%q) = %v, want %v", tt.input, got, tt.want)
		}
	}
}
