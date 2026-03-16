package tools

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func TestGlobalConfigDirDefault(t *testing.T) {
	// Clear the env var so we get the default.
	original := os.Getenv("TANRENAI_CONFIG_DIR")
	os.Unsetenv("TANRENAI_CONFIG_DIR")
	defer os.Setenv("TANRENAI_CONFIG_DIR", original)

	got := GlobalConfigDir()
	if got == "" {
		t.Fatal("expected non-empty default config dir")
	}
	home, _ := os.UserHomeDir()
	expected := filepath.Join(home, ".tanrenai")
	if got != expected {
		t.Errorf("expected GlobalConfigDir()=%q, got %q", expected, got)
	}
}

func TestGlobalConfigDirEnvOverride(t *testing.T) {
	custom := "/custom/config/dir"
	os.Setenv("TANRENAI_CONFIG_DIR", custom)
	defer os.Unsetenv("TANRENAI_CONFIG_DIR")

	got := GlobalConfigDir()
	if got != custom {
		t.Errorf("expected GlobalConfigDir()=%q when env is set, got %q", custom, got)
	}
}

func TestLoadPermissionsWithFiles(t *testing.T) {
	// Use a temp dir for both global and local config.
	tmpDir := t.TempDir()

	// Write a global permissions file.
	globalCfg := PermissionsConfig{
		Rules: []PermissionRule{
			{Tool: "file_read"},
		},
	}
	globalData, _ := json.MarshalIndent(globalCfg, "", "  ")
	if err := os.WriteFile(filepath.Join(tmpDir, "permissions.json"), globalData, 0644); err != nil {
		t.Fatal(err)
	}

	// Override the global config dir.
	os.Setenv("TANRENAI_CONFIG_DIR", tmpDir)
	defer os.Unsetenv("TANRENAI_CONFIG_DIR")

	// Create a local .tanrenai dir and permissions file in a temp working area.
	localDir := t.TempDir()
	localTanrenai := filepath.Join(localDir, ".tanrenai")
	if err := os.MkdirAll(localTanrenai, 0755); err != nil {
		t.Fatal(err)
	}
	localCfg := PermissionsConfig{
		Rules: []PermissionRule{
			{Tool: "list_dir"},
		},
	}
	localData, _ := json.MarshalIndent(localCfg, "", "  ")
	if err := os.WriteFile(filepath.Join(localTanrenai, "permissions.json"), localData, 0644); err != nil {
		t.Fatal(err)
	}

	// Change cwd so LoadPermissions picks up the local file.
	origDir, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	defer os.Chdir(origDir)
	if err := os.Chdir(localDir); err != nil {
		t.Fatal(err)
	}

	p := LoadPermissions()
	if p == nil {
		t.Fatal("expected non-nil Permissions from LoadPermissions")
	}

	// Both global and local rules should be loaded.
	if !p.IsAllowed("file_read", `{}`) {
		t.Error("expected 'file_read' to be allowed (from global config)")
	}
	if !p.IsAllowed("list_dir", `{}`) {
		t.Error("expected 'list_dir' to be allowed (from local config)")
	}
}

func TestAllowTool(t *testing.T) {
	// Use a temp dir as the write target for AllowTool.
	tmpDir := t.TempDir()

	p := &Permissions{
		path: filepath.Join(tmpDir, ".tanrenai", "permissions.json"),
	}

	if err := p.AllowTool("grep_search"); err != nil {
		t.Fatalf("AllowTool returned error: %v", err)
	}

	if !p.IsAllowed("grep_search", `{}`) {
		t.Error("expected 'grep_search' to be allowed after AllowTool")
	}
	if p.IsAllowed("shell_exec", `{"command": "rm -rf /"}`) {
		t.Error("expected 'shell_exec' to NOT be allowed (no rule added)")
	}

	// Verify the file was written.
	data, err := os.ReadFile(p.path)
	if err != nil {
		t.Fatalf("expected permissions file to be written, got error: %v", err)
	}
	var saved PermissionsConfig
	if err := json.Unmarshal(data, &saved); err != nil {
		t.Fatalf("saved permissions file is not valid JSON: %v", err)
	}
	if len(saved.Rules) == 0 {
		t.Error("expected at least one rule in saved permissions file")
	}
}

func TestAllowToolWithArgs(t *testing.T) {
	tmpDir := t.TempDir()

	p := &Permissions{
		path: filepath.Join(tmpDir, ".tanrenai", "permissions.json"),
	}

	err := p.AllowToolWithArgs("shell_exec", map[string][]string{
		"command": {"git *", "ls *"},
	})
	if err != nil {
		t.Fatalf("AllowToolWithArgs returned error: %v", err)
	}

	if !p.IsAllowed("shell_exec", `{"command": "git status"}`) {
		t.Error("expected 'git status' to be allowed after AllowToolWithArgs")
	}
	if !p.IsAllowed("shell_exec", `{"command": "ls -la /tmp"}`) {
		t.Error("expected 'ls -la /tmp' to be allowed after AllowToolWithArgs")
	}
	if p.IsAllowed("shell_exec", `{"command": "curl evil.com"}`) {
		t.Error("expected 'curl evil.com' to NOT be allowed")
	}
}

func TestExtractArgBasic(t *testing.T) {
	argsJSON := `{"path": "/tmp/file.txt", "content": "hello"}`
	got := ExtractArg(argsJSON, "path")
	if got != "/tmp/file.txt" {
		t.Errorf("ExtractArg(path) = %q, want %q", got, "/tmp/file.txt")
	}
}

func TestExtractArgMissingKey(t *testing.T) {
	argsJSON := `{"path": "/tmp/file.txt"}`
	got := ExtractArg(argsJSON, "content")
	if got != "" {
		t.Errorf("ExtractArg for missing key = %q, want empty string", got)
	}
}

func TestExtractArgEmptyKey(t *testing.T) {
	argsJSON := `{"path": "/tmp/file.txt"}`
	got := ExtractArg(argsJSON, "")
	if got != "" {
		t.Errorf("ExtractArg with empty key = %q, want empty string", got)
	}
}

func TestExtractArgEmptyJSON(t *testing.T) {
	got := ExtractArg("", "path")
	if got != "" {
		t.Errorf("ExtractArg with empty JSON = %q, want empty string", got)
	}
}

func TestExtractArgInvalidJSON(t *testing.T) {
	got := ExtractArg("{{{bad json", "path")
	if got != "" {
		t.Errorf("ExtractArg with invalid JSON = %q, want empty string", got)
	}
}

func TestApprovalKeyAllTools(t *testing.T) {
	tests := []struct {
		tool string
		want string
	}{
		{"shell_exec", "command"},
		{"file_write", "path"},
		{"patch_file", "path"},
		{"file_read", "path"},
		{"list_dir", ""},
		{"find_files", ""},
		{"grep_search", ""},
		{"web_search", ""},
		{"unknown_tool", ""},
	}

	for _, tt := range tests {
		got := ApprovalKey(tt.tool)
		if got != tt.want {
			t.Errorf("ApprovalKey(%q) = %q, want %q", tt.tool, got, tt.want)
		}
	}
}
