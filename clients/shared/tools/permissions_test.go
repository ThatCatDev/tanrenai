package tools

import "testing"

func TestMatchPattern(t *testing.T) {
	tests := []struct {
		pattern string
		value   string
		want    bool
	}{
		// Exact match
		{"ls -la", "ls -la", true},
		{"ls -la", "ls -lb", false},

		// Command prefix wildcard
		{"git *", "git status", true},
		{"git *", "git log --oneline", true},
		{"git *", "git", true},
		{"git *", "gitk", false},
		{"ls *", "ls", true},
		{"ls *", "ls -la /tmp", true},

		// Path prefix wildcard
		{"/home/user/project/*", "/home/user/project/file.go", true},
		{"/home/user/project/*", "/home/user/project/sub/file.go", true},
		{"/home/user/project/*", "/home/user/other/file.go", false},
		{"/home/user/project/*", "/home/user/project", true},
	}

	for _, tt := range tests {
		got := matchPattern(tt.pattern, tt.value)
		if got != tt.want {
			t.Errorf("matchPattern(%q, %q) = %v, want %v", tt.pattern, tt.value, got, tt.want)
		}
	}
}

func TestIsAllowed(t *testing.T) {
	p := &Permissions{
		config: PermissionsConfig{
			Rules: []PermissionRule{
				{Tool: "file_read"},
				{Tool: "shell_exec", AllowedArgs: map[string][]string{
					"command": {"git *", "ls *"},
				}},
				{Tool: "file_write", AllowedArgs: map[string][]string{
					"path": {"/tmp/*"},
				}},
			},
		},
	}

	tests := []struct {
		tool string
		args string
		want bool
	}{
		// Blanket allow
		{"file_read", `{"path": "/etc/passwd"}`, true},
		{"file_read", ``, true},

		// shell_exec with allowed prefix
		{"shell_exec", `{"command": "git status"}`, true},
		{"shell_exec", `{"command": "ls -la"}`, true},
		{"shell_exec", `{"command": "rm -rf /"}`, false},
		{"shell_exec", `{"command": "curl evil.com"}`, false},

		// file_write with path prefix
		{"file_write", `{"path": "/tmp/test.txt", "content": "hi"}`, true},
		{"file_write", `{"path": "/etc/shadow", "content": "bad"}`, false},

		// Unknown tool — not allowed
		{"unknown_tool", `{}`, false},
	}

	for _, tt := range tests {
		got := p.IsAllowed(tt.tool, tt.args)
		if got != tt.want {
			t.Errorf("IsAllowed(%q, %q) = %v, want %v", tt.tool, tt.args, got, tt.want)
		}
	}
}

func TestToolRisk(t *testing.T) {
	if ToolRisk("file_read") != RiskReadOnly {
		t.Error("file_read should be RiskReadOnly")
	}
	if ToolRisk("file_write") != RiskWrite {
		t.Error("file_write should be RiskWrite")
	}
	if ToolRisk("shell_exec") != RiskExecute {
		t.Error("shell_exec should be RiskExecute")
	}
	if ToolRisk("web_search") != RiskNetwork {
		t.Error("web_search should be RiskNetwork")
	}
	if ToolRisk("unknown") != RiskExecute {
		t.Error("unknown should default to RiskExecute")
	}
}

func TestCommandPrefix(t *testing.T) {
	tests := []struct {
		cmd  string
		want string
	}{
		{"git status", "git"},
		{"ls -la /tmp", "ls"},
		{"rm -rf /", "rm"},
		{"", ""},
	}
	for _, tt := range tests {
		got := CommandPrefix(tt.cmd)
		if got != tt.want {
			t.Errorf("CommandPrefix(%q) = %q, want %q", tt.cmd, got, tt.want)
		}
	}
}
