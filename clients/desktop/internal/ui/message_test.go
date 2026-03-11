package ui

import "testing"

func TestExtractKeyArg(t *testing.T) {
	tests := []struct {
		name     string
		toolName string
		args     string
		want     string
	}{
		{
			name:     "file_read path",
			toolName: "file_read",
			args:     `{"path": "/home/user/file.go"}`,
			want:     "/home/user/file.go",
		},
		{
			name:     "grep_search pattern",
			toolName: "grep_search",
			args:     `{"pattern": "func main", "path": "."}`,
			want:     "func main",
		},
		{
			name:     "shell_exec command",
			toolName: "shell_exec",
			args:     `{"command": "go build ./..."}`,
			want:     "go build ./...",
		},
		{
			name:     "unknown tool with path",
			toolName: "custom_tool",
			args:     `{"path": "/some/path", "other": "val"}`,
			want:     "/some/path",
		},
		{
			name:     "unknown tool no known keys",
			toolName: "custom_tool",
			args:     `{"foo": "bar"}`,
			want:     "",
		},
		{
			name:     "invalid json",
			toolName: "file_read",
			args:     `not json`,
			want:     "",
		},
		{
			name:     "empty args",
			toolName: "file_read",
			args:     `{}`,
			want:     "",
		},
		{
			name:     "web_search query",
			toolName: "web_search",
			args:     `{"query": "golang testing"}`,
			want:     "golang testing",
		},
		{
			name:     "find_files pattern",
			toolName: "find_files",
			args:     `{"pattern": "*.go"}`,
			want:     "*.go",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := extractKeyArg(tt.toolName, tt.args)
			if got != tt.want {
				t.Errorf("extractKeyArg(%q, %q) = %q, want %q", tt.toolName, tt.args, got, tt.want)
			}
		})
	}
}
