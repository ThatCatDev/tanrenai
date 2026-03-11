package ui

import (
	"strings"
	"testing"
)

func TestMarkdownToPango(t *testing.T) {
	tests := []struct {
		name string
		md   string
		want string // substring that must appear
	}{
		{"plain text", "hello world", "hello world"},
		{"bold", "**bold text**", "<b>bold text</b>"},
		{"inline code", "`code here`", "<tt>code here</tt>"},
		{"code block", "```go\nfmt.Println()\n```", "<tt>fmt.Println()</tt>"},
		{"escapes ampersand", "a & b", "a &amp; b"},
		{"escapes angle brackets", "a < b > c", "a &lt; b &gt; c"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := markdownToPango(tt.md)
			if !strings.Contains(got, tt.want) {
				t.Errorf("markdownToPango(%q) = %q, want substring %q", tt.md, got, tt.want)
			}
		})
	}
}

func TestPangoEscape(t *testing.T) {
	got := pangoEscape(`a & b < c > d`)
	want := "a &amp; b &lt; c &gt; d"
	if got != want {
		t.Errorf("pangoEscape = %q, want %q", got, want)
	}
}

func TestPlainTextSummary(t *testing.T) {
	tests := []struct {
		name   string
		input  string
		maxLen int
		want   string
	}{
		{"short", "hello", 10, "hello"},
		{"strips bold", "**bold** text", 20, "bold text"},
		{"strips inline code", "`code` here", 20, "code here"},
		{"strips newlines", "line1\nline2", 20, "line1 line2"},
		{"truncates", "a very long string that should be truncated", 10, "a very lon..."},
		{"empty", "", 10, ""},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := plainTextSummary(tt.input, tt.maxLen)
			if got != tt.want {
				t.Errorf("plainTextSummary(%q, %d) = %q, want %q", tt.input, tt.maxLen, got, tt.want)
			}
		})
	}
}
