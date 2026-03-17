package memory

import (
	"strings"
	"testing"
)

func TestEntryContent(t *testing.T) {
	e := &Entry{
		UserMsg:   "hello",
		AssistMsg: "world",
	}

	got := e.Content()
	want := "User: hello\nAssistant: world"
	if got != want {
		t.Errorf("Content() = %q, want %q", got, want)
	}
}

func TestEntryContentEmpty(t *testing.T) {
	e := &Entry{}

	got := e.Content()
	want := "User: \nAssistant: "
	if got != want {
		t.Errorf("Content() = %q, want %q", got, want)
	}
}

func TestEntryContentLongUserMsg(t *testing.T) {
	// Craft a message whose combined content exceeds 1200 chars
	longUser := strings.Repeat("a", 1200)
	e := &Entry{
		UserMsg:   longUser,
		AssistMsg: "short reply",
	}

	got := e.Content()
	if len(got) > 1200 {
		t.Errorf("Content() length = %d, want <= 1200", len(got))
	}
}

func TestEntryContentLongAssistMsg(t *testing.T) {
	longAssist := strings.Repeat("b", 1200)
	e := &Entry{
		UserMsg:   "short question",
		AssistMsg: longAssist,
	}

	got := e.Content()
	if len(got) > 1200 {
		t.Errorf("Content() length = %d, want <= 1200", len(got))
	}
}

func TestEntryContentExactly1200(t *testing.T) {
	// Make content that is exactly 1200 chars — should not be truncated
	prefix := "User: \nAssistant: " // 18 chars
	fill := strings.Repeat("x", 1200-len(prefix))
	e := &Entry{
		AssistMsg: fill,
	}

	got := e.Content()
	if len(got) != 1200 {
		t.Errorf("Content() length = %d, want exactly 1200", len(got))
	}
}

func TestEntryContentTruncationAt1200(t *testing.T) {
	// A very long combined message should be truncated to exactly 1200 chars
	e := &Entry{
		UserMsg:   strings.Repeat("u", 700),
		AssistMsg: strings.Repeat("a", 700),
	}

	got := e.Content()
	if len(got) != 1200 {
		t.Errorf("Content() length = %d, want 1200 (truncated)", len(got))
	}
}

func TestEntryContentUnderLimit(t *testing.T) {
	// Content well under 1200 chars should not be modified
	e := &Entry{
		UserMsg:   "short",
		AssistMsg: "also short",
	}

	got := e.Content()
	want := "User: short\nAssistant: also short"
	if got != want {
		t.Errorf("Content() = %q, want %q", got, want)
	}
	if len(got) >= 1200 {
		t.Errorf("short content should not be truncated: len=%d", len(got))
	}
}

func TestEntryContentPrefixStructure(t *testing.T) {
	e := &Entry{
		UserMsg:   "my question",
		AssistMsg: "my answer",
	}

	got := e.Content()
	if !strings.HasPrefix(got, "User: ") {
		t.Errorf("Content() should start with 'User: ', got %q", got)
	}
	if !strings.Contains(got, "\nAssistant: ") {
		t.Errorf("Content() should contain '\\nAssistant: ', got %q", got)
	}
}
