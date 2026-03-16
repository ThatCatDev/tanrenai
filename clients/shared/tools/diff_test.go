package tools

import (
	"strings"
	"testing"
)

func TestGenerateUnifiedDiffIdentical(t *testing.T) {
	result := GenerateUnifiedDiff("file.txt", "hello\nworld\n", "hello\nworld\n")
	if result != "" {
		t.Errorf("expected empty diff for identical content, got: %q", result)
	}
}

func TestGenerateUnifiedDiffChanged(t *testing.T) {
	old := "line one\nline two\nline three\n"
	new := "line one\nline TWO\nline three\n"
	result := GenerateUnifiedDiff("file.txt", old, new)
	if result == "" {
		t.Fatal("expected non-empty diff for changed content, got empty string")
	}
	if !strings.Contains(result, "-line two") {
		t.Errorf("expected diff to contain '-line two', got: %s", result)
	}
	if !strings.Contains(result, "+line TWO") {
		t.Errorf("expected diff to contain '+line TWO', got: %s", result)
	}
	if !strings.Contains(result, "a/file.txt") {
		t.Errorf("expected diff to contain 'a/file.txt', got: %s", result)
	}
	if !strings.Contains(result, "b/file.txt") {
		t.Errorf("expected diff to contain 'b/file.txt', got: %s", result)
	}
}

func TestGenerateUnifiedDiffBinary(t *testing.T) {
	// Content with null bytes is treated as binary.
	old := "hello\x00world"
	new := "goodbye\x00world"
	result := GenerateUnifiedDiff("binary.bin", old, new)
	if result != "Binary file differs" {
		t.Errorf("expected 'Binary file differs' for binary content, got: %q", result)
	}
}

func TestGenerateUnifiedDiffBinaryNewContent(t *testing.T) {
	// Null byte only in new content.
	old := "plain text"
	new := "binary\x00data"
	result := GenerateUnifiedDiff("file.bin", old, new)
	if result != "Binary file differs" {
		t.Errorf("expected 'Binary file differs' when new content has null bytes, got: %q", result)
	}
}

func TestDiffStatsEmpty(t *testing.T) {
	added, removed := DiffStats("")
	if added != 0 {
		t.Errorf("expected 0 added lines for empty diff, got %d", added)
	}
	if removed != 0 {
		t.Errorf("expected 0 removed lines for empty diff, got %d", removed)
	}
}

func TestDiffStatsWithChanges(t *testing.T) {
	// Construct a unified diff manually.
	diff := `--- a/file.txt
+++ b/file.txt
@@ -1,3 +1,4 @@
 context line
-removed line one
-removed line two
+added line one
+added line two
+added line three
 another context line
`
	added, removed := DiffStats(diff)
	if added != 3 {
		t.Errorf("expected 3 added lines, got %d", added)
	}
	if removed != 2 {
		t.Errorf("expected 2 removed lines, got %d", removed)
	}
}

func TestDiffStatsIgnoresHeaders(t *testing.T) {
	// The +++ and --- header lines should not be counted.
	diff := `--- a/old.txt
+++ b/new.txt
@@ -1,1 +1,1 @@
-old value
+new value
`
	added, removed := DiffStats(diff)
	if added != 1 {
		t.Errorf("expected 1 added line (excluding +++ header), got %d", added)
	}
	if removed != 1 {
		t.Errorf("expected 1 removed line (excluding --- header), got %d", removed)
	}
}
