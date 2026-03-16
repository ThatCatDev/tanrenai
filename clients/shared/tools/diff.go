package tools

import (
	"strings"

	"github.com/pmezard/go-difflib/difflib"
)

const maxDiffLen = 10000

// GenerateUnifiedDiff produces a unified diff between oldContent and newContent.
// Returns empty string if contents are identical.
func GenerateUnifiedDiff(filename, oldContent, newContent string) string {
	if oldContent == newContent {
		return ""
	}

	// Binary detection: check for null bytes
	if strings.ContainsRune(oldContent, 0) || strings.ContainsRune(newContent, 0) {
		return "Binary file differs"
	}

	diff := difflib.UnifiedDiff{
		A:        difflib.SplitLines(oldContent),
		B:        difflib.SplitLines(newContent),
		FromFile: "a/" + filename,
		ToFile:   "b/" + filename,
		Context:  3,
	}

	text, err := difflib.GetUnifiedDiffString(diff)
	if err != nil {
		return ""
	}

	if len(text) > maxDiffLen {
		text = text[:maxDiffLen] + "\n... (diff truncated)"
	}

	return text
}

// DiffStats counts added and removed lines in a unified diff string.
func DiffStats(diff string) (added, removed int) {
	for _, line := range strings.Split(diff, "\n") {
		if len(line) == 0 {
			continue
		}
		switch line[0] {
		case '+':
			if !strings.HasPrefix(line, "+++") {
				added++
			}
		case '-':
			if !strings.HasPrefix(line, "---") {
				removed++
			}
		}
	}

	return
}
