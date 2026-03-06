package ui

import (
	"regexp"
	"strings"
)

var (
	// Match fenced code blocks: ```lang\n...\n```
	codeBlockRe = regexp.MustCompile("(?s)```(\\w*)\\n(.*?)\\n```")
	// Match inline code: `code`
	inlineCodeRe = regexp.MustCompile("`([^`]+)`")
	// Match bold: **text**
	boldRe = regexp.MustCompile(`\*\*(.+?)\*\*`)
	// Match italic: *text*
	italicRe = regexp.MustCompile(`(?:^|[^*])\*([^*]+)\*(?:[^*]|$)`)
)

// markdownToPango converts basic markdown to Pango markup for GTK labels.
func markdownToPango(md string) string {
	// Escape XML entities first
	s := pangoEscape(md)

	// Replace code blocks with monospace
	s = codeBlockRe.ReplaceAllStringFunc(s, func(match string) string {
		parts := codeBlockRe.FindStringSubmatch(match)
		if len(parts) < 3 {
			return match
		}
		return "<tt>" + parts[2] + "</tt>"
	})

	// Replace inline code
	s = inlineCodeRe.ReplaceAllString(s, "<tt>$1</tt>")

	// Replace bold
	s = boldRe.ReplaceAllString(s, "<b>$1</b>")

	// Replace italic (simple version)
	s = italicRe.ReplaceAllStringFunc(s, func(match string) string {
		// Extract the content between * markers, preserving surrounding chars
		parts := italicRe.FindStringSubmatch(match)
		if len(parts) < 2 {
			return match
		}
		prefix := ""
		suffix := ""
		if len(match) > 0 && match[0] != '*' {
			prefix = string(match[0])
		}
		if len(match) > 0 && match[len(match)-1] != '*' {
			suffix = string(match[len(match)-1])
		}
		return prefix + "<i>" + parts[1] + "</i>" + suffix
	})

	return s
}

// pangoEscape escapes special XML characters for Pango markup.
func pangoEscape(s string) string {
	s = strings.ReplaceAll(s, "&", "&amp;")
	s = strings.ReplaceAll(s, "<", "&lt;")
	s = strings.ReplaceAll(s, ">", "&gt;")
	return s
}

// plainTextSummary returns a short plain-text summary of content.
func plainTextSummary(s string, maxLen int) string {
	// Strip markdown formatting
	s = strings.ReplaceAll(s, "**", "")
	s = strings.ReplaceAll(s, "*", "")
	s = strings.ReplaceAll(s, "`", "")
	s = strings.ReplaceAll(s, "\n", " ")
	s = strings.TrimSpace(s)
	if len(s) > maxLen {
		s = s[:maxLen] + "..."
	}
	return s
}
