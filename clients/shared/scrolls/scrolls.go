// Package scrolls provides loading and matching of procedural knowledge files
// (guides, runbooks, recipes) stored as markdown with YAML frontmatter.
package scrolls

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// Scroll represents a loaded scroll file.
type Scroll struct {
	Name        string
	Description string
	Tags        []string
	Content     string // full markdown body (after frontmatter)
	Source      string // "project" or "global"
	Path        string // absolute file path
}

// Load globs *.md from both dirs, parses each, and deduplicates by name
// (project scrolls override global scrolls with the same name).
func Load(projectDir, globalDir string) ([]Scroll, error) {
	byName := make(map[string]Scroll)

	// Load global scrolls first so project can override.
	if globalDir != "" {
		if err := loadDir(globalDir, "global", byName); err != nil {
			return nil, fmt.Errorf("global scrolls: %w", err)
		}
	}

	// Project scrolls override global by name.
	if projectDir != "" {
		if err := loadDir(projectDir, "project", byName); err != nil {
			return nil, fmt.Errorf("project scrolls: %w", err)
		}
	}

	scrolls := make([]Scroll, 0, len(byName))
	for _, s := range byName {
		scrolls = append(scrolls, s)
	}
	return scrolls, nil
}

func loadDir(dir, source string, byName map[string]Scroll) error {
	matches, err := filepath.Glob(filepath.Join(dir, "*.md"))
	if err != nil {
		return err
	}
	for _, path := range matches {
		s, err := parse(path, source)
		if err != nil {
			continue // skip malformed scrolls
		}
		byName[s.Name] = *s
	}
	return nil
}

func parse(path, source string) (*Scroll, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}

	content := string(data)

	// Split on frontmatter delimiters (--- on its own line).
	if !strings.HasPrefix(content, "---") {
		return nil, fmt.Errorf("missing frontmatter start in %s", path)
	}

	// Find the closing ---
	rest := content[3:]
	// Skip the newline after opening ---
	if idx := strings.Index(rest, "\n"); idx >= 0 {
		rest = rest[idx+1:]
	}

	endIdx := strings.Index(rest, "\n---")
	if endIdx < 0 {
		return nil, fmt.Errorf("missing frontmatter end in %s", path)
	}

	frontmatter := rest[:endIdx]
	body := strings.TrimLeft(rest[endIdx+4:], "\r\n")

	s := &Scroll{
		Content: body,
		Source:  source,
		Path:    path,
	}

	// Hand-parse the simple YAML frontmatter.
	for _, line := range strings.Split(frontmatter, "\n") {
		line = strings.TrimSpace(line)
		if line == "" || line == "---" {
			continue
		}

		key, val, ok := strings.Cut(line, ":")
		if !ok {
			continue
		}
		key = strings.TrimSpace(key)
		val = strings.TrimSpace(val)

		switch key {
		case "name":
			s.Name = strings.Trim(val, "\"'")
		case "description":
			s.Description = strings.Trim(val, "\"'")
		case "tags":
			s.Tags = parseTags(val)
		}
	}

	if s.Name == "" {
		return nil, fmt.Errorf("scroll missing name in %s", path)
	}
	if s.Description == "" {
		return nil, fmt.Errorf("scroll missing description in %s", path)
	}

	return s, nil
}

// parseTags parses a YAML-style list: [tag1, tag2, tag3] or bare comma-separated.
func parseTags(val string) []string {
	val = strings.TrimPrefix(val, "[")
	val = strings.TrimSuffix(val, "]")
	var tags []string
	for _, t := range strings.Split(val, ",") {
		t = strings.TrimSpace(t)
		t = strings.Trim(t, "\"'")
		if t != "" {
			tags = append(tags, t)
		}
	}
	return tags
}
