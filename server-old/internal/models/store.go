package models

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// Store manages locally available GGUF model files.
type Store struct {
	dir string
}

// NewStore creates a new Store for the given directory.
func NewStore(dir string) *Store {
	return &Store{dir: dir}
}

// List returns all available models by scanning the models directory for .gguf files.
func (s *Store) List() []ModelEntry {
	var entries []ModelEntry

	err := filepath.Walk(s.dir, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return nil
		}
		if info.IsDir() || !strings.HasSuffix(strings.ToLower(info.Name()), ".gguf") {
			return nil
		}
		name := strings.TrimSuffix(info.Name(), filepath.Ext(info.Name()))
		entries = append(entries, ModelEntry{
			Name:       name,
			Path:       path,
			Size:       info.Size(),
			ModifiedAt: info.ModTime().Unix(),
		})
		return nil
	})
	if err != nil {
		// Directory may not exist yet, that's OK
	}

	return entries
}

// Resolve finds a model by name and returns its full path.
// It searches for an exact filename match (with or without .gguf extension),
// or a partial name match.
func (s *Store) Resolve(name string) (string, error) {
	// Try exact path first
	if filepath.IsAbs(name) {
		if _, err := os.Stat(name); err == nil {
			return name, nil
		}
	}

	// Try with .gguf extension in models dir
	candidate := filepath.Join(s.dir, name)
	if _, err := os.Stat(candidate); err == nil {
		return candidate, nil
	}
	candidate = candidate + ".gguf"
	if _, err := os.Stat(candidate); err == nil {
		return candidate, nil
	}

	// Search by partial name match
	entries := s.List()
	for _, e := range entries {
		if strings.EqualFold(e.Name, name) {
			return e.Path, nil
		}
		if strings.Contains(strings.ToLower(e.Name), strings.ToLower(name)) {
			return e.Path, nil
		}
	}

	return "", fmt.Errorf("model %q not found in %s", name, s.dir)
}
