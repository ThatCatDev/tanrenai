package memory

import "errors"

var (
	ErrNotFound = errors.New("memory entry not found")
	ErrEmpty    = errors.New("empty query")
)
