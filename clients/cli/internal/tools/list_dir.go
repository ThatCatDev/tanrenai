package tools

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"unicode"
)

// ListDirTool lists directory contents.
type ListDirTool struct{}

type listDirArgs struct {
	Path  string `json:"path"`
	Depth *int   `json:"depth"`
}

func (t *ListDirTool) Name() string { return "list_dir" }

func (t *ListDirTool) Description() string {
	return "List the contents of a directory. Returns file and directory names with type indicators. Use \".\" for the current working directory. Set depth > 1 to recurse into subdirectories."
}

func (t *ListDirTool) Parameters() json.RawMessage {
	return Schema{
		Type: "object",
		Properties: map[string]SchemaProperty{
			"path":  {Type: "string", Description: "Absolute or relative path to the directory. Use \".\" for current directory."},
			"depth": {Type: "integer", Description: "How many levels deep to recurse. Default: 2. Use 1 for immediate children only, 0 for unlimited."},
		},
		Required: []string{"path"},
	}.MustMarshal()
}

func (t *ListDirTool) Execute(_ context.Context, arguments string) (*ToolResult, error) {
	var args listDirArgs
	if err := json.Unmarshal([]byte(arguments), &args); err != nil {
		return ErrorResult(fmt.Sprintf("invalid arguments: %v", err)), nil
	}

	// Default to current directory
	if args.Path == "" {
		args.Path = "."
	}

	// Resolve depth: nil (not specified) → 2 for a useful default.
	// Explicit 0 means unlimited (capped at 50). Negative → 1.
	depth := 2
	if args.Depth != nil {
		depth = *args.Depth
		if depth < 0 {
			depth = 1
		}
		if depth == 0 {
			depth = 50
		}
	}

	// If the path doesn't exist and looks like a natural-language placeholder,
	// fall back to "." so the model still gets useful output.
	if _, err := os.Stat(args.Path); os.IsNotExist(err) && !isRealPath(args.Path) {
		args.Path = "."
	}

	var b strings.Builder
	if err := listDirRecursive(&b, args.Path, "", depth); err != nil {
		return ErrorResult(fmt.Sprintf("failed to read directory: %v", err)), nil
	}

	if b.Len() == 0 {
		return &ToolResult{Output: "(empty directory)"}, nil
	}

	return &ToolResult{Output: b.String()}, nil
}

// listDirRecursive writes directory entries to b, recursing up to maxDepth levels.
func listDirRecursive(b *strings.Builder, root, prefix string, remainingDepth int) error {
	entries, err := os.ReadDir(filepath.Join(root, prefix))
	if err != nil {
		return err
	}

	for _, entry := range entries {
		rel := filepath.Join(prefix, entry.Name())
		if entry.IsDir() {
			fmt.Fprintf(b, "[dir]  %s\n", rel)
			if remainingDepth > 1 {
				if err := listDirRecursive(b, root, rel, remainingDepth-1); err != nil {
					// Skip unreadable subdirectories
					continue
				}
			}
		} else {
			fmt.Fprintf(b, "[file] %s\n", rel)
		}
	}
	return nil
}

// isRealPath returns true if s looks like an actual filesystem path
// (starts with /, ./, ../, ~/, or is just "." or "..") rather than a
// natural-language placeholder like "current_directory".
func isRealPath(s string) bool {
	if s == "." || s == ".." {
		return true
	}
	if strings.HasPrefix(s, "/") || strings.HasPrefix(s, "./") || strings.HasPrefix(s, "../") || strings.HasPrefix(s, "~/") {
		return true
	}
	// If it contains spaces or only letters/underscores with no path separators,
	// it's probably a placeholder like "current_directory" or "my files"
	for _, r := range s {
		if r == '/' {
			return true
		}
		if unicode.IsSpace(r) {
			return false
		}
	}
	// Single word with no slashes — could be a real relative dir name,
	// but if it doesn't exist (caller already checked), treat as placeholder
	return false
}
