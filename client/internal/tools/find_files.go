package tools

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

const maxFindResults = 200

// FindFilesTool finds files by name pattern.
type FindFilesTool struct{}

type findFilesArgs struct {
	Pattern    string `json:"pattern"`
	Path       string `json:"path"`
	MaxResults int    `json:"max_results"`
}

func (t *FindFilesTool) Name() string { return "find_files" }

func (t *FindFilesTool) Description() string {
	return "Find files by name pattern. Searches recursively from the given path. Pattern uses glob syntax (e.g. \"*.go\", \"*_test.go\", \"Makefile\"). Returns matching file paths."
}

func (t *FindFilesTool) Parameters() json.RawMessage {
	return Schema{
		Type: "object",
		Properties: map[string]SchemaProperty{
			"pattern":     {Type: "string", Description: "Glob pattern to match filenames (e.g. \"*.go\", \"*.py\", \"Dockerfile*\")"},
			"path":        {Type: "string", Description: "Directory to search in (default: \".\")"},
			"max_results": {Type: "integer", Description: "Maximum number of results (default: 100)"},
		},
		Required: []string{"pattern"},
	}.MustMarshal()
}

func (t *FindFilesTool) Execute(_ context.Context, arguments string) (*ToolResult, error) {
	var args findFilesArgs
	if err := json.Unmarshal([]byte(arguments), &args); err != nil {
		return ErrorResult(fmt.Sprintf("invalid arguments: %v", err)), nil
	}
	if args.Pattern == "" {
		return ErrorResult("pattern is required"), nil
	}
	if args.Path == "" {
		args.Path = "."
	}
	if args.MaxResults <= 0 {
		args.MaxResults = 100
	}
	if args.MaxResults > maxFindResults {
		args.MaxResults = maxFindResults
	}

	var b strings.Builder
	count := 0

	err := filepath.Walk(args.Path, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return nil
		}
		if count >= args.MaxResults {
			return filepath.SkipAll
		}
		if info.IsDir() {
			name := info.Name()
			if name == ".git" || name == "node_modules" || name == "vendor" || name == "__pycache__" || name == ".venv" || name == "venv" {
				return filepath.SkipDir
			}
			return nil
		}

		matched, _ := filepath.Match(args.Pattern, info.Name())
		if matched {
			fmt.Fprintf(&b, "%s\n", path)
			count++
		}
		return nil
	})

	if err != nil {
		return ErrorResult(fmt.Sprintf("search failed: %v", err)), nil
	}

	if count == 0 {
		return &ToolResult{Output: fmt.Sprintf("No files found matching %q in %s", args.Pattern, args.Path)}, nil
	}

	output := b.String()
	if count >= args.MaxResults {
		output += fmt.Sprintf("[showing first %d results]", args.MaxResults)
	}

	return &ToolResult{Output: output}, nil
}
