package tools

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
)

// FileWriteTool writes or creates a file.
type FileWriteTool struct{}

type fileWriteArgs struct {
	Path    string `json:"path"`
	Content string `json:"content"`
}

func (t *FileWriteTool) Name() string { return "file_write" }

func (t *FileWriteTool) Description() string {
	return "Write content to a file at the given path. Creates the file and any parent directories if they don't exist. Overwrites existing content."
}

func (t *FileWriteTool) Parameters() json.RawMessage {
	return Schema{
		Type: "object",
		Properties: map[string]SchemaProperty{
			"path":    {Type: "string", Description: "Absolute or relative path to the file to write"},
			"content": {Type: "string", Description: "Content to write to the file"},
		},
		Required: []string{"path", "content"},
	}.MustMarshal()
}

func (t *FileWriteTool) Execute(_ context.Context, arguments string) (*ToolResult, error) {
	var args fileWriteArgs
	if err := json.Unmarshal([]byte(arguments), &args); err != nil {
		return ErrorResult(fmt.Sprintf("invalid arguments: %v", err)), nil
	}
	if args.Path == "" {
		return ErrorResult("path is required"), nil
	}

	dir := filepath.Dir(args.Path)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return ErrorResult(fmt.Sprintf("failed to create directories: %v", err)), nil
	}

	if err := os.WriteFile(args.Path, []byte(args.Content), 0644); err != nil {
		return ErrorResult(fmt.Sprintf("failed to write file: %v", err)), nil
	}

	return &ToolResult{Output: fmt.Sprintf("Successfully wrote %d bytes to %s", len(args.Content), args.Path)}, nil
}
