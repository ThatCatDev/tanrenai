package tools

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
)

const maxFileReadBytes = 32 * 1024 // 32KB

// FileReadTool reads file contents.
type FileReadTool struct{}

type fileReadArgs struct {
	Path string `json:"path"`
}

func (t *FileReadTool) Name() string { return "file_read" }

func (t *FileReadTool) Description() string {
	return "Read the contents of a file at the given path. Returns the file content as text. Use actual filesystem paths like \"./file.txt\" or \"/home/user/file.txt\"."
}

func (t *FileReadTool) Parameters() json.RawMessage {
	return Schema{
		Type: "object",
		Properties: map[string]SchemaProperty{
			"path": {Type: "string", Description: "Absolute or relative path to the file to read"},
		},
		Required: []string{"path"},
	}.MustMarshal()
}

func (t *FileReadTool) Execute(_ context.Context, arguments string) (*ToolResult, error) {
	var args fileReadArgs
	if err := json.Unmarshal([]byte(arguments), &args); err != nil {
		return ErrorResult(fmt.Sprintf("invalid arguments: %v", err)), nil
	}
	if args.Path == "" {
		return ErrorResult("path is required"), nil
	}

	data, err := os.ReadFile(args.Path)
	if err != nil {
		return ErrorResult(fmt.Sprintf("failed to read file: %v", err)), nil
	}

	output := string(data)
	if len(data) > maxFileReadBytes {
		output = string(data[:maxFileReadBytes]) + fmt.Sprintf("\n\n[truncated: file is %d bytes, showing first %d]", len(data), maxFileReadBytes)
	}

	return &ToolResult{Output: output}, nil
}
