package tools

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"strings"
)

// PatchFileTool performs a targeted find-and-replace edit on an existing file.
type PatchFileTool struct{}

type patchFileArgs struct {
	Path      string `json:"path"`
	OldString string `json:"old_string"`
	NewString string `json:"new_string"`
}

func (t *PatchFileTool) Name() string { return "patch_file" }

func (t *PatchFileTool) Description() string {
	return "Edit an existing file by replacing a specific string. Provide the exact text to find (old_string) and its replacement (new_string). The old_string must match exactly one location in the file. Use file_read first to see the current content."
}

func (t *PatchFileTool) Parameters() json.RawMessage {
	return Schema{
		Type: "object",
		Properties: map[string]SchemaProperty{
			"path":       {Type: "string", Description: "Path to the file to edit"},
			"old_string": {Type: "string", Description: "The exact text to find in the file (must match exactly once)"},
			"new_string": {Type: "string", Description: "The text to replace old_string with"},
		},
		Required: []string{"path", "old_string", "new_string"},
	}.MustMarshal()
}

func (t *PatchFileTool) Execute(_ context.Context, arguments string) (*ToolResult, error) {
	var args patchFileArgs
	if err := json.Unmarshal([]byte(arguments), &args); err != nil {
		return ErrorResult(fmt.Sprintf("invalid arguments: %v", err)), nil
	}
	if args.Path == "" {
		return ErrorResult("path is required"), nil
	}
	if args.OldString == "" {
		return ErrorResult("old_string is required"), nil
	}
	if args.OldString == args.NewString {
		return ErrorResult("old_string and new_string are identical — nothing to change"), nil
	}

	content, err := os.ReadFile(args.Path)
	if err != nil {
		if os.IsNotExist(err) {
			return ErrorResult(fmt.Sprintf("file not found: %s", args.Path)), nil
		}
		return ErrorResult(fmt.Sprintf("failed to read file: %v", err)), nil
	}

	fileStr := string(content)
	count := strings.Count(fileStr, args.OldString)

	switch count {
	case 0:
		// Show a snippet of the file to help the LLM correct its call
		snippet := fileStr
		if len(snippet) > 500 {
			snippet = snippet[:500] + "\n...(truncated)"
		}
		return ErrorResult(fmt.Sprintf("old_string not found in %s. File content starts with:\n%s", args.Path, snippet)), nil

	case 1:
		// Exactly one match — perform the replacement
		newContent := strings.Replace(fileStr, args.OldString, args.NewString, 1)
		if err := os.WriteFile(args.Path, []byte(newContent), 0644); err != nil {
			return ErrorResult(fmt.Sprintf("failed to write file: %v", err)), nil
		}
		return &ToolResult{Output: fmt.Sprintf("Replaced %d characters with %d characters in %s",
			len(args.OldString), len(args.NewString), args.Path)}, nil

	default:
		return ErrorResult(fmt.Sprintf("old_string matches %d locations in %s. Include more surrounding context in old_string to make it unique.", count, args.Path)), nil
	}
}
