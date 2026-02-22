package tools

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"os/exec"
	"time"
)

const (
	defaultTimeout   = 30 * time.Second
	maxTimeout       = 120 * time.Second
	maxShellOutput   = 64 * 1024 // 64KB
)

// ShellExecTool runs a shell command.
type ShellExecTool struct{}

type shellExecArgs struct {
	Command        string `json:"command"`
	TimeoutSeconds int    `json:"timeout_seconds,omitempty"`
}

func (t *ShellExecTool) Name() string { return "shell_exec" }

func (t *ShellExecTool) Description() string {
	return "Execute a shell command and return its output. Captures both stdout and stderr. Runs in the current working directory. Example: \"ls -la\" to list files, \"pwd\" to show current directory."
}

func (t *ShellExecTool) Parameters() json.RawMessage {
	return Schema{
		Type: "object",
		Properties: map[string]SchemaProperty{
			"command":         {Type: "string", Description: "The shell command to execute"},
			"timeout_seconds": {Type: "integer", Description: "Timeout in seconds (default 30, max 120)"},
		},
		Required: []string{"command"},
	}.MustMarshal()
}

func (t *ShellExecTool) Execute(ctx context.Context, arguments string) (*ToolResult, error) {
	var args shellExecArgs
	if err := json.Unmarshal([]byte(arguments), &args); err != nil {
		return ErrorResult(fmt.Sprintf("invalid arguments: %v", err)), nil
	}
	if args.Command == "" {
		return ErrorResult("command is required"), nil
	}

	timeout := defaultTimeout
	if args.TimeoutSeconds > 0 {
		timeout = time.Duration(args.TimeoutSeconds) * time.Second
		if timeout > maxTimeout {
			timeout = maxTimeout
		}
	}

	ctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	cmd := exec.CommandContext(ctx, "sh", "-c", args.Command)
	var buf bytes.Buffer
	cmd.Stdout = &buf
	cmd.Stderr = &buf

	err := cmd.Run()

	output := buf.String()
	if len(output) > maxShellOutput {
		output = output[:maxShellOutput] + fmt.Sprintf("\n\n[truncated: output was %d bytes, showing first %d]", len(buf.String()), maxShellOutput)
	}

	if err != nil {
		if ctx.Err() == context.DeadlineExceeded {
			return ErrorResult(fmt.Sprintf("command timed out after %s\n\n%s", timeout, output)), nil
		}
		return ErrorResult(fmt.Sprintf("command failed: %v\n\n%s", err, output)), nil
	}

	if output == "" {
		output = "(no output)"
	}

	return &ToolResult{Output: output}, nil
}
