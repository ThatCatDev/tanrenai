package tools

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"os/exec"
	"runtime"
	"strings"
	"time"
)

const (
	defaultTimeout    = 30 * time.Second
	maxTimeout        = 120 * time.Second
	maxShellOutput    = 64 * 1024 // 64KB
	backgroundPromote = 5 * time.Second
)

// ShellExecTool runs a shell command.
type ShellExecTool struct {
	ProcessManager *ProcessManager
}

type shellExecArgs struct {
	Command        string `json:"command"`
	TimeoutSeconds int    `json:"timeout_seconds,omitempty"`
	Background     *bool  `json:"background,omitempty"`
}

func (t *ShellExecTool) Name() string { return "shell_exec" }

func (t *ShellExecTool) Description() string {
	return "Execute a shell command and return its output. Captures both stdout and stderr. Commands that take longer than 5 seconds are automatically promoted to background processes. Use \"background\": true/false to override. Example: \"ls -la\" to list files, \"npm start\" to start a dev server."
}

func (t *ShellExecTool) Parameters() json.RawMessage {
	return Schema{
		Type: "object",
		Properties: map[string]SchemaProperty{
			"command":         {Type: "string", Description: "The shell command to execute"},
			"timeout_seconds": {Type: "integer", Description: "Timeout in seconds (default 30, max 120). Ignored for background processes."},
			"background":      {Type: "boolean", Description: "Force background (true) or foreground (false) execution. Default: auto-promote after 5s."},
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

	command := strings.TrimSuffix(strings.TrimSpace(args.Command), "&")
	command = strings.TrimSpace(command)

	// No process manager → legacy foreground-only path
	if t.ProcessManager == nil {
		return t.runLegacy(ctx, command, args.TimeoutSeconds)
	}

	// Explicit background=true → start and return immediately
	if args.Background != nil && *args.Background {
		return t.runBackground(command)
	}

	// Explicit background=false → wait with timeout, never promote
	if args.Background != nil && !*args.Background {
		return t.runWait(ctx, command, args.TimeoutSeconds)
	}

	// Auto mode: spawn, wait up to 5s, promote if still running
	return t.runAutoPromote(ctx, command)
}

// runAutoPromote spawns the command via ProcessManager. If it finishes within
// backgroundPromote (5s), returns the output and removes it from the manager.
// If still running, leaves it managed and returns background status.
func (t *ShellExecTool) runAutoPromote(ctx context.Context, command string) (*ToolResult, error) {
	p, err := t.ProcessManager.Start(command)
	if err != nil {
		return ErrorResult(fmt.Sprintf("command failed to start: %v", err)), nil
	}

	done := t.ProcessManager.Done(p.ID)

	timer := time.NewTimer(backgroundPromote)
	defer timer.Stop()

	select {
	case <-done:
		// Finished in time — collect output, remove from manager
		output := t.ProcessManager.Output(p.ID)
		exitCode := p.ExitCode
		// Re-read exit code from snapshot since p may be stale
		for _, snap := range t.ProcessManager.List() {
			if snap.ID == p.ID {
				exitCode = snap.ExitCode

				break
			}
		}
		t.ProcessManager.Remove(p.ID)

		if len(output) > maxShellOutput {
			output = output[:maxShellOutput] + "\n\n[truncated]"
		}
		if exitCode != 0 {
			return ErrorResult(fmt.Sprintf("command failed (exit code %d)\n\n%s", exitCode, output)), nil
		}
		if output == "" {
			output = "(no output)"
		}

		return &ToolResult{Output: output}, nil

	case <-timer.C:
		// Still running — leave in process manager
		output := t.ProcessManager.Output(p.ID)
		msg := fmt.Sprintf("Command still running after %s — promoted to background (ID: %d, PID: %d): %s",
			backgroundPromote, p.ID, p.PID, command)
		if output != "" {
			msg += fmt.Sprintf("\n\nOutput so far:\n%s", output)
		}

		return &ToolResult{Output: msg}, nil

	case <-ctx.Done():
		_ = t.ProcessManager.Kill(p.ID)
		output := t.ProcessManager.Output(p.ID)
		t.ProcessManager.Remove(p.ID)

		return ErrorResult(fmt.Sprintf("cancelled\n\n%s", output)), nil
	}
}

// runBackground starts a command and returns immediately with background info.
func (t *ShellExecTool) runBackground(command string) (*ToolResult, error) {
	p, err := t.ProcessManager.Start(command)
	if err != nil {
		return ErrorResult(fmt.Sprintf("failed to start background process: %v", err)), nil
	}

	// Wait briefly for early crash
	timer := time.NewTimer(500 * time.Millisecond)
	select {
	case <-t.ProcessManager.Done(p.ID):
		timer.Stop()
		output := t.ProcessManager.Output(p.ID)
		// Check exit code from list
		exitCode := 0
		for _, snap := range t.ProcessManager.List() {
			if snap.ID == p.ID {
				exitCode = snap.ExitCode

				break
			}
		}
		if exitCode != 0 {
			t.ProcessManager.Remove(p.ID)

			return ErrorResult(fmt.Sprintf("process exited immediately with code %d\n\n%s", exitCode, output)), nil
		}
		t.ProcessManager.Remove(p.ID)
		if output == "" {
			output = "(no output)"
		}

		return &ToolResult{Output: fmt.Sprintf("Process completed (exit code 0):\n%s", output)}, nil
	case <-timer.C:
		output := t.ProcessManager.Output(p.ID)
		msg := fmt.Sprintf("Started background process (ID: %d, PID: %d): %s", p.ID, p.PID, command)
		if output != "" {
			msg += fmt.Sprintf("\n\nInitial output:\n%s", output)
		}

		return &ToolResult{Output: msg}, nil
	}
}

// runWait spawns via ProcessManager but waits with a full timeout (never promotes).
func (t *ShellExecTool) runWait(ctx context.Context, command string, timeoutSeconds int) (*ToolResult, error) {
	timeout := defaultTimeout
	if timeoutSeconds > 0 {
		timeout = time.Duration(timeoutSeconds) * time.Second
		if timeout > maxTimeout {
			timeout = maxTimeout
		}
	}

	p, err := t.ProcessManager.Start(command)
	if err != nil {
		return ErrorResult(fmt.Sprintf("command failed to start: %v", err)), nil
	}

	done := t.ProcessManager.Done(p.ID)

	timer := time.NewTimer(timeout)
	defer timer.Stop()

	select {
	case <-done:
		output := t.ProcessManager.Output(p.ID)
		exitCode := 0
		for _, snap := range t.ProcessManager.List() {
			if snap.ID == p.ID {
				exitCode = snap.ExitCode

				break
			}
		}
		t.ProcessManager.Remove(p.ID)

		if len(output) > maxShellOutput {
			output = output[:maxShellOutput] + "\n\n[truncated]"
		}
		if exitCode != 0 {
			return ErrorResult(fmt.Sprintf("command failed (exit code %d)\n\n%s", exitCode, output)), nil
		}
		if output == "" {
			output = "(no output)"
		}

		return &ToolResult{Output: output}, nil

	case <-timer.C:
		_ = t.ProcessManager.Kill(p.ID)
		output := t.ProcessManager.Output(p.ID)
		t.ProcessManager.Remove(p.ID)

		return ErrorResult(fmt.Sprintf("command timed out after %s\n\n%s", timeout, output)), nil

	case <-ctx.Done():
		_ = t.ProcessManager.Kill(p.ID)
		output := t.ProcessManager.Output(p.ID)
		t.ProcessManager.Remove(p.ID)

		return ErrorResult(fmt.Sprintf("cancelled\n\n%s", output)), nil
	}
}

// runLegacy is the fallback when no ProcessManager is available.
func (t *ShellExecTool) runLegacy(ctx context.Context, command string, timeoutSeconds int) (*ToolResult, error) {
	timeout := defaultTimeout
	if timeoutSeconds > 0 {
		timeout = time.Duration(timeoutSeconds) * time.Second
		if timeout > maxTimeout {
			timeout = maxTimeout
		}
	}

	ctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	var cmd *exec.Cmd
	if runtime.GOOS == "windows" {
		cmd = exec.CommandContext(ctx, "cmd", "/C", command)
	} else {
		cmd = exec.CommandContext(ctx, "sh", "-c", command)
	}
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
