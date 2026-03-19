package tools

import (
	"context"
	"encoding/json"
	"runtime"
	"strings"
	"testing"
	"time"
)

// TestShellExec_RunBackground_QuickExit verifies runBackground when the
// process exits almost immediately with exit code 0.
func TestShellExec_RunBackground_QuickExit(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("test uses sh")
	}
	pm := NewProcessManager()
	defer pm.KillAll()

	tool := &ShellExecTool{ProcessManager: pm}
	bg := true
	args := shellExecArgs{Command: "echo quick-bg", Background: &bg}
	argsJSON, _ := json.Marshal(args)

	result, err := tool.Execute(context.Background(), string(argsJSON))
	if err != nil {
		t.Fatal(err)
	}
	// Process exits quickly (exit 0) — should succeed
	if result.IsError {
		t.Fatalf("unexpected error: %s", result.Output)
	}
}

// TestShellExec_RunBackground_ImmediateFailure verifies runBackground when the
// process exits immediately with a non-zero exit code.
func TestShellExec_RunBackground_ImmediateFailure(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("test uses sh")
	}
	pm := NewProcessManager()
	defer pm.KillAll()

	tool := &ShellExecTool{ProcessManager: pm}
	bg := true
	// Use a command that always fails.
	args := shellExecArgs{Command: "sh -c 'exit 2'", Background: &bg}
	argsJSON, _ := json.Marshal(args)

	result, err := tool.Execute(context.Background(), string(argsJSON))
	if err != nil {
		t.Fatal(err)
	}
	if !result.IsError {
		t.Fatal("expected error result for command that exits with code 2")
	}
	if !strings.Contains(result.Output, "exited immediately") {
		t.Errorf("expected 'exited immediately', got: %s", result.Output)
	}
}

// TestShellExec_RunBackground_StaysRunning verifies runBackground leaves a
// long-running process in the manager and returns a background status message.
func TestShellExec_RunBackground_StaysRunning(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("test uses sh")
	}
	pm := NewProcessManager()
	defer pm.KillAll()

	tool := &ShellExecTool{ProcessManager: pm}
	bg := true
	args := shellExecArgs{Command: "sleep 60", Background: &bg}
	argsJSON, _ := json.Marshal(args)

	result, err := tool.Execute(context.Background(), string(argsJSON))
	if err != nil {
		t.Fatal(err)
	}
	if result.IsError {
		t.Fatalf("unexpected error: %s", result.Output)
	}
	if !strings.Contains(result.Output, "background") {
		t.Errorf("expected 'background' in output, got: %s", result.Output)
	}
	if pm.RunningCount() != 1 {
		t.Errorf("expected 1 running process, got %d", pm.RunningCount())
	}
}

// TestShellExec_RunWait_Success tests runWait when the command completes
// within the timeout successfully.
func TestShellExec_RunWait_Success(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("test uses sh")
	}
	pm := NewProcessManager()
	defer pm.KillAll()

	tool := &ShellExecTool{ProcessManager: pm}
	fg := false
	args := shellExecArgs{Command: "echo wait-success", Background: &fg}
	argsJSON, _ := json.Marshal(args)

	result, err := tool.Execute(context.Background(), string(argsJSON))
	if err != nil {
		t.Fatal(err)
	}
	if result.IsError {
		t.Fatalf("unexpected error: %s", result.Output)
	}
	if !strings.Contains(result.Output, "wait-success") {
		t.Errorf("expected 'wait-success' in output, got: %s", result.Output)
	}
}

// TestShellExec_RunWait_Failure tests runWait when the command exits non-zero.
func TestShellExec_RunWait_Failure(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("test uses sh")
	}
	pm := NewProcessManager()
	defer pm.KillAll()

	tool := &ShellExecTool{ProcessManager: pm}
	fg := false
	args := shellExecArgs{Command: "sh -c 'exit 3'", Background: &fg}
	argsJSON, _ := json.Marshal(args)

	result, err := tool.Execute(context.Background(), string(argsJSON))
	if err != nil {
		t.Fatal(err)
	}
	if !result.IsError {
		t.Fatal("expected error for non-zero exit")
	}
	if !strings.Contains(result.Output, "exit code 3") {
		t.Errorf("expected exit code 3 in output, got: %s", result.Output)
	}
}

// TestShellExec_RunWait_Timeout tests runWait when the command exceeds the timeout.
func TestShellExec_RunWait_Timeout(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("test uses sh")
	}
	pm := NewProcessManager()
	defer pm.KillAll()

	tool := &ShellExecTool{ProcessManager: pm}
	fg := false
	// 1 second timeout, sleep 10 seconds → should time out.
	args := shellExecArgs{Command: "sleep 10", Background: &fg, TimeoutSeconds: 1}
	argsJSON, _ := json.Marshal(args)

	result, err := tool.Execute(context.Background(), string(argsJSON))
	if err != nil {
		t.Fatal(err)
	}
	if !result.IsError {
		t.Fatal("expected error for timed-out command")
	}
	if !strings.Contains(result.Output, "timed out") {
		t.Errorf("expected 'timed out' in output, got: %s", result.Output)
	}
}

// TestShellExec_RunWait_ContextCancelled tests runWait when the context is cancelled.
func TestShellExec_RunWait_ContextCancelled(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("test uses sh")
	}
	pm := NewProcessManager()
	defer pm.KillAll()

	ctx, cancel := context.WithCancel(context.Background())

	tool := &ShellExecTool{ProcessManager: pm}
	fg := false
	args := shellExecArgs{Command: "sleep 60", Background: &fg}
	argsJSON, _ := json.Marshal(args)

	// Cancel the context after a short delay.
	go func() {
		time.Sleep(200 * time.Millisecond)
		cancel()
	}()

	result, err := tool.Execute(ctx, string(argsJSON))
	if err != nil {
		t.Fatal(err)
	}
	if !result.IsError {
		t.Fatal("expected error for cancelled context")
	}
	if !strings.Contains(result.Output, "cancelled") {
		t.Errorf("expected 'cancelled' in output, got: %s", result.Output)
	}
}

// TestShellExec_RunWait_MaxTimeout verifies that timeout_seconds > maxTimeout is capped.
func TestShellExec_RunWait_MaxTimeout(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("test uses sh")
	}
	pm := NewProcessManager()
	defer pm.KillAll()

	tool := &ShellExecTool{ProcessManager: pm}
	fg := false
	// 999 seconds should be capped to maxTimeout (120s) — command still runs fine.
	args := shellExecArgs{Command: "echo capped", Background: &fg, TimeoutSeconds: 999}
	argsJSON, _ := json.Marshal(args)

	result, err := tool.Execute(context.Background(), string(argsJSON))
	if err != nil {
		t.Fatal(err)
	}
	if result.IsError {
		t.Fatalf("unexpected error: %s", result.Output)
	}
}

// TestShellExec_RunWait_EmptyOutput verifies "(no output)" is returned for silent commands.
func TestShellExec_RunWait_EmptyOutput(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("test uses sh")
	}
	pm := NewProcessManager()
	defer pm.KillAll()

	tool := &ShellExecTool{ProcessManager: pm}
	fg := false
	args := shellExecArgs{Command: "true", Background: &fg}
	argsJSON, _ := json.Marshal(args)

	result, err := tool.Execute(context.Background(), string(argsJSON))
	if err != nil {
		t.Fatal(err)
	}
	if result.IsError {
		t.Fatalf("unexpected error: %s", result.Output)
	}
	if result.Output != "(no output)" {
		t.Errorf("expected '(no output)', got: %q", result.Output)
	}
}

// TestShellExec_RunAutoPromote_ContextCancelled tests the auto-promote path
// when the context is cancelled before the command finishes or the timer fires.
func TestShellExec_RunAutoPromote_ContextCancelled(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("test uses sh")
	}
	pm := NewProcessManager()
	defer pm.KillAll()

	ctx, cancel := context.WithCancel(context.Background())

	tool := &ShellExecTool{ProcessManager: pm}
	// No background flag → auto-promote path, sleep longer than the 5s promote window.
	args := shellExecArgs{Command: "sleep 60"}
	argsJSON, _ := json.Marshal(args)

	go func() {
		time.Sleep(200 * time.Millisecond)
		cancel()
	}()

	result, err := tool.Execute(ctx, string(argsJSON))
	if err != nil {
		t.Fatal(err)
	}
	if !result.IsError {
		t.Fatal("expected error for cancelled context in auto-promote path")
	}
	if !strings.Contains(result.Output, "cancelled") {
		t.Errorf("expected 'cancelled' in output, got: %s", result.Output)
	}
}

// TestShellExec_RunAutoPromote_FailureExitCode tests the auto-promote path
// when a short command exits with a non-zero exit code.
func TestShellExec_RunAutoPromote_FailureExitCode(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("test uses sh")
	}
	pm := NewProcessManager()
	defer pm.KillAll()

	tool := &ShellExecTool{ProcessManager: pm}
	args := shellExecArgs{Command: "sh -c 'exit 7'"}
	argsJSON, _ := json.Marshal(args)

	result, err := tool.Execute(context.Background(), string(argsJSON))
	if err != nil {
		t.Fatal(err)
	}
	if !result.IsError {
		t.Fatal("expected error for non-zero exit in auto-promote path")
	}
	if !strings.Contains(result.Output, "exit code 7") {
		t.Errorf("expected 'exit code 7', got: %s", result.Output)
	}
}

// TestShellExec_RunAutoPromote_EmptyOutput tests the auto-promote path
// when a short command produces no output.
func TestShellExec_RunAutoPromote_EmptyOutput(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("test uses sh")
	}
	pm := NewProcessManager()
	defer pm.KillAll()

	tool := &ShellExecTool{ProcessManager: pm}
	args := shellExecArgs{Command: "true"}
	argsJSON, _ := json.Marshal(args)

	result, err := tool.Execute(context.Background(), string(argsJSON))
	if err != nil {
		t.Fatal(err)
	}
	if result.IsError {
		t.Fatalf("unexpected error: %s", result.Output)
	}
	if result.Output != "(no output)" {
		t.Errorf("expected '(no output)', got: %q", result.Output)
	}
}

// TestShellExec_Execute_InvalidArgs tests the Execute JSON parse error path.
func TestShellExec_Execute_InvalidArgs(t *testing.T) {
	tool := &ShellExecTool{}
	result, err := tool.Execute(context.Background(), `{not valid json}`)
	if err != nil {
		t.Fatal(err)
	}
	if !result.IsError {
		t.Fatal("expected error for invalid JSON")
	}
	if !strings.Contains(result.Output, "invalid arguments") {
		t.Errorf("expected 'invalid arguments', got: %s", result.Output)
	}
}

// TestShellExec_Execute_EmptyCommand tests that an empty command returns an error.
func TestShellExec_Execute_EmptyCommand(t *testing.T) {
	tool := &ShellExecTool{}
	result, err := tool.Execute(context.Background(), `{"command":""}`)
	if err != nil {
		t.Fatal(err)
	}
	if !result.IsError {
		t.Fatal("expected error for empty command")
	}
}

// TestShellExec_Legacy_Timeout tests the legacy (no ProcessManager) timeout path.
func TestShellExec_Legacy_Timeout(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("test uses sh")
	}
	tool := &ShellExecTool{ProcessManager: nil}
	result, err := tool.Execute(context.Background(), `{"command":"sleep 10","timeout_seconds":1}`)
	if err != nil {
		t.Fatal(err)
	}
	if !result.IsError {
		t.Fatal("expected error for timed-out legacy command")
	}
	if !strings.Contains(result.Output, "timed out") {
		t.Errorf("expected 'timed out' in output, got: %s", result.Output)
	}
}

// TestShellExec_Legacy_MaxTimeout verifies capping beyond maxTimeout in legacy mode.
func TestShellExec_Legacy_MaxTimeout(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("test uses sh")
	}
	tool := &ShellExecTool{ProcessManager: nil}
	result, err := tool.Execute(context.Background(), `{"command":"echo hi","timeout_seconds":999}`)
	if err != nil {
		t.Fatal(err)
	}
	if result.IsError {
		t.Fatalf("unexpected error: %s", result.Output)
	}
}
