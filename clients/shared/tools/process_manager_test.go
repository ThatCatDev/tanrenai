package tools

import (
	"context"
	"encoding/json"
	"runtime"
	"testing"
	"time"
)

func TestProcessManager_StartAndList(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("test uses sh")
	}
	pm := NewProcessManager()
	defer pm.KillAll()

	p, err := pm.Start("echo hello")
	if err != nil {
		t.Fatal(err)
	}
	if p.ID != 1 {
		t.Errorf("expected ID 1, got %d", p.ID)
	}
	if p.PID == 0 {
		t.Error("expected non-zero PID")
	}

	// Wait for process to finish
	<-pm.Done(p.ID)

	procs := pm.List()
	if len(procs) != 1 {
		t.Fatalf("expected 1 process, got %d", len(procs))
	}
	if procs[0].Status != ProcessExited {
		t.Errorf("expected exited, got %v", procs[0].Status)
	}
	if procs[0].ExitCode != 0 {
		t.Errorf("expected exit code 0, got %d", procs[0].ExitCode)
	}
}

func TestProcessManager_Output(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("test uses sh")
	}
	pm := NewProcessManager()
	defer pm.KillAll()

	p, err := pm.Start("echo hello world")
	if err != nil {
		t.Fatal(err)
	}
	<-pm.Done(p.ID)

	output := pm.Output(p.ID)
	if output != "hello world\n" {
		t.Errorf("got %q, want %q", output, "hello world\n")
	}
}

func TestProcessManager_Kill(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("test uses sh")
	}
	pm := NewProcessManager()
	defer pm.KillAll()

	p, err := pm.Start("sleep 60")
	if err != nil {
		t.Fatal(err)
	}

	if pm.RunningCount() != 1 {
		t.Fatalf("expected 1 running, got %d", pm.RunningCount())
	}

	if err := pm.Kill(p.ID); err != nil {
		t.Fatal(err)
	}

	select {
	case <-pm.Done(p.ID):
	// On a contended CI runner the kernel can take several seconds to
	// reap a killed process; 2s used to flake on shared/tools CI runs.
	// Generous timeout — local runs finish in <50ms anyway.
	case <-time.After(10 * time.Second):
		t.Fatal("process did not exit after kill")
	}

	procs := pm.List()
	if procs[0].Status != ProcessExited {
		t.Errorf("expected exited after kill, got %v", procs[0].Status)
	}
}

func TestProcessManager_KillNotFound(t *testing.T) {
	pm := NewProcessManager()
	err := pm.Kill(999)
	if err == nil {
		t.Error("expected error for non-existent process")
	}
}

func TestShellExec_FastCommandForeground(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("test uses sh")
	}
	pm := NewProcessManager()
	defer pm.KillAll()

	tool := &ShellExecTool{ProcessManager: pm}
	result, err := tool.Execute(context.Background(), `{"command":"echo fast"}`)
	if err != nil {
		t.Fatal(err)
	}
	if result.IsError {
		t.Fatalf("unexpected error: %s", result.Output)
	}
	if result.Output != "fast\n" {
		t.Errorf("got %q, want %q", result.Output, "fast\n")
	}
	// Fast commands should NOT be added to process manager
	if pm.Count() != 0 {
		t.Errorf("expected 0 processes, got %d", pm.Count())
	}
}

func TestShellExec_ExplicitBackground(t *testing.T) {
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
	if pm.RunningCount() != 1 {
		t.Errorf("expected 1 running process, got %d", pm.RunningCount())
	}
}

func TestShellExec_ExplicitForeground(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("test uses sh")
	}
	pm := NewProcessManager()
	defer pm.KillAll()

	tool := &ShellExecTool{ProcessManager: pm}
	fg := false
	args := shellExecArgs{Command: "echo hello", Background: &fg}
	argsJSON, _ := json.Marshal(args)
	result, err := tool.Execute(context.Background(), string(argsJSON))
	if err != nil {
		t.Fatal(err)
	}
	if result.IsError {
		t.Fatalf("unexpected error: %s", result.Output)
	}
	if pm.Count() != 0 {
		t.Errorf("expected 0 processes, got %d", pm.Count())
	}
}

func TestProcessManager_RemoveExited(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("test uses sh")
	}
	pm := NewProcessManager()
	defer pm.KillAll()

	// Start a fast command that exits quickly
	p1, err := pm.Start("echo one")
	if err != nil {
		t.Fatal(err)
	}
	// Start a long-running command
	p2, err := pm.Start("sleep 60")
	if err != nil {
		t.Fatal(err)
	}

	// Wait for p1 to exit
	<-pm.Done(p1.ID)

	if pm.Count() != 2 {
		t.Fatalf("expected 2 processes before RemoveExited, got %d", pm.Count())
	}

	pm.RemoveExited()

	if pm.Count() != 1 {
		t.Fatalf("expected 1 process after RemoveExited, got %d", pm.Count())
	}
	if pm.RunningCount() != 1 {
		t.Fatalf("expected 1 running process after RemoveExited, got %d", pm.RunningCount())
	}

	// The remaining process should be p2 (still running)
	procs := pm.List()
	if procs[0].ID != p2.ID {
		t.Errorf("expected remaining process ID %d, got %d", p2.ID, procs[0].ID)
	}

	// Cleanup
	_ = pm.Kill(p2.ID)
}

func TestProcessManager_RemoveExitedAllExited(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("test uses sh")
	}
	pm := NewProcessManager()

	p1, err := pm.Start("echo a")
	if err != nil {
		t.Fatal(err)
	}
	p2, err := pm.Start("echo b")
	if err != nil {
		t.Fatal(err)
	}

	<-pm.Done(p1.ID)
	<-pm.Done(p2.ID)

	pm.RemoveExited()

	if pm.Count() != 0 {
		t.Errorf("expected 0 processes after removing all exited, got %d", pm.Count())
	}
}

func TestShellExec_LegacyMode(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("test uses sh")
	}

	tool := &ShellExecTool{ProcessManager: nil}

	t.Run("success", func(t *testing.T) {
		result, err := tool.Execute(context.Background(), `{"command":"echo legacy"}`)
		if err != nil {
			t.Fatal(err)
		}
		if result.IsError {
			t.Fatalf("unexpected error: %s", result.Output)
		}
		if result.Output != "legacy\n" {
			t.Errorf("got %q, want %q", result.Output, "legacy\n")
		}
	})

	t.Run("failure", func(t *testing.T) {
		result, err := tool.Execute(context.Background(), `{"command":"exit 1"}`)
		if err != nil {
			t.Fatal(err)
		}
		if !result.IsError {
			t.Fatal("expected error result for failing command")
		}
	})

	t.Run("custom_timeout", func(t *testing.T) {
		result, err := tool.Execute(context.Background(), `{"command":"echo timed","timeout_seconds":10}`)
		if err != nil {
			t.Fatal(err)
		}
		if result.IsError {
			t.Fatalf("unexpected error: %s", result.Output)
		}
	})
}
