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
	case <-time.After(2 * time.Second):
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
