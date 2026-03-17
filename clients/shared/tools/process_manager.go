package tools

import (
	"context"
	"fmt"
	"os/exec"
	"runtime"
	"sync"
	"time"
)

// ProcessStatus represents the state of a managed background process.
type ProcessStatus int

const (
	ProcessRunning ProcessStatus = iota
	ProcessExited
)

// ManagedProcess tracks a background process spawned by shell_exec.
type ManagedProcess struct {
	ID        int
	PID       int
	ExitCode  int
	Command   string
	Status    ProcessStatus
	StartedAt time.Time
	buf       *RingBuffer
	cmd       *exec.Cmd
	cancel    context.CancelFunc
	done      chan struct{}
}

// ProcessSnapshot is a safe copy of process state for display.
type ProcessSnapshot struct {
	ID        int
	PID       int
	ExitCode  int
	Command   string
	Status    ProcessStatus
	StartedAt time.Time
}

// ProcessManager manages background processes spawned by the agent.
type ProcessManager struct {
	mu        sync.Mutex
	processes []*ManagedProcess
	nextID    int
	OnChange  func() // TUI sets this for redraws
}

// NewProcessManager creates a new process manager.
func NewProcessManager() *ProcessManager {
	return &ProcessManager{}
}

// Start spawns a command in the background, capturing output into a ring buffer.
func (pm *ProcessManager) Start(command string) (*ManagedProcess, error) {
	ctx, cancel := context.WithCancel(context.Background())

	var cmd *exec.Cmd
	if runtime.GOOS == "windows" {
		cmd = exec.CommandContext(ctx, "cmd", "/C", command)
	} else {
		cmd = exec.CommandContext(ctx, "sh", "-c", command)
	}

	buf := NewRingBuffer(maxShellOutput) // 64KB
	cmd.Stdout = buf
	cmd.Stderr = buf

	if err := cmd.Start(); err != nil {
		cancel()

		return nil, fmt.Errorf("failed to start: %w", err)
	}

	return pm.adopt(cmd, buf, cancel, command)
}

func (pm *ProcessManager) adopt(cmd *exec.Cmd, buf *RingBuffer, cancel context.CancelFunc, command string) (*ManagedProcess, error) {
	pm.mu.Lock()
	pm.nextID++
	p := &ManagedProcess{
		ID:        pm.nextID,
		PID:       cmd.Process.Pid,
		Command:   command,
		Status:    ProcessRunning,
		StartedAt: time.Now(),
		buf:       buf,
		cmd:       cmd,
		cancel:    cancel,
		done:      make(chan struct{}),
	}
	pm.processes = append(pm.processes, p)
	pm.mu.Unlock()

	go func() {
		err := cmd.Wait()
		pm.mu.Lock()
		p.Status = ProcessExited
		if err != nil {
			if exitErr, ok := err.(*exec.ExitError); ok {
				p.ExitCode = exitErr.ExitCode()
			} else {
				p.ExitCode = -1
			}
		}
		pm.mu.Unlock()
		close(p.done)
		if pm.OnChange != nil {
			pm.OnChange()
		}
	}()

	if pm.OnChange != nil {
		pm.OnChange()
	}

	return p, nil
}

// List returns snapshots of all managed processes.
func (pm *ProcessManager) List() []ProcessSnapshot {
	pm.mu.Lock()
	defer pm.mu.Unlock()
	out := make([]ProcessSnapshot, len(pm.processes))
	for i, p := range pm.processes {
		out[i] = ProcessSnapshot{
			ID:        p.ID,
			PID:       p.PID,
			ExitCode:  p.ExitCode,
			Command:   p.Command,
			Status:    p.Status,
			StartedAt: p.StartedAt,
		}
	}

	return out
}

// Output returns the ring buffer contents for a process.
func (pm *ProcessManager) Output(id int) string {
	pm.mu.Lock()
	defer pm.mu.Unlock()
	for _, p := range pm.processes {
		if p.ID == id {
			return p.buf.String()
		}
	}

	return ""
}

// Kill stops a background process by ID.
func (pm *ProcessManager) Kill(id int) error {
	pm.mu.Lock()
	defer pm.mu.Unlock()
	for _, p := range pm.processes {
		if p.ID == id {
			if p.Status == ProcessRunning {
				p.cancel()
			}

			return nil
		}
	}

	return fmt.Errorf("process %d not found", id)
}

// Done returns the done channel for a process, or nil if not found.
func (pm *ProcessManager) Done(id int) <-chan struct{} {
	pm.mu.Lock()
	defer pm.mu.Unlock()
	for _, p := range pm.processes {
		if p.ID == id {
			return p.done
		}
	}

	return nil
}

// Count returns the number of managed processes.
func (pm *ProcessManager) Count() int {
	pm.mu.Lock()
	defer pm.mu.Unlock()

	return len(pm.processes)
}

// RunningCount returns the number of currently running processes.
func (pm *ProcessManager) RunningCount() int {
	pm.mu.Lock()
	defer pm.mu.Unlock()
	n := 0
	for _, p := range pm.processes {
		if p.Status == ProcessRunning {
			n++
		}
	}

	return n
}

// Remove removes a specific process by ID.
func (pm *ProcessManager) Remove(id int) {
	pm.mu.Lock()
	defer pm.mu.Unlock()
	for i, p := range pm.processes {
		if p.ID == id {
			pm.processes = append(pm.processes[:i], pm.processes[i+1:]...)

			return
		}
	}
}

// RemoveExited removes all exited processes from the list.
func (pm *ProcessManager) RemoveExited() {
	pm.mu.Lock()
	defer pm.mu.Unlock()
	var kept []*ManagedProcess
	for _, p := range pm.processes {
		if p.Status == ProcessRunning {
			kept = append(kept, p)
		}
	}
	pm.processes = kept
}

// KillAll stops all running background processes.
func (pm *ProcessManager) KillAll() {
	pm.mu.Lock()
	defer pm.mu.Unlock()
	for _, p := range pm.processes {
		if p.Status == ProcessRunning {
			p.cancel()
		}
	}
}
