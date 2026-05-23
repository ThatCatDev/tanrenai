//go:build !windows

package tools

import (
	"os/exec"
	"syscall"
)

// setProcessGroup makes the child the leader of a new process group so
// we can later SIGKILL the entire tree (sh + any grandchildren) with a
// single signal to -pgid. Without this, killing `sh -c "sleep 60"`
// only reaps `sh` and leaks `sleep` — which also keeps our captured
// stdout/stderr FDs open, hanging cmd.Wait indefinitely.
func setProcessGroup(cmd *exec.Cmd) {
	if cmd.SysProcAttr == nil {
		cmd.SysProcAttr = &syscall.SysProcAttr{}
	}
	cmd.SysProcAttr.Setpgid = true
}

// killProcessGroup sends SIGKILL to the entire process group rooted at
// cmd's child. Returns the error from the signal call so exec's cancel
// wiring sees a successful or failed signal as appropriate. If the
// process already exited the kill is a no-op (ESRCH); ignore it so
// cmd.Cancel doesn't report a spurious error.
func killProcessGroup(cmd *exec.Cmd) error {
	if cmd.Process == nil {
		return nil
	}
	// Negative PID == "signal the group with this PGID". The PGID
	// equals the leader's PID because we set Setpgid above.
	err := syscall.Kill(-cmd.Process.Pid, syscall.SIGKILL)
	if err == syscall.ESRCH {
		return nil
	}
	return err
}
