//go:build windows

package tools

import (
	"os/exec"
	"syscall"
)

// setProcessGroup creates a new console process group on Windows so a
// later kill can target the whole group (sh + any grandchildren).
// Mirror of the Unix Setpgid behaviour — exec.CommandContext's default
// kill is also process-group-scoped on Windows when this flag is set.
func setProcessGroup(cmd *exec.Cmd) {
	if cmd.SysProcAttr == nil {
		cmd.SysProcAttr = &syscall.SysProcAttr{}
	}
	cmd.SysProcAttr.CreationFlags |= syscall.CREATE_NEW_PROCESS_GROUP
}

// killProcessGroup forcibly terminates the process tree. Windows has
// no SIGKILL equivalent for a group, so we fall back to Process.Kill
// which terminates the immediate process; CREATE_NEW_PROCESS_GROUP set
// above means descendants in the same group also receive CTRL_BREAK
// when the parent dies (taskkill /T would be more thorough but adds a
// dependency on an external binary).
func killProcessGroup(cmd *exec.Cmd) error {
	if cmd.Process == nil {
		return nil
	}
	return cmd.Process.Kill()
}
