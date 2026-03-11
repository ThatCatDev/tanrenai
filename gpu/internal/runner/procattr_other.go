//go:build !linux

package runner

import "os/exec"

// setSysProcAttr is a no-op on non-Linux platforms.
// Pdeathsig is a Linux-specific feature.
func setSysProcAttr(cmd *exec.Cmd) {}
