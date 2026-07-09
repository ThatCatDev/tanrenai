//go:build !windows

package cmd

import (
	"os"
	"syscall"
)

// flockExclusive takes a blocking exclusive advisory lock on f.
func flockExclusive(f *os.File) error {
	return syscall.Flock(int(f.Fd()), syscall.LOCK_EX)
}

// flockUnlock releases the lock taken by flockExclusive.
func flockUnlock(f *os.File) error {
	return syscall.Flock(int(f.Fd()), syscall.LOCK_UN)
}
