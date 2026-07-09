//go:build windows

package cmd

import (
	"os"

	"golang.org/x/sys/windows"
)

// flockExclusive takes a blocking exclusive lock on f via LockFileEx.
func flockExclusive(f *os.File) error {
	ol := new(windows.Overlapped)
	return windows.LockFileEx(windows.Handle(f.Fd()), windows.LOCKFILE_EXCLUSIVE_LOCK, 0, 1, 0, ol)
}

// flockUnlock releases the lock taken by flockExclusive.
func flockUnlock(f *os.File) error {
	ol := new(windows.Overlapped)
	return windows.UnlockFileEx(windows.Handle(f.Fd()), 0, 1, 0, ol)
}
