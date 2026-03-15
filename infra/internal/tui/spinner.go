package tui

import (
	"context"
	"fmt"
	"io"
	"sync"
	"time"
)

var frames = [...]string{"⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"}

// Spinner displays an animated spinner with a message while a long-running operation runs.
type Spinner struct {
	w      io.Writer
	mu     sync.Mutex
	cancel context.CancelFunc
}

// NewSpinner creates and starts a spinner with the given message.
func NewSpinner(w io.Writer, msg string) *Spinner {
	ctx, cancel := context.WithCancel(context.Background())
	s := &Spinner{w: w, cancel: cancel}

	go func() {
		i := 0
		ticker := time.NewTicker(80 * time.Millisecond)
		defer ticker.Stop()

		for {
			select {
			case <-ctx.Done():
				return
			case <-ticker.C:
				s.mu.Lock()
				fmt.Fprintf(s.w, "\r%s %s", frames[i%len(frames)], msg)
				s.mu.Unlock()
				i++
			}
		}
	}()

	return s
}

// Stop stops the spinner and prints a final message.
func (s *Spinner) Stop(msg string) {
	s.cancel()
	s.mu.Lock()
	defer s.mu.Unlock()
	fmt.Fprintf(s.w, "\r✓ %s\n", msg)
}

// StopFail stops the spinner and prints a failure message.
func (s *Spinner) StopFail(msg string) {
	s.cancel()
	s.mu.Lock()
	defer s.mu.Unlock()
	fmt.Fprintf(s.w, "\r✗ %s\n", msg)
}

// RunWithSpinner runs a function while showing a spinner. Returns the function's error.
func RunWithSpinner(w io.Writer, msg string, fn func() error) error {
	sp := NewSpinner(w, msg)
	err := fn()
	if err != nil {
		sp.StopFail(fmt.Sprintf("%s — %v", msg, err))
	} else {
		sp.Stop(msg)
	}
	return err
}
