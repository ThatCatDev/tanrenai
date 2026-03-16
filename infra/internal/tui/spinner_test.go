package tui

import (
	"bytes"
	"errors"
	"strings"
	"testing"
)

func TestRunWithSpinnerSuccess(t *testing.T) {
	var buf bytes.Buffer
	called := false

	err := RunWithSpinner(&buf, "doing work", func() error {
		called = true

		return nil
	})

	if err != nil {
		t.Errorf("RunWithSpinner() unexpected error: %v", err)
	}
	if !called {
		t.Error("RunWithSpinner() did not call the function")
	}
	// Should have written a success marker to output
	out := buf.String()
	if !strings.Contains(out, "doing work") {
		t.Errorf("output should contain message, got: %q", out)
	}
}

func TestRunWithSpinnerError(t *testing.T) {
	var buf bytes.Buffer
	want := errors.New("operation failed")

	err := RunWithSpinner(&buf, "doing work", func() error {
		return want
	})

	if err != want {
		t.Errorf("RunWithSpinner() = %v, want %v", err, want)
	}
	out := buf.String()
	if !strings.Contains(out, "doing work") {
		t.Errorf("output should contain message on failure, got: %q", out)
	}
	if !strings.Contains(out, "operation failed") {
		t.Errorf("output should contain error message, got: %q", out)
	}
}

func TestSpinnerStopWritesMessage(t *testing.T) {
	var buf bytes.Buffer
	sp := NewSpinner(&buf, "my task")
	sp.Stop("my task")

	out := buf.String()
	if !strings.Contains(out, "my task") {
		t.Errorf("Stop() output should contain message, got: %q", out)
	}
}

func TestSpinnerStopFailWritesMessage(t *testing.T) {
	var buf bytes.Buffer
	sp := NewSpinner(&buf, "my task")
	sp.StopFail("my task failed")

	out := buf.String()
	if !strings.Contains(out, "my task failed") {
		t.Errorf("StopFail() output should contain message, got: %q", out)
	}
}

func TestRunWithSpinnerFunctionCalledOnce(t *testing.T) {
	var buf bytes.Buffer
	count := 0

	RunWithSpinner(&buf, "test", func() error {
		count++

		return nil
	})

	if count != 1 {
		t.Errorf("function called %d times, want 1", count)
	}
}

func TestRunWithSpinnerReturnsCorrectError(t *testing.T) {
	var buf bytes.Buffer
	sentinel := errors.New("sentinel error")

	err := RunWithSpinner(&buf, "task", func() error {
		return sentinel
	})
	if !errors.Is(err, sentinel) {
		t.Errorf("RunWithSpinner() returned %v, want sentinel error", err)
	}
}
