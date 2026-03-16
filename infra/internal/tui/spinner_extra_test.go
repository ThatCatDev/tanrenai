package tui

import (
	"bytes"
	"strings"
	"testing"
	"time"
)

// TestSpinnerTickerFires verifies the spinner goroutine actually writes spinner frames.
// We wait > 80ms to let at least one ticker tick before stopping.
func TestSpinnerTickerFires(t *testing.T) {
	var buf bytes.Buffer
	sp := NewSpinner(&buf, "spinning")
	// Wait for at least 2 ticker ticks (80ms each)
	time.Sleep(200 * time.Millisecond)
	sp.Stop("done")

	out := buf.String()
	// Should contain a frame character (the spinner writes \r<frame> <msg>)
	// The frames are: ⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏
	hasFrame := strings.Contains(out, "⠋") || strings.Contains(out, "⠙") ||
		strings.Contains(out, "⠹") || strings.Contains(out, "⠸") ||
		strings.Contains(out, "⠼") || strings.Contains(out, "⠴") ||
		strings.Contains(out, "⠦") || strings.Contains(out, "⠧") ||
		strings.Contains(out, "⠇") || strings.Contains(out, "⠏")
	if !hasFrame {
		t.Errorf("spinner should have written a frame character, got: %q", out)
	}
}

// TestSpinnerWritesCorrectMessage verifies the message is included in spinner output.
func TestSpinnerWritesCorrectMessage(t *testing.T) {
	var buf bytes.Buffer
	sp := NewSpinner(&buf, "test-message-xyz")
	time.Sleep(150 * time.Millisecond)
	sp.Stop("finished")

	out := buf.String()
	if !strings.Contains(out, "test-message-xyz") {
		t.Errorf("spinner output should contain message, got: %q", out)
	}
}

// TestSpinnerStopSuccessMarker verifies Stop writes the success marker.
func TestSpinnerStopSuccessMarker(t *testing.T) {
	var buf bytes.Buffer
	sp := NewSpinner(&buf, "task")
	sp.Stop("task complete")

	out := buf.String()
	if !strings.Contains(out, "✓") {
		t.Errorf("Stop() should write ✓ marker, got: %q", out)
	}
}

// TestSpinnerStopFailMarker verifies StopFail writes the failure marker.
func TestSpinnerStopFailMarker(t *testing.T) {
	var buf bytes.Buffer
	sp := NewSpinner(&buf, "task")
	sp.StopFail("task failed")

	out := buf.String()
	if !strings.Contains(out, "✗") {
		t.Errorf("StopFail() should write ✗ marker, got: %q", out)
	}
}
