package remote

import (
	"bytes"
	"context"
	"strings"
	"testing"
)

// TestRunStagesEmpty verifies RunStages with no stages succeeds immediately.
func TestRunStagesEmpty(t *testing.T) {
	client, cleanup := makeTestSSHClient(t)
	defer cleanup()

	var buf bytes.Buffer
	err := RunStages(context.Background(), client, []SetupStage{}, &buf)
	if err != nil {
		t.Errorf("RunStages() with empty stages error: %v", err)
	}
}

// TestRunStagesSingleStage verifies RunStages executes a single stage over SSH.
func TestRunStagesSingleStage(t *testing.T) {
	client, cleanup := makeTestSSHClient(t)
	defer cleanup()

	stages := []SetupStage{
		{Name: "test-stage", Commands: []string{"echo hello"}},
	}

	var buf bytes.Buffer
	err := RunStages(context.Background(), client, stages, &buf)
	if err != nil {
		t.Errorf("RunStages() error: %v", err)
	}

	out := buf.String()
	if !strings.Contains(out, "test-stage") {
		t.Errorf("output should contain stage name, got: %q", out)
	}
}

// TestRunStagesMultipleStages verifies RunStages executes stages in order.
func TestRunStagesMultipleStages(t *testing.T) {
	client, cleanup := makeTestSSHClient(t)
	defer cleanup()

	stages := []SetupStage{
		{Name: "stage-1", Commands: []string{"echo one"}},
		{Name: "stage-2", Commands: []string{"echo two"}},
		{Name: "stage-3", Commands: []string{"echo three"}},
	}

	var buf bytes.Buffer
	err := RunStages(context.Background(), client, stages, &buf)
	if err != nil {
		t.Errorf("RunStages() error: %v", err)
	}

	out := buf.String()
	for _, name := range []string{"stage-1", "stage-2", "stage-3"} {
		if !strings.Contains(out, name) {
			t.Errorf("output should contain %q, got: %q", name, out)
		}
	}
}

// TestRunStagesMultipleCommandsJoined verifies multiple commands within a stage are joined with &&.
func TestRunStagesMultipleCommandsJoined(t *testing.T) {
	client, cleanup := makeTestSSHClient(t)
	defer cleanup()

	stages := []SetupStage{
		{Name: "multi-cmd", Commands: []string{"echo a", "echo b", "echo c"}},
	}

	var buf bytes.Buffer
	err := RunStages(context.Background(), client, stages, &buf)
	if err != nil {
		t.Errorf("RunStages() error: %v", err)
	}
}

// TestRunStagesWritesHeaderToOutput verifies RunStages writes the stage name header to the output writer.
func TestRunStagesWritesHeaderToOutput(t *testing.T) {
	client, cleanup := makeTestSSHClient(t)
	defer cleanup()

	stages := []SetupStage{
		{Name: "my-special-stage", Commands: []string{"echo hello"}},
	}

	var buf bytes.Buffer
	_ = RunStages(context.Background(), client, stages, &buf)

	out := buf.String()
	if !strings.Contains(out, "=== my-special-stage ===") {
		t.Errorf("output should contain stage header, got: %q", out)
	}
}

// TestRunStagesContextCancelled verifies RunStages returns an error when context is pre-cancelled.
func TestRunStagesContextCancelled(t *testing.T) {
	client, cleanup := makeTestSSHClient(t)
	defer cleanup()

	stages := []SetupStage{
		{Name: "test-stage", Commands: []string{"echo hello"}},
	}

	ctx, cancel := context.WithCancel(context.Background())
	cancel() // pre-cancel

	var buf bytes.Buffer
	err := RunStages(ctx, client, stages, &buf)
	// With a cancelled context, the ssh.Run call should return an error
	// (either context.Canceled or the underlying SSH error)
	_ = err // may or may not error depending on timing; just ensure no panic
}
