package cmd

import (
	"bytes"
	"io"
	"os"
	"strings"
	"testing"

	"github.com/ThatCatDev/tanrenai/infra/internal/vastai"
)

// captureStdout replaces os.Stdout with a pipe and captures what's written.
func captureStdout(fn func()) string {
	r, w, _ := os.Pipe()
	orig := os.Stdout
	os.Stdout = w
	fn()
	_ = w.Close()
	os.Stdout = orig
	var buf bytes.Buffer
	_, _ = io.Copy(&buf, r)

	return buf.String()
}

// TestPrintInstanceBasic verifies printInstance outputs the expected fields.
func TestPrintInstanceBasic(t *testing.T) {
	inst := &vastai.Instance{
		ID:        42,
		Status:    "running",
		GPUName:   "RTX 4090",
		NumGPUs:   2,
		CostPerHr: 0.456,
	}

	out := captureStdout(func() {
		printInstance(inst)
	})

	if !strings.Contains(out, "42") {
		t.Errorf("output should contain ID 42, got: %q", out)
	}
	if !strings.Contains(out, "running") {
		t.Errorf("output should contain status, got: %q", out)
	}
	if !strings.Contains(out, "RTX 4090") {
		t.Errorf("output should contain GPU name, got: %q", out)
	}
	if !strings.Contains(out, "0.456") {
		t.Errorf("output should contain cost, got: %q", out)
	}
}

// TestPrintInstanceWithSSH verifies printInstance includes SSH info when SSHHost is set.
func TestPrintInstanceWithSSH(t *testing.T) {
	inst := &vastai.Instance{
		ID:      99,
		Status:  "running",
		GPUName: "A100",
		SSHHost: "10.0.0.1",
		SSHPort: 2222,
	}

	out := captureStdout(func() {
		printInstance(inst)
	})

	if !strings.Contains(out, "10.0.0.1") {
		t.Errorf("output should contain SSH host, got: %q", out)
	}
	if !strings.Contains(out, "2222") {
		t.Errorf("output should contain SSH port, got: %q", out)
	}
}

// TestPrintInstanceNoSSH verifies printInstance omits SSH info when SSHHost is empty.
func TestPrintInstanceNoSSH(t *testing.T) {
	inst := &vastai.Instance{
		ID:      55,
		Status:  "loading",
		GPUName: "H100",
	}

	out := captureStdout(func() {
		printInstance(inst)
	})

	// Should not contain SSH: line
	if strings.Contains(out, "SSH:") {
		t.Errorf("output should not contain SSH info when SSHHost is empty, got: %q", out)
	}
}

// TestPrintInstanceMultiGPU verifies NumGPUs is shown correctly.
func TestPrintInstanceMultiGPU(t *testing.T) {
	inst := &vastai.Instance{
		ID:      1,
		Status:  "running",
		GPUName: "A100",
		NumGPUs: 8,
	}

	out := captureStdout(func() {
		printInstance(inst)
	})

	if !strings.Contains(out, "8") {
		t.Errorf("output should contain NumGPUs=8, got: %q", out)
	}
}

// TestPrintInstanceZeroCost verifies zero cost is printed.
func TestPrintInstanceZeroCost(t *testing.T) {
	inst := &vastai.Instance{
		ID:        7,
		Status:    "running",
		GPUName:   "RTX 3090",
		CostPerHr: 0.0,
	}

	out := captureStdout(func() {
		printInstance(inst)
	})

	if !strings.Contains(out, "0.000") {
		t.Errorf("output should contain 0.000 cost, got: %q", out)
	}
}

// TestVersionCommand verifies the version command runs without error.
func TestVersionCommand(t *testing.T) {
	// Run the root command with "version" arg
	// We use a subprocess-style test via cobra's Execute
	old := os.Stdout
	r, w, _ := os.Pipe()
	os.Stdout = w

	rootCmd.SetArgs([]string{"version"})
	err := rootCmd.Execute()

	_ = w.Close()
	os.Stdout = old
	var buf bytes.Buffer
	_, _ = io.Copy(&buf, r)

	if err != nil {
		t.Errorf("version command error: %v", err)
	}
	if !strings.Contains(buf.String(), "tanrenai-infra") {
		t.Errorf("version output should contain 'tanrenai-infra', got: %q", buf.String())
	}
}

// TestHelpCommand verifies the help command runs without error.
func TestHelpCommand(t *testing.T) {
	old := os.Stdout
	r, w, _ := os.Pipe()
	os.Stdout = w

	rootCmd.SetArgs([]string{"--help"})
	err := rootCmd.Execute()

	_ = w.Close()
	os.Stdout = old
	var buf bytes.Buffer
	_, _ = io.Copy(&buf, r)

	if err != nil {
		t.Errorf("help command error: %v", err)
	}
}

// TestExecuteRunsRootCmd verifies Execute() calls rootCmd.Execute() without panicking.
func TestExecuteRunsRootCmd(t *testing.T) {
	// Capture stderr (slog writes there) and stdout
	oldOut := os.Stdout
	oldErr := os.Stderr
	_, wo, _ := os.Pipe()
	_, we, _ := os.Pipe()
	os.Stdout = wo
	os.Stderr = we

	rootCmd.SetArgs([]string{"--help"})
	err := Execute()

	_ = wo.Close()
	_ = we.Close()
	os.Stdout = oldOut
	os.Stderr = oldErr

	// --help returns nil error
	if err != nil {
		t.Errorf("Execute() with --help error: %v", err)
	}
}
