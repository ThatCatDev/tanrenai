package runner

import (
	"bufio"
	"context"
	"fmt"
	"io"
	"log"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"sync"
	"syscall"
	"time"
)

// Subprocess manages the lifecycle of a llama-server child process.
// It handles binary resolution, environment setup, process start/stop,
// structured logging, health polling, and graceful shutdown.
type Subprocess struct {
	cmd  *exec.Cmd
	mu   sync.Mutex
	port int

	binPath   string
	args      []string
	env       []string
	label     string // log prefix, e.g. "llama-server" or "embedding"
	quiet     bool
	baseURL   string
	healthy   bool
	stopped   bool          // true after explicit Close()
	doneCh    chan struct{} // closed when the process exits
	healthTimeout time.Duration
}

// SubprocessConfig holds everything needed to start a llama-server subprocess.
type SubprocessConfig struct {
	BinDir  string
	Args    []string // args to pass after the binary path
	Port    int      // 0 = auto-allocate
	Label   string   // log prefix (default "llama-server")
	Quiet   bool     // suppress subprocess stdout/stderr
	HealthTimeout time.Duration // how long to wait for /health (default 120s)
}

// allocatePort finds a free TCP port by binding to :0 and releasing it.
func allocatePort() (int, error) {
	l, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		return 0, fmt.Errorf("allocate port: %w", err)
	}
	port := l.Addr().(*net.TCPAddr).Port
	l.Close()
	return port, nil
}

// resolveBinary returns the full path to llama-server in binDir.
func resolveBinary(binDir string) (string, error) {
	binName := "llama-server"
	if runtime.GOOS == "windows" {
		binName = "llama-server.exe"
	}
	binPath := filepath.Join(binDir, binName)
	if _, err := os.Stat(binPath); os.IsNotExist(err) {
		return "", fmt.Errorf("llama-server not found at %s — download it with 'tanrenai setup'", binPath)
	}
	return binPath, nil
}

// NewSubprocess creates a Subprocess but does not start it. Call Start() next.
func NewSubprocess(cfg SubprocessConfig) (*Subprocess, error) {
	binPath, err := resolveBinary(cfg.BinDir)
	if err != nil {
		return nil, err
	}

	port := cfg.Port
	if port == 0 {
		port, err = allocatePort()
		if err != nil {
			return nil, err
		}
	}

	label := cfg.Label
	if label == "" {
		label = "llama-server"
	}

	healthTimeout := cfg.HealthTimeout
	if healthTimeout == 0 {
		healthTimeout = 120 * time.Second
	}

	env := append(os.Environ(), "LD_LIBRARY_PATH="+cfg.BinDir)

	return &Subprocess{
		binPath:       binPath,
		args:          cfg.Args,
		env:           env,
		port:          port,
		label:         label,
		quiet:         cfg.Quiet,
		baseURL:       fmt.Sprintf("http://127.0.0.1:%d", port),
		healthTimeout: healthTimeout,
		doneCh:        make(chan struct{}),
	}, nil
}

// Port returns the port the subprocess is listening on.
func (s *Subprocess) Port() int {
	return s.port
}

// BaseURL returns the HTTP base URL of the subprocess.
func (s *Subprocess) BaseURL() string {
	return s.baseURL
}

// Healthy returns whether the subprocess last passed a health check.
func (s *Subprocess) Healthy() bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.healthy
}

// Start launches the subprocess and waits for it to become healthy.
// The provided ctx controls only the health-check wait; the subprocess
// itself runs with a background lifetime.
func (s *Subprocess) Start(ctx context.Context) error {
	s.mu.Lock()
	s.stopped = false
	s.healthy = false
	s.doneCh = make(chan struct{})
	s.mu.Unlock()

	// Inject --port into args.
	args := make([]string, len(s.args))
	copy(args, s.args)
	portFound := false
	for i, a := range args {
		if a == "--port" && i+1 < len(args) {
			args[i+1] = strconv.Itoa(s.port)
			portFound = true
			break
		}
	}
	if !portFound {
		args = append(args, "--port", strconv.Itoa(s.port))
	}

	s.cmd = exec.Command(s.binPath, args...)
	s.cmd.Env = s.env

	if s.quiet {
		s.cmd.Stdout = io.Discard
		s.cmd.Stderr = io.Discard
	} else {
		s.pipeOutput()
	}

	log.Printf("[%s] starting: %s (port %d)", s.label, s.binPath, s.port)

	if err := s.cmd.Start(); err != nil {
		return fmt.Errorf("failed to start %s: %w", s.label, err)
	}

	// Background goroutine to detect process exit.
	go func() {
		s.cmd.Wait()
		close(s.doneCh)
	}()

	if err := s.waitForHealth(ctx); err != nil {
		s.GracefulStop()
		return fmt.Errorf("%s failed to become healthy: %w", s.label, err)
	}

	s.mu.Lock()
	s.healthy = true
	s.mu.Unlock()

	log.Printf("[%s] ready on port %d", s.label, s.port)
	return nil
}

// Done returns a channel that is closed when the subprocess exits.
func (s *Subprocess) Done() <-chan struct{} {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.doneCh
}

// ExitCode returns the process exit code, or -1 if not yet exited.
func (s *Subprocess) ExitCode() int {
	if s.cmd == nil || s.cmd.ProcessState == nil {
		return -1
	}
	return s.cmd.ProcessState.ExitCode()
}

// GracefulStop sends SIGTERM, waits up to 5 seconds, then SIGKILL.
func (s *Subprocess) GracefulStop() error {
	s.mu.Lock()
	s.stopped = true
	s.healthy = false
	s.mu.Unlock()

	if s.cmd == nil || s.cmd.Process == nil {
		return nil
	}

	pid := s.cmd.Process.Pid
	log.Printf("[%s] sending SIGTERM to PID %d", s.label, pid)

	// Send SIGTERM (SIGINT on Windows for graceful shutdown).
	var sigErr error
	if runtime.GOOS == "windows" {
		sigErr = s.cmd.Process.Signal(os.Interrupt)
	} else {
		sigErr = s.cmd.Process.Signal(syscall.SIGTERM)
	}

	if sigErr != nil {
		// Process may already be dead.
		log.Printf("[%s] signal failed (process may have exited): %v", s.label, sigErr)
		return nil
	}

	// Wait up to 5 seconds for clean exit.
	select {
	case <-s.doneCh:
		log.Printf("[%s] process exited cleanly", s.label)
		return nil
	case <-time.After(5 * time.Second):
		log.Printf("[%s] process did not exit after SIGTERM, sending SIGKILL to PID %d", s.label, pid)
		if err := s.cmd.Process.Kill(); err != nil {
			return fmt.Errorf("failed to kill %s: %w", s.label, err)
		}
		<-s.doneCh
		return nil
	}
}

// WasStopped returns true if GracefulStop was called (i.e., this was an intentional shutdown).
func (s *Subprocess) WasStopped() bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.stopped
}

// waitForHealth polls /health until it returns 200, with progress logging.
func (s *Subprocess) waitForHealth(ctx context.Context) error {
	deadline := time.Now().Add(s.healthTimeout)
	ticker := time.NewTicker(500 * time.Millisecond)
	defer ticker.Stop()

	progressTicker := time.NewTicker(5 * time.Second)
	defer progressTicker.Stop()

	start := time.Now()

	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-s.doneCh:
			return fmt.Errorf("%s process exited during startup (exit code %d)", s.label, s.ExitCode())
		case <-progressTicker.C:
			log.Printf("[%s] still loading model... (%.0fs elapsed)", s.label, time.Since(start).Seconds())
		case <-ticker.C:
			if time.Now().After(deadline) {
				return fmt.Errorf("timeout waiting for %s to become ready after %s", s.label, s.healthTimeout)
			}
			if s.healthCheck(ctx) == nil {
				return nil
			}
		}
	}
}

func (s *Subprocess) healthCheck(ctx context.Context) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, s.baseURL+"/health", nil)
	if err != nil {
		return err
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("health check returned %d", resp.StatusCode)
	}
	return nil
}

// pipeOutput connects subprocess stdout+stderr to Go's logger with a prefix.
func (s *Subprocess) pipeOutput() {
	prefix := fmt.Sprintf("[%s] ", s.label)

	stdoutPipe, err := s.cmd.StdoutPipe()
	if err == nil {
		go s.scanLines(stdoutPipe, prefix)
	}

	stderrPipe, err := s.cmd.StderrPipe()
	if err == nil {
		go s.scanLines(stderrPipe, prefix)
	}
}

func (s *Subprocess) scanLines(r io.Reader, prefix string) {
	scanner := bufio.NewScanner(r)
	scanner.Buffer(make([]byte, 0, 64*1024), 256*1024)
	for scanner.Scan() {
		log.Printf("%s%s", prefix, scanner.Text())
	}
}
