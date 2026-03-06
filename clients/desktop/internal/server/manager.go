package server

import (
	"context"
	"fmt"
	"log"
	"net/http"
	"sync"
	"time"

	gpuserve "github.com/ThatCatDev/tanrenai/gpu/pkg/serve"
	backendserve "github.com/ThatCatDev/tanrenai/server/pkg/serve"
)

// Status represents the server lifecycle state.
type Status int

const (
	StatusStopped  Status = iota
	StatusStarting
	StatusRunning
	StatusError
)

func (s Status) String() string {
	switch s {
	case StatusStopped:
		return "Stopped"
	case StatusStarting:
		return "Starting..."
	case StatusRunning:
		return "Running"
	case StatusError:
		return "Error"
	default:
		return "Unknown"
	}
}

// Manager manages GPU server and backend as in-process goroutines.
type Manager struct {
	gpuPort    int
	serverPort int
	status     Status
	lastError  error
	cancel     context.CancelFunc // cancels both servers
	wg         sync.WaitGroup     // waits for both goroutines
	mu         sync.Mutex
	onStatus   func(Status, error)
}

// NewManager creates a new server manager.
func NewManager() *Manager {
	return &Manager{
		gpuPort:    11435,
		serverPort: 8080,
	}
}

// Status returns the current server status.
func (m *Manager) Status() Status {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.status
}

// LastError returns the last error if status is Error.
func (m *Manager) LastError() error {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.lastError
}

// SetOnStatus sets a callback for status changes.
func (m *Manager) SetOnStatus(fn func(Status, error)) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.onStatus = fn
}

// GPUPort returns the GPU server port.
func (m *Manager) GPUPort() int { return m.gpuPort }

// ServerPort returns the backend server port.
func (m *Manager) ServerPort() int { return m.serverPort }

// ServerURL returns the backend server URL.
func (m *Manager) ServerURL() string {
	return fmt.Sprintf("http://127.0.0.1:%d", m.serverPort)
}

func (m *Manager) setStatus(s Status, err error) {
	m.status = s
	m.lastError = err
	if m.onStatus != nil {
		m.onStatus(s, err)
	}
}

// Start launches the GPU server and backend as in-process goroutines.
func (m *Manager) Start() error {
	m.mu.Lock()
	defer m.mu.Unlock()

	if m.status == StatusRunning || m.status == StatusStarting {
		return nil
	}

	m.setStatus(StatusStarting, nil)

	ctx, cancel := context.WithCancel(context.Background())
	m.cancel = cancel

	gpuURL := fmt.Sprintf("http://127.0.0.1:%d", m.gpuPort)

	// Start GPU server in-process
	m.wg.Add(1)
	go func() {
		defer m.wg.Done()
		if err := gpuserve.Start(ctx, gpuserve.Config{
			Host:           "127.0.0.1",
			Port:           m.gpuPort,
			FlashAttention: true,
		}); err != nil && ctx.Err() == nil {
			log.Printf("GPU server error: %v", err)
		}
	}()

	// Wait for GPU server to be healthy
	if err := waitForHealth(gpuURL+"/v1/models", 30*time.Second); err != nil {
		cancel()
		m.wg.Wait()
		err = fmt.Errorf("GPU server failed to start: %w", err)
		m.setStatus(StatusError, err)
		return err
	}

	// Start backend server in-process
	m.wg.Add(1)
	go func() {
		defer m.wg.Done()
		if err := backendserve.Start(ctx, backendserve.Config{
			Host:   "127.0.0.1",
			Port:   m.serverPort,
			GPUURL: gpuURL,
		}); err != nil && ctx.Err() == nil {
			log.Printf("Backend server error: %v", err)
		}
	}()

	// Wait for backend to be healthy
	backendURL := fmt.Sprintf("http://127.0.0.1:%d", m.serverPort)
	if err := waitForHealth(backendURL+"/health", 15*time.Second); err != nil {
		cancel()
		m.wg.Wait()
		err = fmt.Errorf("backend server failed to start: %w", err)
		m.setStatus(StatusError, err)
		return err
	}

	m.setStatus(StatusRunning, nil)
	return nil
}

// Stop gracefully stops both servers.
func (m *Manager) Stop() error {
	m.mu.Lock()
	defer m.mu.Unlock()

	if m.cancel != nil {
		m.cancel()
		m.cancel = nil
	}
	m.wg.Wait()
	m.setStatus(StatusStopped, nil)
	return nil
}

func waitForHealth(url string, timeout time.Duration) error {
	deadline := time.Now().Add(timeout)
	client := &http.Client{Timeout: 2 * time.Second}

	for time.Now().Before(deadline) {
		resp, err := client.Get(url)
		if err == nil {
			resp.Body.Close()
			if resp.StatusCode == http.StatusOK {
				return nil
			}
		}
		time.Sleep(500 * time.Millisecond)
	}
	return fmt.Errorf("timed out waiting for %s", url)
}
