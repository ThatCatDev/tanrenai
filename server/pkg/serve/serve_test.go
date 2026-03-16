package serve

import (
	"context"
	"net"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
)

// findFreePort returns an available TCP port on loopback.
func findFreePort(t *testing.T) int {
	t.Helper()
	l, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("findFreePort: %v", err)
	}
	port := l.Addr().(*net.TCPAddr).Port
	l.Close()

	return port
}

func startAndShutdown(t *testing.T, cfg Config) {
	t.Helper()

	ctx, cancel := context.WithCancel(context.Background())

	errCh := make(chan error, 1)
	go func() {
		errCh <- Start(ctx, cfg)
	}()

	time.Sleep(80 * time.Millisecond)
	cancel()

	select {
	case err := <-errCh:
		if err != nil {
			t.Errorf("Start() returned error: %v", err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("Start() did not shut down in time")
	}
}

func TestStartLocalProviderNoMemory(t *testing.T) {
	gpuSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	}))
	defer gpuSrv.Close()

	startAndShutdown(t, Config{
		Host:   "127.0.0.1",
		Port:   findFreePort(t),
		GPUURL: gpuSrv.URL,
	})
}

func TestStartVastAIProvider(t *testing.T) {
	gpuSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	}))
	defer gpuSrv.Close()

	startAndShutdown(t, Config{
		Host:           "127.0.0.1",
		Port:           findFreePort(t),
		GPUURL:         gpuSrv.URL,
		VastaiAPIKey:   "test-key",
		VastaiInstance: "12345",
		IdleTimeout:    "30m",
	})
}

func TestStartInvalidIdleTimeout(t *testing.T) {
	// An unparseable idle timeout should fall back to the default (20m).
	gpuSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	}))
	defer gpuSrv.Close()

	startAndShutdown(t, Config{
		Host:           "127.0.0.1",
		Port:           findFreePort(t),
		GPUURL:         gpuSrv.URL,
		VastaiAPIKey:   "test-key",
		VastaiInstance: "99",
		IdleTimeout:    "not-a-duration",
	})
}

func TestStartWithMemory(t *testing.T) {
	gpuSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	}))
	defer gpuSrv.Close()

	startAndShutdown(t, Config{
		Host:          "127.0.0.1",
		Port:          findFreePort(t),
		GPUURL:        gpuSrv.URL,
		MemoryEnabled: true,
		MemoryDir:     t.TempDir(),
	})
}
