package server

import (
	"context"
	"encoding/json"
	"net"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/ThatCatDev/tanrenai/server/internal/config"
	"github.com/ThatCatDev/tanrenai/server/internal/gpuclient"
	"github.com/ThatCatDev/tanrenai/server/internal/gpuprovider"
	"github.com/ThatCatDev/tanrenai/server/internal/memory"
)

// testProvider is a minimal gpuprovider.Provider for server tests.
type testProvider struct{}

func (p *testProvider) Name() string                          { return "test" }
func (p *testProvider) EnsureRunning(_ context.Context) error { return nil }
func (p *testProvider) RecordActivity()                       {}
func (p *testProvider) Status(_ context.Context) (*gpuprovider.Status, error) {
	return &gpuprovider.Status{State: "running", Provider: "test"}, nil
}
func (p *testProvider) Stop(_ context.Context) error { return nil }
func (p *testProvider) StartIdleTimer()              {}
func (p *testProvider) Close()                       {}

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

func newTestServer(t *testing.T, withMemory bool) (*Server, *httptest.Server) {
	t.Helper()

	// Fake GPU server
	gpuSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/health" {
			w.WriteHeader(http.StatusOK)

			return
		}
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(`{}`))
	}))
	t.Cleanup(gpuSrv.Close)

	cfg := config.DefaultConfig()
	cfg.GPUURL = gpuSrv.URL
	cfg.Port = findFreePort(t)

	gpuClient := gpuclient.New(gpuSrv.URL)
	provider := &testProvider{}

	var memStore memory.Store
	if withMemory {
		embedFunc := func(_ context.Context, _ string) ([]float32, error) {
			vec := make([]float32, 384)
			for i := range vec {
				vec[i] = 0.1
			}

			return vec, nil
		}
		store, err := memory.NewChromemStoreInMemory(embedFunc)
		if err != nil {
			t.Fatalf("NewChromemStoreInMemory: %v", err)
		}
		memStore = store
	}

	s := New(cfg, gpuClient, memStore, provider)

	return s, gpuSrv
}

func TestServerNew(t *testing.T) {
	s, _ := newTestServer(t, false)
	if s == nil {
		t.Fatal("New() returned nil server")
	}
	if s.http == nil {
		t.Fatal("Server.http is nil")
	}
}

func TestServerNewWithMemory(t *testing.T) {
	s, _ := newTestServer(t, true)
	if s == nil {
		t.Fatal("New() returned nil server")
	}
	if s.memStore == nil {
		t.Fatal("Server.memStore is nil when memory enabled")
	}
}

func TestServerStartAndShutdown(t *testing.T) {
	s, _ := newTestServer(t, false)

	ctx, cancel := context.WithCancel(context.Background())

	errCh := make(chan error, 1)
	go func() {
		errCh <- s.Start(ctx)
	}()

	// Give the server a moment to start.
	time.Sleep(50 * time.Millisecond)

	// Cancel to trigger shutdown.
	cancel()

	select {
	case err := <-errCh:
		if err != nil {
			t.Errorf("Start() returned error: %v", err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("server did not shut down in time")
	}
}

func TestRegisterRoutesHealth(t *testing.T) {
	s, _ := newTestServer(t, false)

	// The server's mux is embedded in s.http.Handler via middleware.
	// Access the handler directly by creating a test request through the mux.
	mux := http.NewServeMux()
	s.registerRoutes(mux)

	req := httptest.NewRequest(http.MethodGet, "/health", nil)
	w := httptest.NewRecorder()
	mux.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("health returned status %d: %s", w.Code, w.Body.String())
	}
}

func TestRegisterRoutesMemoryEndpoints(t *testing.T) {
	s, _ := newTestServer(t, true)

	mux := http.NewServeMux()
	s.registerRoutes(mux)

	// Memory store is set, so /v1/memory/count should be registered.
	req := httptest.NewRequest(http.MethodGet, "/v1/memory/count", nil)
	w := httptest.NewRecorder()
	mux.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("memory count returned status %d: %s", w.Code, w.Body.String())
	}

	var resp struct {
		Count int `json:"count"`
	}
	if err := json.NewDecoder(w.Body).Decode(&resp); err != nil {
		t.Fatalf("decode response: %v", err)
	}
}

func TestWithLogging(t *testing.T) {
	inner := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	})
	handler := withLogging(inner)

	req := httptest.NewRequest(http.MethodGet, "/health", nil)
	w := httptest.NewRecorder()
	handler.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("withLogging: expected 200, got %d", w.Code)
	}
}

func TestWithCORS(t *testing.T) {
	inner := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	})
	handler := withCORS(inner)

	// Regular GET request.
	req := httptest.NewRequest(http.MethodGet, "/health", nil)
	w := httptest.NewRecorder()
	handler.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("withCORS GET: expected 200, got %d", w.Code)
	}
	if origin := w.Header().Get("Access-Control-Allow-Origin"); origin != "*" {
		t.Errorf("Access-Control-Allow-Origin = %q, want *", origin)
	}
}

func TestWithCORSOptions(t *testing.T) {
	inner := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		t.Fatal("inner handler should not be called for OPTIONS preflight")
	})
	handler := withCORS(inner)

	req := httptest.NewRequest(http.MethodOptions, "/v1/chat/completions", nil)
	w := httptest.NewRecorder()
	handler.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("withCORS OPTIONS: expected 200, got %d", w.Code)
	}
}

func TestResponseWriterFlush(t *testing.T) {
	rec := httptest.NewRecorder()
	rw := &responseWriter{ResponseWriter: rec, statusCode: http.StatusOK}

	// Flush should not panic.
	rw.Flush()
}

func TestResponseWriterWriteHeader(t *testing.T) {
	rec := httptest.NewRecorder()
	rw := &responseWriter{ResponseWriter: rec, statusCode: http.StatusOK}

	rw.WriteHeader(http.StatusCreated)
	if rw.statusCode != http.StatusCreated {
		t.Errorf("statusCode = %d, want 201", rw.statusCode)
	}
}
