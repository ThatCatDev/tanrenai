package gpuprovider

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/ThatCatDev/tanrenai/server/internal/gpuclient"
)

func newLocalTestProvider(t *testing.T, healthy bool) (*LocalProvider, *httptest.Server) {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/health" {
			if healthy {
				w.WriteHeader(http.StatusOK)
			} else {
				w.WriteHeader(http.StatusInternalServerError)
			}

			return
		}
		http.NotFound(w, r)
	}))
	t.Cleanup(srv.Close)
	client := gpuclient.New(srv.URL)

	return NewLocalProvider(client), srv
}

func TestLocalProviderName(t *testing.T) {
	p, _ := newLocalTestProvider(t, true)
	if got := p.Name(); got != "local" {
		t.Errorf("Name() = %q, want %q", got, "local")
	}
}

func TestLocalProviderRecordActivity(t *testing.T) {
	p, _ := newLocalTestProvider(t, true)
	// RecordActivity is a no-op; just ensure it doesn't panic.
	p.RecordActivity()
}

func TestLocalProviderClose(t *testing.T) {
	p, _ := newLocalTestProvider(t, true)
	// Close is a no-op; just ensure it doesn't panic.
	p.Close()
}

func TestLocalProviderStartIdleTimer(t *testing.T) {
	p, _ := newLocalTestProvider(t, true)
	// StartIdleTimer is a no-op; just ensure it doesn't panic.
	p.StartIdleTimer()
}

func TestLocalProviderStop(t *testing.T) {
	p, _ := newLocalTestProvider(t, true)
	if err := p.Stop(context.Background()); err != nil {
		t.Errorf("Stop() returned error: %v", err)
	}
}

func TestLocalProviderEnsureRunningHealthy(t *testing.T) {
	p, _ := newLocalTestProvider(t, true)
	if err := p.EnsureRunning(context.Background()); err != nil {
		t.Errorf("EnsureRunning() returned error: %v", err)
	}
}

func TestLocalProviderEnsureRunningUnhealthy(t *testing.T) {
	p, _ := newLocalTestProvider(t, false)
	if err := p.EnsureRunning(context.Background()); err == nil {
		t.Error("EnsureRunning() expected error for unhealthy server, got nil")
	}
}

func TestLocalProviderStatusRunning(t *testing.T) {
	p, srv := newLocalTestProvider(t, true)
	status, err := p.Status(context.Background())
	if err != nil {
		t.Fatalf("Status() returned error: %v", err)
	}
	if status.State != "running" {
		t.Errorf("State = %q, want %q", status.State, "running")
	}
	if status.Provider != "local" {
		t.Errorf("Provider = %q, want %q", status.Provider, "local")
	}
	if status.GPUURL != srv.URL {
		t.Errorf("GPUURL = %q, want %q", status.GPUURL, srv.URL)
	}
}

func TestLocalProviderStatusStopped(t *testing.T) {
	p, _ := newLocalTestProvider(t, false)
	status, err := p.Status(context.Background())
	if err != nil {
		t.Fatalf("Status() returned error: %v", err)
	}
	if status.State != "stopped" {
		t.Errorf("State = %q, want %q", status.State, "stopped")
	}
	if status.Provider != "local" {
		t.Errorf("Provider = %q, want %q", status.Provider, "local")
	}
}
