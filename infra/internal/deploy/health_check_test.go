package deploy

import (
	"bytes"
	"context"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/ThatCatDev/tanrenai/infra/internal/config"
	"github.com/ThatCatDev/tanrenai/infra/internal/network"
	"github.com/ThatCatDev/tanrenai/infra/internal/vastai"
)

// TestHealthCheckSuccess verifies healthCheck returns nil when /health returns 200.
func TestHealthCheckSuccess(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/health" {
			w.WriteHeader(http.StatusOK)
		} else {
			http.NotFound(w, r)
		}
	}))
	defer srv.Close()

	d := &Deployer{
		vastai:  vastai.NewClient("key"),
		network: network.NewNoneProvider(),
		cfg:     config.Defaults(),
		output:  &bytes.Buffer{},
	}

	ctx := context.Background()
	err := d.healthCheck(ctx, srv.URL)
	if err != nil {
		t.Errorf("healthCheck() error: %v", err)
	}
}

// TestHealthCheckContextCancelled verifies healthCheck returns an error when context is pre-cancelled.
func TestHealthCheckContextCancelled(t *testing.T) {
	// Server always returns non-200
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusServiceUnavailable)
	}))
	defer srv.Close()

	d := &Deployer{
		vastai:  vastai.NewClient("key"),
		network: network.NewNoneProvider(),
		cfg:     config.Defaults(),
		output:  &bytes.Buffer{},
	}

	ctx, cancel := context.WithCancel(context.Background())
	cancel() // pre-cancel

	err := d.healthCheck(ctx, srv.URL)
	if err == nil {
		t.Error("expected error when context is pre-cancelled")
	}
}

// TestHealthCheckTimeout verifies healthCheck times out when /health never returns 200.
func TestHealthCheckTimeout(t *testing.T) {
	// Server always returns 503
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusServiceUnavailable)
	}))
	defer srv.Close()

	d := &Deployer{
		vastai:  vastai.NewClient("key"),
		network: network.NewNoneProvider(),
		cfg:     config.Defaults(),
		output:  &bytes.Buffer{},
	}

	// Use a very short timeout so the test doesn't wait the full 30s
	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()

	err := d.healthCheck(ctx, srv.URL)
	if err == nil {
		t.Error("expected timeout error when health check never succeeds")
	}
}

// TestHealthCheckEventuallySucceeds verifies healthCheck succeeds after a few failures.
func TestHealthCheckEventuallySucceeds(t *testing.T) {
	callCount := 0

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		callCount++
		if callCount >= 2 {
			w.WriteHeader(http.StatusOK)
		} else {
			w.WriteHeader(http.StatusServiceUnavailable)
		}
	}))
	defer srv.Close()

	d := &Deployer{
		vastai:  vastai.NewClient("key"),
		network: network.NewNoneProvider(),
		cfg:     config.Defaults(),
		output:  &bytes.Buffer{},
	}

	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()

	err := d.healthCheck(ctx, srv.URL)
	if err != nil {
		t.Errorf("healthCheck() should eventually succeed, got error: %v", err)
	}
}
