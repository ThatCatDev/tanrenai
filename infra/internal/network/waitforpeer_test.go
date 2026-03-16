package network

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

// TestHeadscaleWaitForPeerFindsNode verifies WaitForPeer returns the IP when the node is found on tick.
// This test waits for the 5-second ticker to fire, then verifies the IP is returned.
func TestHeadscaleWaitForPeerFindsNode(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping 5s ticker test in short mode")
	}

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"nodes": []map[string]any{
				{
					"givenName":   "gpu-node-1",
					"ipAddresses": []string{"100.64.0.5"},
					"online":      true,
				},
			},
		})
	}))
	defer srv.Close()

	p := &HeadscaleProvider{
		baseURL:    srv.URL,
		apiKey:     "key",
		httpClient: srv.Client(),
	}

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	ip, err := p.WaitForPeer(ctx, "gpu-node-1")
	if err != nil {
		t.Fatalf("WaitForPeer() error: %v", err)
	}
	if ip != "100.64.0.5" {
		t.Errorf("WaitForPeer() = %q, want \"100.64.0.5\"", ip)
	}
}

// TestTailscaleWaitForPeerFindsDevice verifies Tailscale WaitForPeer returns the IP on tick.
func TestTailscaleWaitForPeerFindsDevice(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping 5s ticker test in short mode")
	}

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"devices": []map[string]any{
				{
					"hostname":  "gpu-node-1",
					"addresses": []string{"100.64.0.9"},
				},
			},
		})
	}))
	defer srv.Close()

	p := &TailscaleProvider{
		authKey:    "tskey-123",
		httpClient: newTailscaleTestClient(srv),
	}

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	ip, err := p.WaitForPeer(ctx, "gpu-node-1")
	if err != nil {
		t.Fatalf("WaitForPeer() error: %v", err)
	}
	if ip != "100.64.0.9" {
		t.Errorf("WaitForPeer() = %q, want \"100.64.0.9\"", ip)
	}
}

// TestHeadscaleWaitForPeerErrorThenFound verifies WaitForPeer retries after errors.
func TestHeadscaleWaitForPeerErrorThenFound(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping 5s ticker test in short mode")
	}

	callCount := 0

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		callCount++
		if callCount == 1 {
			// First call returns error
			http.Error(w, "server error", http.StatusInternalServerError)

			return
		}
		// Second call returns the node
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"nodes": []map[string]any{
				{
					"givenName":   "my-node",
					"ipAddresses": []string{"10.0.0.1"},
					"online":      true,
				},
			},
		})
	}))
	defer srv.Close()

	p := &HeadscaleProvider{
		baseURL:    srv.URL,
		apiKey:     "key",
		httpClient: srv.Client(),
	}

	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()

	ip, err := p.WaitForPeer(ctx, "my-node")
	if err != nil {
		t.Fatalf("WaitForPeer() error: %v", err)
	}
	if ip != "10.0.0.1" {
		t.Errorf("WaitForPeer() = %q, want \"10.0.0.1\"", ip)
	}
}

// TestHeadscaleWaitForPeerEmptyThenFound verifies WaitForPeer continues when node not yet visible.
func TestHeadscaleWaitForPeerEmptyThenFound(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping 5s ticker test in short mode")
	}

	callCount := 0

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		callCount++
		w.Header().Set("Content-Type", "application/json")
		if callCount == 1 {
			// First call: empty nodes list
			json.NewEncoder(w).Encode(map[string]any{"nodes": []any{}})

			return
		}
		// Second call: node appears
		json.NewEncoder(w).Encode(map[string]any{
			"nodes": []map[string]any{
				{
					"givenName":   "new-node",
					"ipAddresses": []string{"192.168.1.1"},
					"online":      true,
				},
			},
		})
	}))
	defer srv.Close()

	p := &HeadscaleProvider{
		baseURL:    srv.URL,
		apiKey:     "key",
		httpClient: srv.Client(),
	}

	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()

	ip, err := p.WaitForPeer(ctx, "new-node")
	if err != nil {
		t.Fatalf("WaitForPeer() error: %v", err)
	}
	if !strings.Contains(ip, "192.168.1.1") {
		t.Errorf("WaitForPeer() = %q, want \"192.168.1.1\"", ip)
	}
}
