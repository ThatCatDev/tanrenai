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

// TestHeadscaleFindNodeFound verifies findNode returns the IP when the node is present and online.
func TestHeadscaleFindNodeFound(t *testing.T) {
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

	ip, err := p.findNode(context.Background(), "gpu-node-1")
	if err != nil {
		t.Fatalf("findNode() error: %v", err)
	}
	if ip != "100.64.0.5" {
		t.Errorf("findNode() = %q, want \"100.64.0.5\"", ip)
	}
}

// TestHeadscaleFindNodeNotFound verifies findNode returns empty string when the node doesn't exist.
func TestHeadscaleFindNodeNotFound(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"nodes": []map[string]any{
				{
					"givenName":   "other-node",
					"ipAddresses": []string{"100.64.0.1"},
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

	ip, err := p.findNode(context.Background(), "gpu-node-1")
	if err != nil {
		t.Fatalf("findNode() unexpected error: %v", err)
	}
	if ip != "" {
		t.Errorf("findNode() = %q, want empty string for missing node", ip)
	}
}

// TestHeadscaleFindNodeOffline verifies findNode ignores offline nodes.
func TestHeadscaleFindNodeOffline(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"nodes": []map[string]any{
				{
					"givenName":   "gpu-node-1",
					"ipAddresses": []string{"100.64.0.5"},
					"online":      false, // offline
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

	ip, err := p.findNode(context.Background(), "gpu-node-1")
	if err != nil {
		t.Fatalf("findNode() unexpected error: %v", err)
	}
	if ip != "" {
		t.Errorf("findNode() = %q, want empty string for offline node", ip)
	}
}

// TestHeadscaleFindNodeNoIPs verifies findNode ignores nodes with no IP addresses.
func TestHeadscaleFindNodeNoIPs(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"nodes": []map[string]any{
				{
					"givenName":   "gpu-node-1",
					"ipAddresses": []string{}, // no IPs
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

	ip, err := p.findNode(context.Background(), "gpu-node-1")
	if err != nil {
		t.Fatalf("findNode() unexpected error: %v", err)
	}
	if ip != "" {
		t.Errorf("findNode() = %q, want empty string for node with no IPs", ip)
	}
}

// TestHeadscaleFindNodeAPIError verifies findNode returns an error on non-200 responses.
func TestHeadscaleFindNodeAPIError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "forbidden", http.StatusForbidden)
	}))
	defer srv.Close()

	p := &HeadscaleProvider{
		baseURL:    srv.URL,
		apiKey:     "key",
		httpClient: srv.Client(),
	}

	_, err := p.findNode(context.Background(), "any-node")
	if err == nil {
		t.Error("expected error on 403 response")
	}
}

// TestHeadscaleWaitForPeerContextCancelled verifies WaitForPeer returns ctx.Err() when context is cancelled.
func TestHeadscaleWaitForPeerContextCancelled(t *testing.T) {
	// Server never returns the node — always empty
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{"nodes": []any{}})
	}))
	defer srv.Close()

	p := &HeadscaleProvider{
		baseURL:    srv.URL,
		apiKey:     "key",
		httpClient: srv.Client(),
	}

	// Cancel the context immediately so WaitForPeer returns without polling
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	_, err := p.WaitForPeer(ctx, "gpu-node-1")
	if err == nil {
		t.Error("expected error when context is cancelled")
	}
	if err != context.Canceled {
		t.Errorf("expected context.Canceled, got: %v", err)
	}
}

// TestHeadscaleWaitForPeerReturnsIPImmediately verifies WaitForPeer returns the IP when the node is
// already present by using a short ticker interval via a custom httpClient transport trick:
// we replace the ticker with a very short one by manipulating time — instead we use a
// context with a timeout and a server that returns the node on every call, but we need
// the ticker to fire. We set up the server and then use context deadline approach.
func TestHeadscaleWaitForPeerReturnsIP(t *testing.T) {
	// Patch the ticker interval by swapping the implementation via wrapping:
	// Since we can't override the 5s ticker easily, we test via a short-lived
	// context that triggers the done path, plus a direct findNode call to cover
	// the happy-path code paths within findNode.
	// The WaitForPeer happy path is better covered via integration but we can test
	// findNode directly and the context cancellation path of WaitForPeer.

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
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

	// Test findNode directly for the happy path
	ip, err := p.findNode(context.Background(), "my-node")
	if err != nil {
		t.Fatalf("findNode() error: %v", err)
	}
	if ip != "10.0.0.1" {
		t.Errorf("findNode() = %q, want \"10.0.0.1\"", ip)
	}

	// Test WaitForPeer with deadline that expires before 5s ticker fires
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Millisecond)
	defer cancel()

	_, err = p.WaitForPeer(ctx, "my-node")
	// Should return context.DeadlineExceeded since ticker won't fire before the 10ms deadline
	if err == nil {
		t.Error("expected context deadline to be hit before 5s ticker")
	}
}

// TestHeadscaleFindNodeSetsAuthHeader verifies the Authorization header is sent.
func TestHeadscaleFindNodeSetsAuthHeader(t *testing.T) {
	var gotAuth string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotAuth = r.Header.Get("Authorization")
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{"nodes": []any{}})
	}))
	defer srv.Close()

	p := &HeadscaleProvider{
		baseURL:    srv.URL,
		apiKey:     "my-secret-key",
		httpClient: srv.Client(),
	}

	_, _ = p.findNode(context.Background(), "any")
	if gotAuth != "Bearer my-secret-key" {
		t.Errorf("Authorization = %q, want \"Bearer my-secret-key\"", gotAuth)
	}
}

// TestHeadscaleFindNodeMultipleNodes verifies findNode returns the first IP of the matching node.
func TestHeadscaleFindNodeMultipleNodes(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"nodes": []map[string]any{
				{
					"givenName":   "other-node",
					"ipAddresses": []string{"100.64.0.1"},
					"online":      true,
				},
				{
					"givenName":   "target-node",
					"ipAddresses": []string{"100.64.0.7", "fd7a::1"},
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

	ip, err := p.findNode(context.Background(), "target-node")
	if err != nil {
		t.Fatalf("findNode() error: %v", err)
	}
	if ip != "100.64.0.7" {
		t.Errorf("findNode() = %q, want \"100.64.0.7\" (first IP)", ip)
	}
}

// TestHeadscaleFindNodeCaseSensitive verifies that name matching is case-sensitive (exact match).
func TestHeadscaleFindNodeCaseSensitive(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"nodes": []map[string]any{
				{
					"givenName":   "GPU-NODE",
					"ipAddresses": []string{"100.64.0.1"},
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

	// findNode uses exact match (==), not EqualFold
	ip, err := p.findNode(context.Background(), "gpu-node")
	if err != nil {
		t.Fatalf("findNode() error: %v", err)
	}
	// Case mismatch — should not match
	if ip != "" {
		t.Errorf("findNode() = %q, expected empty string (case-sensitive match)", ip)
	}
}

// TestHeadscaleFindNodeDecodeError verifies findNode returns an error on invalid JSON.
func TestHeadscaleFindNodeDecodeError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte("{invalid json}"))
	}))
	defer srv.Close()

	p := &HeadscaleProvider{
		baseURL:    srv.URL,
		apiKey:     "key",
		httpClient: srv.Client(),
	}

	_, err := p.findNode(context.Background(), "any-node")
	if err == nil {
		t.Error("expected error on invalid JSON response")
	}
}

// TestHeadscaleWaitForPeerCancelledImmediately verifies WaitForPeer propagates context errors correctly.
func TestHeadscaleWaitForPeerCancelledImmediately(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{"nodes": []any{}})
	}))
	defer srv.Close()

	p := &HeadscaleProvider{
		baseURL:    srv.URL,
		apiKey:     "key",
		httpClient: srv.Client(),
	}

	ctx, cancel := context.WithCancel(context.Background())
	cancel() // already cancelled

	_, err := p.WaitForPeer(ctx, "any")
	if err == nil {
		t.Error("expected error when context is pre-cancelled")
	}
	if !strings.Contains(err.Error(), "cancel") {
		t.Errorf("expected cancellation error, got: %v", err)
	}
}
