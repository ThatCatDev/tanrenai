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

// TestTailscaleFindDeviceFound verifies findDevice returns the IP when the device is present.
func TestTailscaleFindDeviceFound(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"devices": []map[string]any{
				{
					"hostname":  "gpu-node-1",
					"addresses": []string{"100.64.0.5"},
				},
			},
		})
	}))
	defer srv.Close()

	p := &TailscaleProvider{
		authKey:    "tskey-123",
		httpClient: newTailscaleTestClient(srv),
	}

	ip, err := p.findDevice(context.Background(), "gpu-node-1")
	if err != nil {
		t.Fatalf("findDevice() error: %v", err)
	}
	if ip != "100.64.0.5" {
		t.Errorf("findDevice() = %q, want \"100.64.0.5\"", ip)
	}
}

// TestTailscaleFindDeviceNotFound verifies findDevice returns empty string when device is absent.
func TestTailscaleFindDeviceNotFound(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"devices": []map[string]any{
				{
					"hostname":  "other-device",
					"addresses": []string{"100.64.0.1"},
				},
			},
		})
	}))
	defer srv.Close()

	p := &TailscaleProvider{
		authKey:    "tskey-123",
		httpClient: newTailscaleTestClient(srv),
	}

	ip, err := p.findDevice(context.Background(), "gpu-node-1")
	if err != nil {
		t.Fatalf("findDevice() unexpected error: %v", err)
	}
	if ip != "" {
		t.Errorf("findDevice() = %q, want empty string for missing device", ip)
	}
}

// TestTailscaleFindDeviceNoAddresses verifies findDevice returns empty string when device has no addresses.
func TestTailscaleFindDeviceNoAddresses(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"devices": []map[string]any{
				{
					"hostname":  "gpu-node-1",
					"addresses": []string{}, // no IPs
				},
			},
		})
	}))
	defer srv.Close()

	p := &TailscaleProvider{
		authKey:    "tskey-123",
		httpClient: newTailscaleTestClient(srv),
	}

	ip, err := p.findDevice(context.Background(), "gpu-node-1")
	if err != nil {
		t.Fatalf("findDevice() unexpected error: %v", err)
	}
	if ip != "" {
		t.Errorf("findDevice() = %q, want empty string for device with no addresses", ip)
	}
}

// TestTailscaleFindDeviceAPIError verifies findDevice returns an error on non-200 response.
func TestTailscaleFindDeviceAPIError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "unauthorized", http.StatusUnauthorized)
	}))
	defer srv.Close()

	p := &TailscaleProvider{
		authKey:    "tskey-123",
		httpClient: newTailscaleTestClient(srv),
	}

	_, err := p.findDevice(context.Background(), "any-device")
	if err == nil {
		t.Error("expected error on 401 response")
	}
	if !strings.Contains(err.Error(), "401") {
		t.Errorf("error should mention status code, got: %v", err)
	}
}

// TestTailscaleFindDeviceSetsAuthHeader verifies the Authorization header is set correctly.
func TestTailscaleFindDeviceSetsAuthHeader(t *testing.T) {
	var gotAuth string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotAuth = r.Header.Get("Authorization")
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{"devices": []any{}})
	}))
	defer srv.Close()

	p := &TailscaleProvider{
		authKey:    "tskey-my-secret",
		httpClient: newTailscaleTestClient(srv),
	}

	_, _ = p.findDevice(context.Background(), "any")
	if gotAuth != "Bearer tskey-my-secret" {
		t.Errorf("Authorization = %q, want \"Bearer tskey-my-secret\"", gotAuth)
	}
}

// TestTailscaleFindDeviceCaseInsensitive verifies findDevice matches hostname case-insensitively.
func TestTailscaleFindDeviceCaseInsensitive(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"devices": []map[string]any{
				{
					"hostname":  "GPU-NODE-1",
					"addresses": []string{"100.64.0.5"},
				},
			},
		})
	}))
	defer srv.Close()

	p := &TailscaleProvider{
		authKey:    "tskey-123",
		httpClient: newTailscaleTestClient(srv),
	}

	// findDevice uses strings.EqualFold, so case-insensitive
	ip, err := p.findDevice(context.Background(), "gpu-node-1")
	if err != nil {
		t.Fatalf("findDevice() error: %v", err)
	}
	if ip != "100.64.0.5" {
		t.Errorf("findDevice() = %q, want \"100.64.0.5\" (case-insensitive match)", ip)
	}
}

// TestTailscaleFindDeviceDecodeError verifies findDevice returns an error on invalid JSON.
func TestTailscaleFindDeviceDecodeError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte("{invalid json}"))
	}))
	defer srv.Close()

	p := &TailscaleProvider{
		authKey:    "tskey-123",
		httpClient: newTailscaleTestClient(srv),
	}

	_, err := p.findDevice(context.Background(), "any")
	if err == nil {
		t.Error("expected error on invalid JSON response")
	}
}

// TestTailscaleFindDeviceMultiple verifies findDevice picks the right device among many.
func TestTailscaleFindDeviceMultiple(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"devices": []map[string]any{
				{"hostname": "laptop", "addresses": []string{"100.64.0.1"}},
				{"hostname": "server", "addresses": []string{"100.64.0.2"}},
				{"hostname": "gpu-node", "addresses": []string{"100.64.0.9", "fd7a::9"}},
			},
		})
	}))
	defer srv.Close()

	p := &TailscaleProvider{
		authKey:    "tskey-123",
		httpClient: newTailscaleTestClient(srv),
	}

	ip, err := p.findDevice(context.Background(), "gpu-node")
	if err != nil {
		t.Fatalf("findDevice() error: %v", err)
	}
	if ip != "100.64.0.9" {
		t.Errorf("findDevice() = %q, want \"100.64.0.9\"", ip)
	}
}

// TestTailscaleWaitForPeerContextCancelled verifies WaitForPeer returns ctx.Err() when context is pre-cancelled.
func TestTailscaleWaitForPeerContextCancelled(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{"devices": []any{}})
	}))
	defer srv.Close()

	p := &TailscaleProvider{
		authKey:    "tskey-123",
		httpClient: newTailscaleTestClient(srv),
	}

	ctx, cancel := context.WithCancel(context.Background())
	cancel() // pre-cancel

	_, err := p.WaitForPeer(ctx, "gpu-node-1")
	if err == nil {
		t.Error("expected error when context is pre-cancelled")
	}
	if err != context.Canceled {
		t.Errorf("expected context.Canceled, got: %v", err)
	}
}

// TestTailscaleWaitForPeerDeadlineExceeded verifies WaitForPeer exits when deadline passes before 5s ticker.
func TestTailscaleWaitForPeerDeadlineExceeded(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{"devices": []any{}})
	}))
	defer srv.Close()

	p := &TailscaleProvider{
		authKey:    "tskey-123",
		httpClient: newTailscaleTestClient(srv),
	}

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Millisecond)
	defer cancel()

	_, err := p.WaitForPeer(ctx, "gpu-node-1")
	if err == nil {
		t.Error("expected timeout error before 5s ticker fires")
	}
}

// newTailscaleTestClient creates an http.Client that redirects all requests to the test server.
// It does this by using a custom RoundTripper that replaces the host.
func newTailscaleTestClient(srv *httptest.Server) *http.Client {
	return &http.Client{
		Transport: &tailscaleRedirectTransport{target: srv.URL, base: srv.Client().Transport},
	}
}

type tailscaleRedirectTransport struct {
	target string
	base   http.RoundTripper
}

func (rt *tailscaleRedirectTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	cloned := req.Clone(req.Context())
	cloned.URL.Scheme = "http"
	cloned.URL.Host = strings.TrimPrefix(rt.target, "http://")
	if rt.base != nil {
		return rt.base.RoundTrip(cloned)
	}

	return http.DefaultTransport.RoundTrip(cloned)
}
