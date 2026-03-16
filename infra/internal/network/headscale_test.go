package network

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestHeadscaleProviderName(t *testing.T) {
	p := NewHeadscaleProvider("https://hs.example.com", "key", "user")
	if p.Name() != "headscale" {
		t.Errorf("Name() = %q, want \"headscale\"", p.Name())
	}
}

func TestHeadscaleProviderDefaultUser(t *testing.T) {
	p := NewHeadscaleProvider("https://hs.example.com", "key", "")
	if p.user != "tanrenai" {
		t.Errorf("default user = %q, want \"tanrenai\"", p.user)
	}
}

func TestHeadscaleProviderStripsTrailingSlash(t *testing.T) {
	p := NewHeadscaleProvider("https://hs.example.com/", "key", "user")
	if strings.HasSuffix(p.baseURL, "/") {
		t.Errorf("baseURL should not have trailing slash, got %q", p.baseURL)
	}
}

func TestHeadscaleInstallCommandsContents(t *testing.T) {
	p := NewHeadscaleProvider("https://hs.example.com", "api-key", "myuser")
	cmds := p.InstallCommands("authkey-abc", "my-gpu-host")

	if len(cmds) != 2 {
		t.Fatalf("InstallCommands() returned %d commands, want 2", len(cmds))
	}

	// First command: install tailscale
	if !strings.Contains(cmds[0], "tailscale.com/install.sh") {
		t.Errorf("cmd[0] should install tailscale, got %q", cmds[0])
	}

	// Second command: join headscale
	if !strings.Contains(cmds[1], "--login-server https://hs.example.com") {
		t.Errorf("cmd[1] should use --login-server, got: %q", cmds[1])
	}
	if !strings.Contains(cmds[1], "--authkey authkey-abc") {
		t.Errorf("cmd[1] should contain --authkey authkey-abc")
	}
	if !strings.Contains(cmds[1], "--hostname my-gpu-host") {
		t.Errorf("cmd[1] should contain --hostname my-gpu-host")
	}
	if !strings.Contains(cmds[1], "tailscaled") {
		t.Errorf("cmd[1] should start tailscaled daemon")
	}
	if !strings.Contains(cmds[1], "userspace-networking") {
		t.Errorf("cmd[1] should use userspace-networking")
	}
}

func TestHeadscaleGenerateAuthKey(t *testing.T) {
	// Track which endpoints were called
	var getUsersCalled, createKeyCalled bool

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") != "Bearer test-api-key" {
			http.Error(w, "unauthorized", http.StatusUnauthorized)

			return
		}

		switch r.URL.Path {
		case "/api/v1/user":
			getUsersCalled = true
			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(map[string]any{
				"users": []map[string]any{
					{"id": "42", "name": "tanrenai"},
				},
			})

		case "/api/v1/preauthkey":
			createKeyCalled = true
			if r.Method != http.MethodPost {
				t.Errorf("preauthkey expected POST, got %s", r.Method)
			}
			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(map[string]any{
				"preAuthKey": map[string]any{
					"key": "generated-auth-key-xyz",
				},
			})

		default:
			http.NotFound(w, r)
		}
	}))
	defer srv.Close()

	p := &HeadscaleProvider{
		baseURL:    srv.URL,
		apiKey:     "test-api-key",
		user:       "tanrenai",
		httpClient: srv.Client(),
	}

	key, err := p.GenerateAuthKey(context.Background())
	if err != nil {
		t.Fatalf("GenerateAuthKey() error: %v", err)
	}
	if key != "generated-auth-key-xyz" {
		t.Errorf("GenerateAuthKey() = %q, want \"generated-auth-key-xyz\"", key)
	}
	if !getUsersCalled {
		t.Error("expected /api/v1/user to be called")
	}
	if !createKeyCalled {
		t.Error("expected /api/v1/preauthkey to be called")
	}
}

func TestHeadscaleGenerateAuthKeyUserNotFound(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/api/v1/user" {
			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(map[string]any{
				"users": []map[string]any{
					{"id": "1", "name": "other-user"},
				},
			})
		} else {
			http.NotFound(w, r)
		}
	}))
	defer srv.Close()

	p := &HeadscaleProvider{
		baseURL:    srv.URL,
		apiKey:     "key",
		user:       "nonexistent",
		httpClient: srv.Client(),
	}

	_, err := p.GenerateAuthKey(context.Background())
	if err == nil {
		t.Error("expected error when user not found")
	}
	if !strings.Contains(err.Error(), "nonexistent") {
		t.Errorf("error should mention missing user, got: %v", err)
	}
}

func TestHeadscaleGenerateAuthKeyAPIError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "internal server error", http.StatusInternalServerError)
	}))
	defer srv.Close()

	p := &HeadscaleProvider{
		baseURL:    srv.URL,
		apiKey:     "key",
		user:       "user",
		httpClient: srv.Client(),
	}

	_, err := p.GenerateAuthKey(context.Background())
	if err == nil {
		t.Error("expected error on 500 response")
	}
}

func TestHeadscaleListNodes(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") != "Bearer mykey" {
			http.Error(w, "unauthorized", http.StatusUnauthorized)

			return
		}
		if r.URL.Path != "/api/v1/node" {
			http.NotFound(w, r)

			return
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"nodes": []map[string]any{
				{
					"givenName":   "gpu-node-1",
					"ipAddresses": []string{"100.64.0.1"},
					"online":      true,
				},
				{
					"givenName":   "gpu-node-2",
					"ipAddresses": []string{"100.64.0.2"},
					"online":      false,
				},
			},
		})
	}))
	defer srv.Close()

	p := &HeadscaleProvider{
		baseURL:    srv.URL,
		apiKey:     "mykey",
		user:       "user",
		httpClient: srv.Client(),
	}

	nodes, err := p.ListNodes(context.Background())
	if err != nil {
		t.Fatalf("ListNodes() error: %v", err)
	}
	if len(nodes) != 2 {
		t.Fatalf("ListNodes() returned %d nodes, want 2", len(nodes))
	}

	if nodes[0].Name != "gpu-node-1" {
		t.Errorf("nodes[0].Name = %q, want \"gpu-node-1\"", nodes[0].Name)
	}
	if !nodes[0].Online {
		t.Error("nodes[0].Online should be true")
	}
	if len(nodes[0].IPs) == 0 || nodes[0].IPs[0] != "100.64.0.1" {
		t.Errorf("nodes[0].IPs = %v, want [\"100.64.0.1\"]", nodes[0].IPs)
	}

	if nodes[1].Name != "gpu-node-2" {
		t.Errorf("nodes[1].Name = %q, want \"gpu-node-2\"", nodes[1].Name)
	}
	if nodes[1].Online {
		t.Error("nodes[1].Online should be false")
	}
}

func TestHeadscaleListNodesAPIError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "forbidden", http.StatusForbidden)
	}))
	defer srv.Close()

	p := &HeadscaleProvider{
		baseURL:    srv.URL,
		apiKey:     "key",
		user:       "user",
		httpClient: srv.Client(),
	}

	_, err := p.ListNodes(context.Background())
	if err == nil {
		t.Error("expected error on 403 response")
	}
}

func TestHeadscaleListNodesEmpty(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{"nodes": []any{}})
	}))
	defer srv.Close()

	p := &HeadscaleProvider{
		baseURL:    srv.URL,
		apiKey:     "key",
		user:       "user",
		httpClient: srv.Client(),
	}

	nodes, err := p.ListNodes(context.Background())
	if err != nil {
		t.Fatalf("ListNodes() error: %v", err)
	}
	if len(nodes) != 0 {
		t.Errorf("ListNodes() = %v, want empty slice", nodes)
	}
}

func TestHeadscaleGenerateAuthKeyBearerToken(t *testing.T) {
	var receivedAuth string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		receivedAuth = r.Header.Get("Authorization")
		w.Header().Set("Content-Type", "application/json")
		if r.URL.Path == "/api/v1/user" {
			json.NewEncoder(w).Encode(map[string]any{
				"users": []map[string]any{{"id": "1", "name": "user"}},
			})
		} else {
			json.NewEncoder(w).Encode(map[string]any{
				"preAuthKey": map[string]any{"key": "k"},
			})
		}
	}))
	defer srv.Close()

	p := &HeadscaleProvider{
		baseURL:    srv.URL,
		apiKey:     "secret-api-key",
		user:       "user",
		httpClient: srv.Client(),
	}

	_, _ = p.GenerateAuthKey(context.Background())
	if receivedAuth != "Bearer secret-api-key" {
		t.Errorf("Authorization header = %q, want \"Bearer secret-api-key\"", receivedAuth)
	}
}
