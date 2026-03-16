package cmd

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"
	"unsafe"

	"github.com/spf13/cobra"

	"github.com/ThatCatDev/tanrenai/infra/internal/network"
)

// makeNetworkAuthKeyCmd creates a fresh network auth-key command for testing.
func makeNetworkAuthKeyCmd() *cobra.Command {
	cmd := &cobra.Command{Use: "auth-key"}
	cmd.Flags().String("headscale-url", "", "")
	cmd.Flags().String("headscale-api-key", "", "")
	cmd.Flags().String("headscale-user", "", "")
	cmd.Flags().Bool("reusable", false, "")
	cmd.Flags().Bool("ephemeral", true, "")

	return cmd
}

// makeNetworkNodesCmd creates a fresh network nodes command for testing.
func makeNetworkNodesCmd() *cobra.Command {
	cmd := &cobra.Command{Use: "nodes"}
	cmd.Flags().String("headscale-url", "", "")
	cmd.Flags().String("headscale-api-key", "", "")
	cmd.Flags().String("headscale-user", "", "")

	return cmd
}

// makeNetworkJoinCmd creates a fresh network join command for testing.
func makeNetworkJoinCmd() *cobra.Command {
	cmd := &cobra.Command{Use: "join"}
	cmd.Flags().String("headscale-url", "", "")
	cmd.Flags().String("headscale-api-key", "", "")
	cmd.Flags().String("headscale-user", "", "")
	cmd.Flags().String("hostname", "", "")

	return cmd
}

// makeHeadscaleProviderWithMock creates a HeadscaleProvider whose HTTP calls go to srv.
func makeHeadscaleProviderWithMock(t *testing.T, srv *httptest.Server) *network.HeadscaleProvider {
	t.Helper()
	p := network.NewHeadscaleProvider(srv.URL, "test-key", "tanrenai")
	// Override the private httpClient using unsafe reflect
	v := reflect.ValueOf(p).Elem()
	f := v.FieldByName("httpClient")
	fp := (**http.Client)(unsafe.Pointer(f.UnsafeAddr()))
	*fp = &http.Client{Transport: &cmdRedirectTransport{target: srv.URL}}

	return p
}

// TestHeadscaleProviderMissingURL tests headscaleProvider exits when URL is missing.
func TestHeadscaleProviderMissingURL(t *testing.T) {
	t.Setenv("HEADSCALE_URL", "")
	t.Setenv("HEADSCALE_API_KEY", "")
	withMockExit(t)

	cmd := makeNetworkAuthKeyCmd()
	_ = cmd.Flags().Set("headscale-api-key", "test-api-key")
	// No headscale-url set

	code := recoverExitPanic(func() {
		headscaleProvider(cmd)
	})

	if code != 1 {
		t.Errorf("expected exit code 1 when URL missing, got %d", code)
	}
}

// TestHeadscaleProviderMissingAPIKey tests headscaleProvider exits when API key is missing.
func TestHeadscaleProviderMissingAPIKey(t *testing.T) {
	t.Setenv("HEADSCALE_URL", "")
	t.Setenv("HEADSCALE_API_KEY", "")
	withMockExit(t)

	cmd := makeNetworkAuthKeyCmd()
	_ = cmd.Flags().Set("headscale-url", "https://headscale.example.com")
	// No headscale-api-key set

	code := recoverExitPanic(func() {
		headscaleProvider(cmd)
	})

	if code != 1 {
		t.Errorf("expected exit code 1 when API key missing, got %d", code)
	}
}

// TestHeadscaleProviderSuccess tests headscaleProvider returns a provider when config is valid.
func TestHeadscaleProviderSuccess(t *testing.T) {
	t.Setenv("HEADSCALE_URL", "")
	t.Setenv("HEADSCALE_API_KEY", "")
	withMockExit(t)

	cmd := makeNetworkAuthKeyCmd()
	_ = cmd.Flags().Set("headscale-url", "https://headscale.example.com")
	_ = cmd.Flags().Set("headscale-api-key", "my-key")
	_ = cmd.Flags().Set("headscale-user", "myuser")

	provider := headscaleProvider(cmd)
	if provider == nil {
		t.Error("expected non-nil provider")
	}
}

// TestHeadscaleProviderDefaultUser tests headscaleProvider uses default user when not set.
func TestHeadscaleProviderDefaultUser(t *testing.T) {
	t.Setenv("HEADSCALE_URL", "")
	t.Setenv("HEADSCALE_API_KEY", "")
	withMockExit(t)

	cmd := makeNetworkAuthKeyCmd()
	_ = cmd.Flags().Set("headscale-url", "https://headscale.example.com")
	_ = cmd.Flags().Set("headscale-api-key", "my-key")
	// No user set — default should be used

	provider := headscaleProvider(cmd)
	if provider == nil {
		t.Error("expected non-nil provider")
	}
}

// TestRunNetworkAuthKeyMissingConfig tests runNetworkAuthKey exits when config is missing.
func TestRunNetworkAuthKeyMissingConfig(t *testing.T) {
	t.Setenv("HEADSCALE_URL", "")
	t.Setenv("HEADSCALE_API_KEY", "")
	withMockExit(t)

	cmd := makeNetworkAuthKeyCmd()

	code := recoverExitPanic(func() {
		runNetworkAuthKey(cmd, nil)
	})

	if code != 1 {
		t.Errorf("expected exit code 1, got %d", code)
	}
}

// TestRunNetworkNodesMissingConfig tests runNetworkNodes exits when config is missing.
func TestRunNetworkNodesMissingConfig(t *testing.T) {
	t.Setenv("HEADSCALE_URL", "")
	t.Setenv("HEADSCALE_API_KEY", "")
	withMockExit(t)

	cmd := makeNetworkNodesCmd()

	code := recoverExitPanic(func() {
		runNetworkNodes(cmd, nil)
	})

	if code != 1 {
		t.Errorf("expected exit code 1, got %d", code)
	}
}

// TestRunNetworkNodesEmpty tests runNetworkNodes when no nodes are returned.
func TestRunNetworkNodesEmpty(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{"nodes": []any{}})
	}))
	defer srv.Close()

	withMockExit(t)

	origNewHS := newHeadscaleProvider
	newHeadscaleProvider = func(baseURL, apiKey, user string) *network.HeadscaleProvider {
		return makeHeadscaleProviderWithMock(t, srv)
	}
	defer func() { newHeadscaleProvider = origNewHS }()

	cmd := makeNetworkNodesCmd()
	_ = cmd.Flags().Set("headscale-url", srv.URL)
	_ = cmd.Flags().Set("headscale-api-key", "test-key")

	stdout, _ := captureOutput(func() {
		runNetworkNodes(cmd, nil)
	})

	if !strings.Contains(stdout, "No nodes found") {
		t.Errorf("expected 'No nodes found', got: %q", stdout)
	}
}

// TestRunNetworkNodesWithNodes tests runNetworkNodes when nodes are returned.
func TestRunNetworkNodesWithNodes(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		if strings.Contains(r.URL.Path, "/node") {
			json.NewEncoder(w).Encode(map[string]any{
				"nodes": []map[string]any{
					{"givenName": "gpu-node-1", "online": true, "ipAddresses": []string{"100.64.0.1"}},
					{"givenName": "gpu-node-2", "online": false, "ipAddresses": []string{"100.64.0.2"}},
				},
			})
		}
	}))
	defer srv.Close()

	withMockExit(t)

	origNewHS := newHeadscaleProvider
	newHeadscaleProvider = func(baseURL, apiKey, user string) *network.HeadscaleProvider {
		return makeHeadscaleProviderWithMock(t, srv)
	}
	defer func() { newHeadscaleProvider = origNewHS }()

	cmd := makeNetworkNodesCmd()
	_ = cmd.Flags().Set("headscale-url", srv.URL)
	_ = cmd.Flags().Set("headscale-api-key", "test-key")

	stdout, _ := captureOutput(func() {
		runNetworkNodes(cmd, nil)
	})

	if !strings.Contains(stdout, "gpu-node-1") {
		t.Errorf("output should contain gpu-node-1, got: %q", stdout)
	}
	if !strings.Contains(stdout, "online") {
		t.Errorf("output should contain 'online', got: %q", stdout)
	}
	if !strings.Contains(stdout, "offline") {
		t.Errorf("output should contain 'offline', got: %q", stdout)
	}
}

// TestRunNetworkAuthKeySuccess tests runNetworkAuthKey when the server returns a key.
func TestRunNetworkAuthKeySuccess(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		if strings.Contains(r.URL.Path, "/user") && r.Method == http.MethodGet {
			json.NewEncoder(w).Encode(map[string]any{
				"users": []map[string]any{
					{"id": "1", "name": "tanrenai"},
				},
			})
		} else if strings.Contains(r.URL.Path, "/preauthkey") {
			json.NewEncoder(w).Encode(map[string]any{
				"preAuthKey": map[string]any{
					"key": "tskey-auth-test123",
				},
			})
		}
	}))
	defer srv.Close()

	withMockExit(t)

	origNewHS := newHeadscaleProvider
	newHeadscaleProvider = func(baseURL, apiKey, user string) *network.HeadscaleProvider {
		return makeHeadscaleProviderWithMock(t, srv)
	}
	defer func() { newHeadscaleProvider = origNewHS }()

	cmd := makeNetworkAuthKeyCmd()
	_ = cmd.Flags().Set("headscale-url", srv.URL)
	_ = cmd.Flags().Set("headscale-api-key", "test-key")

	stdout, _ := captureOutput(func() {
		runNetworkAuthKey(cmd, nil)
	})

	if !strings.Contains(stdout, "tskey-auth-test123") {
		t.Errorf("output should contain the auth key, got: %q", stdout)
	}
}

// TestRunNetworkAuthKeyError tests runNetworkAuthKey exits when server returns error.
func TestRunNetworkAuthKeyError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "unauthorized", http.StatusUnauthorized)
	}))
	defer srv.Close()

	withMockExit(t)

	origNewHS := newHeadscaleProvider
	newHeadscaleProvider = func(baseURL, apiKey, user string) *network.HeadscaleProvider {
		return makeHeadscaleProviderWithMock(t, srv)
	}
	defer func() { newHeadscaleProvider = origNewHS }()

	cmd := makeNetworkAuthKeyCmd()
	_ = cmd.Flags().Set("headscale-url", srv.URL)
	_ = cmd.Flags().Set("headscale-api-key", "test-key")

	code := recoverExitPanic(func() {
		runNetworkAuthKey(cmd, nil)
	})

	if code != 1 {
		t.Errorf("expected exit code 1 on auth key error, got %d", code)
	}
}

// TestRunNetworkNodesError tests runNetworkNodes exits when server returns error.
func TestRunNetworkNodesError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "server error", http.StatusInternalServerError)
	}))
	defer srv.Close()

	withMockExit(t)

	origNewHS := newHeadscaleProvider
	newHeadscaleProvider = func(baseURL, apiKey, user string) *network.HeadscaleProvider {
		return makeHeadscaleProviderWithMock(t, srv)
	}
	defer func() { newHeadscaleProvider = origNewHS }()

	cmd := makeNetworkNodesCmd()
	_ = cmd.Flags().Set("headscale-url", srv.URL)
	_ = cmd.Flags().Set("headscale-api-key", "test-key")

	code := recoverExitPanic(func() {
		runNetworkNodes(cmd, nil)
	})

	if code != 1 {
		t.Errorf("expected exit code 1 on nodes error, got %d", code)
	}
}

// TestRunNetworkJoinMissingConfig tests runNetworkJoin exits when config is missing.
func TestRunNetworkJoinMissingConfig(t *testing.T) {
	t.Setenv("HEADSCALE_URL", "")
	t.Setenv("HEADSCALE_API_KEY", "")
	withMockExit(t)

	cmd := makeNetworkJoinCmd()

	code := recoverExitPanic(func() {
		runNetworkJoin(cmd, nil)
	})

	if code != 1 {
		t.Errorf("expected exit code 1, got %d", code)
	}
}

// TestRunNetworkJoinAuthKeyError tests runNetworkJoin exits when GenerateAuthKey fails.
func TestRunNetworkJoinAuthKeyError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "unauthorized", http.StatusUnauthorized)
	}))
	defer srv.Close()

	withMockExit(t)

	origNewHS := newHeadscaleProvider
	newHeadscaleProvider = func(baseURL, apiKey, user string) *network.HeadscaleProvider {
		return makeHeadscaleProviderWithMock(t, srv)
	}
	defer func() { newHeadscaleProvider = origNewHS }()

	cmd := makeNetworkJoinCmd()
	_ = cmd.Flags().Set("headscale-url", srv.URL)
	_ = cmd.Flags().Set("headscale-api-key", "test-key")
	_ = cmd.Flags().Set("hostname", "my-node")

	code := recoverExitPanic(func() {
		runNetworkJoin(cmd, nil)
	})

	if code != 1 {
		t.Errorf("expected exit code 1 when GenerateAuthKey fails, got %d", code)
	}
}
