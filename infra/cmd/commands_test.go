package cmd

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"reflect"
	"strings"
	"testing"
	"unsafe"

	"github.com/spf13/cobra"

	"github.com/ThatCatDev/tanrenai/infra/internal/vastai"
)

// cmdRedirectTransport redirects all requests to a test server.
type cmdRedirectTransport struct {
	target string
}

func (rt *cmdRedirectTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	cloned := req.Clone(req.Context())
	cloned.URL.Scheme = "http"
	cloned.URL.Host = strings.TrimPrefix(rt.target, "http://")

	return http.DefaultTransport.RoundTrip(cloned)
}

// makeMockClient creates a vastai.Client whose HTTP requests go to the test server.
func makeMockClient(t *testing.T, srv *httptest.Server) *vastai.Client {
	t.Helper()
	c := vastai.NewClient("test-key")
	v := reflect.ValueOf(c).Elem()
	f := v.FieldByName("httpClient")
	fp := (**http.Client)(unsafe.Pointer(f.UnsafeAddr()))
	*fp = &http.Client{Transport: &cmdRedirectTransport{target: srv.URL}}

	return c
}

// withMockExit replaces osExit with a function that panics with a sentinel value,
// then restores the original after the test.
type exitPanic struct{ code int }

func withMockExit(t *testing.T) {
	t.Helper()
	orig := osExit
	osExit = func(code int) { panic(exitPanic{code}) }
	t.Cleanup(func() { osExit = orig })
}

// recoverExitPanic recovers from an exitPanic and returns the exit code.
// Returns -1 if no panic occurred.
func recoverExitPanic(fn func()) (code int) {
	code = -1
	defer func() {
		if r := recover(); r != nil {
			if ep, ok := r.(exitPanic); ok {
				code = ep.code
			} else {
				panic(r)
			}
		}
	}()
	fn()

	return
}

// captureOutput captures both stdout and stderr during fn.
func captureOutput(fn func()) (stdout, stderr string) {
	rOut, wOut, _ := os.Pipe()
	rErr, wErr, _ := os.Pipe()
	origOut := os.Stdout
	origErr := os.Stderr
	os.Stdout = wOut
	os.Stderr = wErr

	fn()

	_ = wOut.Close()
	_ = wErr.Close()
	os.Stdout = origOut
	os.Stderr = origErr

	var bufOut, bufErr bytes.Buffer
	_, _ = io.Copy(&bufOut, rOut)
	_, _ = io.Copy(&bufErr, rErr)

	return bufOut.String(), bufErr.String()
}

// makeListCmd creates a fresh list command for testing.
func makeListCmd() *cobra.Command {
	cmd := &cobra.Command{Use: "list"}
	cmd.Flags().String("vastai-api-key", "", "")

	return cmd
}

// makeStatusCmd creates a fresh status command for testing.
func makeStatusCmd() *cobra.Command {
	cmd := &cobra.Command{Use: "status"}
	cmd.Flags().String("vastai-api-key", "", "")
	cmd.Flags().String("vastai-instance-id", "", "")

	return cmd
}

// makeDestroyCmd creates a fresh destroy command for testing.
func makeDestroyCmd() *cobra.Command {
	cmd := &cobra.Command{Use: "destroy"}
	cmd.Flags().String("vastai-api-key", "", "")
	cmd.Flags().String("vastai-instance-id", "", "")

	return cmd
}

// TestRunListSuccess tests runList with a mock server returning instances.
func TestRunListSuccess(t *testing.T) {
	instances := []vastai.Instance{
		{ID: 1, Status: "running", GPUName: "RTX 4090", NumGPUs: 1, CostPerHr: 0.5, SSHHost: "1.2.3.4", SSHPort: 22},
		{ID: 2, Status: "exited", GPUName: "A100", NumGPUs: 2, CostPerHr: 1.0},
	}

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{"instances": instances})
	}))
	defer srv.Close()

	origNewClient := newVastaiClient
	newVastaiClient = func(apiKey string) *vastai.Client {
		return makeMockClient(t, srv)
	}
	defer func() { newVastaiClient = origNewClient }()

	withMockExit(t)

	cmd := makeListCmd()
	_ = cmd.Flags().Set("vastai-api-key", "test-key")

	stdout, _ := captureOutput(func() {
		runList(cmd, nil)
	})

	if !strings.Contains(stdout, "1") {
		t.Errorf("output should contain instance ID 1, got: %q", stdout)
	}
	if !strings.Contains(stdout, "RTX 4090") {
		t.Errorf("output should contain GPU name, got: %q", stdout)
	}
	if !strings.Contains(stdout, "ssh://1.2.3.4:22") {
		t.Errorf("output should contain SSH info for instance 1, got: %q", stdout)
	}
}

// TestRunListEmpty tests runList when no instances are returned.
func TestRunListEmpty(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{"instances": []vastai.Instance{}})
	}))
	defer srv.Close()

	origNewClient := newVastaiClient
	newVastaiClient = func(apiKey string) *vastai.Client {
		return makeMockClient(t, srv)
	}
	defer func() { newVastaiClient = origNewClient }()

	withMockExit(t)

	cmd := makeListCmd()
	_ = cmd.Flags().Set("vastai-api-key", "test-key")

	stdout, _ := captureOutput(func() {
		runList(cmd, nil)
	})

	if !strings.Contains(stdout, "No instances found") {
		t.Errorf("output should say no instances found, got: %q", stdout)
	}
}

// TestRunListMissingAPIKey tests runList exits when API key is missing.
func TestRunListMissingAPIKey(t *testing.T) {
	t.Setenv("VASTAI_API_KEY", "")
	withMockExit(t)

	cmd := makeListCmd()

	code := recoverExitPanic(func() {
		runList(cmd, nil)
	})

	if code != 1 {
		t.Errorf("expected exit code 1, got %d", code)
	}
}

// TestRunListAPIError tests runList exits when the API returns an error.
func TestRunListAPIError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "unauthorized", http.StatusUnauthorized)
	}))
	defer srv.Close()

	origNewClient := newVastaiClient
	newVastaiClient = func(apiKey string) *vastai.Client {
		return makeMockClient(t, srv)
	}
	defer func() { newVastaiClient = origNewClient }()

	withMockExit(t)

	cmd := makeListCmd()
	_ = cmd.Flags().Set("vastai-api-key", "test-key")

	code := recoverExitPanic(func() {
		runList(cmd, nil)
	})

	if code != 1 {
		t.Errorf("expected exit code 1 on API error, got %d", code)
	}
}

// TestRunStatusSingleInstance tests runStatus with --vastai-instance-id.
func TestRunStatusSingleInstance(t *testing.T) {
	inst := vastai.Instance{ID: 42, Status: "running", GPUName: "H100", NumGPUs: 4, CostPerHr: 2.5, SSHHost: "10.0.0.1", SSHPort: 2222}

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{"instances": []vastai.Instance{inst}})
	}))
	defer srv.Close()

	origNewClient := newVastaiClient
	newVastaiClient = func(apiKey string) *vastai.Client {
		return makeMockClient(t, srv)
	}
	defer func() { newVastaiClient = origNewClient }()

	withMockExit(t)

	cmd := makeStatusCmd()
	_ = cmd.Flags().Set("vastai-api-key", "test-key")
	_ = cmd.Flags().Set("vastai-instance-id", "42")

	stdout, _ := captureOutput(func() {
		runStatus(cmd, nil)
	})

	if !strings.Contains(stdout, "42") {
		t.Errorf("output should contain ID 42, got: %q", stdout)
	}
	if !strings.Contains(stdout, "H100") {
		t.Errorf("output should contain GPU name, got: %q", stdout)
	}
}

// TestRunStatusAllInstances tests runStatus listing all instances.
func TestRunStatusAllInstances(t *testing.T) {
	instances := []vastai.Instance{
		{ID: 10, Status: "running", GPUName: "A100", NumGPUs: 1, CostPerHr: 1.0},
		{ID: 11, Status: "loading", GPUName: "RTX 3090", NumGPUs: 2, CostPerHr: 0.3},
	}

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{"instances": instances})
	}))
	defer srv.Close()

	origNewClient := newVastaiClient
	newVastaiClient = func(apiKey string) *vastai.Client {
		return makeMockClient(t, srv)
	}
	defer func() { newVastaiClient = origNewClient }()

	withMockExit(t)

	cmd := makeStatusCmd()
	_ = cmd.Flags().Set("vastai-api-key", "test-key")

	stdout, _ := captureOutput(func() {
		runStatus(cmd, nil)
	})

	if !strings.Contains(stdout, "A100") {
		t.Errorf("output should contain A100, got: %q", stdout)
	}
	if !strings.Contains(stdout, "RTX 3090") {
		t.Errorf("output should contain RTX 3090, got: %q", stdout)
	}
	if !strings.Contains(stdout, "---") {
		t.Errorf("output should have separator between instances, got: %q", stdout)
	}
}

// TestRunStatusEmpty tests runStatus when no instances exist.
func TestRunStatusEmpty(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{"instances": []vastai.Instance{}})
	}))
	defer srv.Close()

	origNewClient := newVastaiClient
	newVastaiClient = func(apiKey string) *vastai.Client {
		return makeMockClient(t, srv)
	}
	defer func() { newVastaiClient = origNewClient }()

	withMockExit(t)

	cmd := makeStatusCmd()
	_ = cmd.Flags().Set("vastai-api-key", "test-key")

	stdout, _ := captureOutput(func() {
		runStatus(cmd, nil)
	})

	if !strings.Contains(stdout, "No instances found") {
		t.Errorf("expected no instances message, got: %q", stdout)
	}
}

// TestRunStatusMissingAPIKey tests runStatus exits when API key is missing.
func TestRunStatusMissingAPIKey(t *testing.T) {
	t.Setenv("VASTAI_API_KEY", "")
	withMockExit(t)

	cmd := makeStatusCmd()

	code := recoverExitPanic(func() {
		runStatus(cmd, nil)
	})

	if code != 1 {
		t.Errorf("expected exit code 1, got %d", code)
	}
}

// TestRunStatusGetInstanceError tests runStatus exits when GetInstance fails.
func TestRunStatusGetInstanceError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "forbidden", http.StatusForbidden)
	}))
	defer srv.Close()

	origNewClient := newVastaiClient
	newVastaiClient = func(apiKey string) *vastai.Client {
		return makeMockClient(t, srv)
	}
	defer func() { newVastaiClient = origNewClient }()

	withMockExit(t)

	cmd := makeStatusCmd()
	_ = cmd.Flags().Set("vastai-api-key", "test-key")
	_ = cmd.Flags().Set("vastai-instance-id", "42")

	code := recoverExitPanic(func() {
		runStatus(cmd, nil)
	})

	if code != 1 {
		t.Errorf("expected exit code 1 on API error, got %d", code)
	}
}

// TestRunStatusListError tests runStatus exits when ListInstances fails.
func TestRunStatusListError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "server error", http.StatusInternalServerError)
	}))
	defer srv.Close()

	origNewClient := newVastaiClient
	newVastaiClient = func(apiKey string) *vastai.Client {
		return makeMockClient(t, srv)
	}
	defer func() { newVastaiClient = origNewClient }()

	withMockExit(t)

	cmd := makeStatusCmd()
	_ = cmd.Flags().Set("vastai-api-key", "test-key")

	code := recoverExitPanic(func() {
		runStatus(cmd, nil)
	})

	if code != 1 {
		t.Errorf("expected exit code 1 on list error, got %d", code)
	}
}

// TestRunDestroySuccess tests runDestroy with a mock server.
func TestRunDestroySuccess(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		json.NewEncoder(w).Encode(map[string]any{})
	}))
	defer srv.Close()

	origNewClient := newVastaiClient
	newVastaiClient = func(apiKey string) *vastai.Client {
		return makeMockClient(t, srv)
	}
	defer func() { newVastaiClient = origNewClient }()

	withMockExit(t)

	cmd := makeDestroyCmd()
	_ = cmd.Flags().Set("vastai-api-key", "test-key")
	_ = cmd.Flags().Set("vastai-instance-id", "123")

	stdout, _ := captureOutput(func() {
		runDestroy(cmd, nil)
	})

	if !strings.Contains(stdout, "Destroying") {
		t.Errorf("output should mention destroying, got: %q", stdout)
	}
	if !strings.Contains(stdout, "destroyed") {
		t.Errorf("output should confirm destroyed, got: %q", stdout)
	}
}

// TestRunDestroyMissingAPIKey tests runDestroy exits when API key is missing.
func TestRunDestroyMissingAPIKey(t *testing.T) {
	t.Setenv("VASTAI_API_KEY", "")
	withMockExit(t)

	cmd := makeDestroyCmd()
	_ = cmd.Flags().Set("vastai-instance-id", "123")

	code := recoverExitPanic(func() {
		runDestroy(cmd, nil)
	})

	if code != 1 {
		t.Errorf("expected exit code 1, got %d", code)
	}
}

// TestRunDestroyMissingInstanceID tests runDestroy exits when instance ID is missing.
func TestRunDestroyMissingInstanceID(t *testing.T) {
	t.Setenv("VASTAI_API_KEY", "")
	withMockExit(t)

	cmd := makeDestroyCmd()
	// Set API key but no instance ID
	_ = cmd.Flags().Set("vastai-api-key", "test-key")

	code := recoverExitPanic(func() {
		runDestroy(cmd, nil)
	})

	if code != 1 {
		t.Errorf("expected exit code 1, got %d", code)
	}
}

// TestRunDestroyAPIError tests runDestroy exits when the API returns an error.
func TestRunDestroyAPIError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "not found", http.StatusNotFound)
	}))
	defer srv.Close()

	origNewClient := newVastaiClient
	newVastaiClient = func(apiKey string) *vastai.Client {
		return makeMockClient(t, srv)
	}
	defer func() { newVastaiClient = origNewClient }()

	withMockExit(t)

	cmd := makeDestroyCmd()
	_ = cmd.Flags().Set("vastai-api-key", "test-key")
	_ = cmd.Flags().Set("vastai-instance-id", "999")

	code := recoverExitPanic(func() {
		runDestroy(cmd, nil)
	})

	if code != 1 {
		t.Errorf("expected exit code 1 on API error, got %d", code)
	}
}

// TestExitErrorWritesToStderr verifies exitError writes the message to stderr.
func TestExitErrorWritesToStderr(t *testing.T) {
	withMockExit(t)

	_, stderr := captureOutput(func() {
		recoverExitPanic(func() {
			exitError("something went wrong: %v", "details here")
		})
	})

	if !strings.Contains(stderr, "something went wrong") {
		t.Errorf("stderr should contain error message, got: %q", stderr)
	}
	if !strings.Contains(stderr, "details here") {
		t.Errorf("stderr should contain details, got: %q", stderr)
	}
}
