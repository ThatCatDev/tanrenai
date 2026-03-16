package deploy

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"
	"unsafe"

	"github.com/ThatCatDev/tanrenai/infra/internal/config"
	"github.com/ThatCatDev/tanrenai/infra/internal/network"
	"github.com/ThatCatDev/tanrenai/infra/internal/vastai"
)

// deployRedirectTransport redirects all vast.ai API requests to a test server.
type deployRedirectTransport struct {
	target string
	base   http.RoundTripper
}

func (rt *deployRedirectTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	cloned := req.Clone(req.Context())
	cloned.URL.Scheme = "http"
	cloned.URL.Host = strings.TrimPrefix(rt.target, "http://")
	if rt.base != nil {
		return rt.base.RoundTrip(cloned)
	}

	return http.DefaultTransport.RoundTrip(cloned)
}

// overrideVastaiClientTransport replaces the httpClient inside a *vastai.Client using unsafe reflect.
// This is acceptable for tests since we need to redirect requests without modifying production code.
func overrideVastaiClientTransport(c *vastai.Client, srv *httptest.Server) {
	v := reflect.ValueOf(c).Elem()
	f := v.FieldByName("httpClient")
	// httpClient is a *http.Client pointer field. We get a pointer to the field
	// then dereference it to set a new *http.Client value.
	// The field type is *http.Client, stored as a pointer.
	fp := (**http.Client)(unsafe.Pointer(f.UnsafeAddr()))
	newClient := &http.Client{
		Transport: &deployRedirectTransport{target: srv.URL},
	}
	*fp = newClient
}

// TestResolveInstanceWithExistingRunningInstance tests resolveInstance when VastaiInstance is set
// and the instance is already running.
func TestResolveInstanceWithExistingRunningInstance(t *testing.T) {
	inst := vastai.Instance{ID: 111, Status: "running", GPUName: "A100", SSHHost: "1.2.3.4", SSHPort: 22}

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"instances": []vastai.Instance{inst},
		})
	}))
	defer srv.Close()

	client := vastai.NewClient("key")
	overrideVastaiClientTransport(client, srv)

	cfg := config.Defaults()
	cfg.VastaiInstance = "111"

	d := &Deployer{
		vastai:  client,
		network: network.NewNoneProvider(),
		cfg:     cfg,
		output:  &bytes.Buffer{},
	}

	got, err := d.resolveInstance(context.Background())
	if err != nil {
		t.Fatalf("resolveInstance() error: %v", err)
	}
	if got == nil {
		t.Fatal("resolveInstance() returned nil")
	}
	if got.ID != 111 {
		t.Errorf("resolved instance ID = %d, want 111", got.ID)
	}
	if got.Status != "running" {
		t.Errorf("resolved instance status = %q, want \"running\"", got.Status)
	}
}

// TestResolveInstanceStartsExitedInstance verifies resolveInstance starts an exited instance.
func TestResolveInstanceStartsExitedInstance(t *testing.T) {
	callCount := 0
	inst := vastai.Instance{ID: 222, Status: "exited", GPUName: "RTX 4090"}
	runningInst := vastai.Instance{ID: 222, Status: "running", GPUName: "RTX 4090", SSHHost: "5.6.7.8", SSHPort: 22}

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")

		switch r.Method {
		case http.MethodGet:
			callCount++
			if callCount == 1 {
				// First GET: return exited instance
				json.NewEncoder(w).Encode(map[string]any{"instances": []vastai.Instance{inst}})
			} else {
				// Subsequent GETs: return running instance
				json.NewEncoder(w).Encode(map[string]any{"instances": []vastai.Instance{runningInst}})
			}
		case http.MethodPut:
			// StartInstance call
			json.NewEncoder(w).Encode(map[string]any{"success": true})
		}
	}))
	defer srv.Close()

	client := vastai.NewClient("key")
	overrideVastaiClientTransport(client, srv)

	cfg := config.Defaults()
	cfg.VastaiInstance = "222"

	d := &Deployer{
		vastai:  client,
		network: network.NewNoneProvider(),
		cfg:     cfg,
		output:  &bytes.Buffer{},
	}

	got, err := d.resolveInstance(context.Background())
	if err != nil {
		t.Fatalf("resolveInstance() error: %v", err)
	}
	if got == nil {
		t.Fatal("resolveInstance() returned nil")
	}
	if got.Status != "running" {
		t.Errorf("resolved instance status = %q, want \"running\"", got.Status)
	}
}

// TestResolveInstanceGetError verifies resolveInstance propagates GetInstance errors.
func TestResolveInstanceGetError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "not found", http.StatusNotFound)
	}))
	defer srv.Close()

	client := vastai.NewClient("key")
	overrideVastaiClientTransport(client, srv)

	cfg := config.Defaults()
	cfg.VastaiInstance = "999"

	d := &Deployer{
		vastai:  client,
		network: network.NewNoneProvider(),
		cfg:     cfg,
		output:  &bytes.Buffer{},
	}

	_, err := d.resolveInstance(context.Background())
	if err == nil {
		t.Error("expected error when GetInstance fails")
	}
}

// TestResolveInstanceStartError verifies resolveInstance propagates StartInstance errors.
func TestResolveInstanceStartError(t *testing.T) {
	callCount := 0
	inst := vastai.Instance{ID: 333, Status: "exited", GPUName: "RTX 4090"}

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		switch r.Method {
		case http.MethodGet:
			callCount++
			json.NewEncoder(w).Encode(map[string]any{"instances": []vastai.Instance{inst}})
		case http.MethodPut:
			// StartInstance fails
			http.Error(w, "server error", http.StatusInternalServerError)
		}
	}))
	defer srv.Close()

	client := vastai.NewClient("key")
	overrideVastaiClientTransport(client, srv)

	cfg := config.Defaults()
	cfg.VastaiInstance = "333"

	d := &Deployer{
		vastai:  client,
		network: network.NewNoneProvider(),
		cfg:     cfg,
		output:  &bytes.Buffer{},
	}

	_, err := d.resolveInstance(context.Background())
	if err == nil {
		t.Error("expected error when StartInstance fails")
	}
	if !strings.Contains(err.Error(), "start instance") {
		t.Errorf("error should mention start instance, got: %v", err)
	}
}
