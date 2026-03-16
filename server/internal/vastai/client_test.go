package vastai

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// newTestClient creates a Client that points to the given test server URL
// instead of the real vast.ai API.
func newTestClient(t *testing.T, handler http.Handler) (*Client, *httptest.Server) {
	t.Helper()
	srv := httptest.NewServer(handler)
	t.Cleanup(srv.Close)

	c := NewClient("test-api-key")
	// Override the baseURL constant by using a wrapper that redirects calls.
	// Since baseURL is a package-level const, we achieve test isolation by
	// substituting the http client's transport.  Instead we use a test server
	// and a custom http.Client that rewrites the host.
	c.httpClient = &http.Client{
		Transport: &rewriteTransport{base: srv.URL, inner: http.DefaultTransport},
	}

	return c, srv
}

// rewriteTransport rewrites all requests to go to `base` instead of the
// real vast.ai API, preserving path and query.
type rewriteTransport struct {
	base  string
	inner http.RoundTripper
}

func (rt *rewriteTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	// Clone request and replace host
	cloned := req.Clone(req.Context())
	cloned.URL.Scheme = "http"
	// Strip scheme+host from base URL for the host part
	host := strings.TrimPrefix(rt.base, "http://")
	host = strings.TrimPrefix(host, "https://")
	cloned.URL.Host = host
	// Strip the vast.ai base path prefix so paths resolve correctly
	cloned.URL.Path = strings.TrimPrefix(cloned.URL.Path, "/api/v0")

	return rt.inner.RoundTrip(cloned)
}

// ---- Auth header ----

func TestAuthHeader(t *testing.T) {
	var gotAuth string

	client, _ := newTestClient(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotAuth = r.Header.Get("Authorization")
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string][]Instance{"instances": {}})
	}))

	_, _ = client.ListInstances(context.Background())

	want := "Bearer test-api-key"
	if gotAuth != want {
		t.Errorf("Authorization header = %q, want %q", gotAuth, want)
	}
}

// ---- ListInstances ----

func TestListInstances(t *testing.T) {
	instances := []Instance{
		{ID: 1, Status: "running", SSHHost: "1.2.3.4", SSHPort: 22, GPUName: "RTX 4090", NumGPUs: 1, CostPerHr: 0.5},
		{ID: 2, Status: "exited", SSHHost: "5.6.7.8", SSHPort: 2222, GPUName: "A100", NumGPUs: 2, CostPerHr: 1.2},
	}

	client, _ := newTestClient(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string][]Instance{"instances": instances})
	}))

	got, err := client.ListInstances(context.Background())
	if err != nil {
		t.Fatalf("ListInstances: %v", err)
	}
	if len(got) != 2 {
		t.Errorf("got %d instances, want 2", len(got))
	}
	if got[0].ID != 1 {
		t.Errorf("first instance ID = %d, want 1", got[0].ID)
	}
	if got[1].GPUName != "A100" {
		t.Errorf("second instance GPUName = %q, want A100", got[1].GPUName)
	}
}

func TestListInstancesEmpty(t *testing.T) {
	client, _ := newTestClient(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string][]Instance{"instances": {}})
	}))

	got, err := client.ListInstances(context.Background())
	if err != nil {
		t.Fatalf("ListInstances: %v", err)
	}
	if len(got) != 0 {
		t.Errorf("expected empty slice, got %d instances", len(got))
	}
}

func TestListInstances500(t *testing.T) {
	client, _ := newTestClient(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "server error", http.StatusInternalServerError)
	}))

	_, err := client.ListInstances(context.Background())
	if err == nil {
		t.Fatal("expected error for 500, got nil")
	}
}

func TestListInstances404(t *testing.T) {
	client, _ := newTestClient(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "not found", http.StatusNotFound)
	}))

	_, err := client.ListInstances(context.Background())
	if err == nil {
		t.Fatal("expected error for 404, got nil")
	}
}

// ---- GetInstance ----

func TestGetInstance(t *testing.T) {
	instances := []Instance{
		{ID: 42, Status: "running", GPUName: "RTX 4090"},
	}

	client, _ := newTestClient(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string][]Instance{"instances": instances})
	}))

	got, err := client.GetInstance(context.Background(), "42")
	if err != nil {
		t.Fatalf("GetInstance: %v", err)
	}
	if got.ID != 42 {
		t.Errorf("instance ID = %d, want 42", got.ID)
	}
	if got.Status != "running" {
		t.Errorf("instance Status = %q, want running", got.Status)
	}
}

func TestGetInstanceNotFound(t *testing.T) {
	// Return a list that does not contain the requested ID
	instances := []Instance{
		{ID: 99, Status: "running"},
	}

	client, _ := newTestClient(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string][]Instance{"instances": instances})
	}))

	_, err := client.GetInstance(context.Background(), "42")
	if err == nil {
		t.Fatal("expected error for missing instance, got nil")
	}
	if !strings.Contains(err.Error(), "42") {
		t.Errorf("error %q should mention ID 42", err.Error())
	}
}

func TestGetInstance500(t *testing.T) {
	client, _ := newTestClient(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "server error", http.StatusInternalServerError)
	}))

	_, err := client.GetInstance(context.Background(), "1")
	if err == nil {
		t.Fatal("expected error for 500, got nil")
	}
}

// ---- StartInstance ----

func TestStartInstance(t *testing.T) {
	var gotBody string

	client, _ := newTestClient(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPut {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)

			return
		}
		buf := new(strings.Builder)
		fmt.Fprintf(buf, "")
		body := make([]byte, 256)
		n, _ := r.Body.Read(body)
		gotBody = string(body[:n])
		w.WriteHeader(http.StatusOK)
	}))

	if err := client.StartInstance(context.Background(), "5"); err != nil {
		t.Fatalf("StartInstance: %v", err)
	}
	if !strings.Contains(gotBody, "running") {
		t.Errorf("StartInstance body %q should contain 'running'", gotBody)
	}
}

func TestStartInstance500(t *testing.T) {
	client, _ := newTestClient(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "cannot start", http.StatusInternalServerError)
	}))

	if err := client.StartInstance(context.Background(), "5"); err == nil {
		t.Fatal("expected error for 500, got nil")
	}
}

// ---- StopInstance ----

func TestStopInstance(t *testing.T) {
	var gotBody string

	client, _ := newTestClient(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPut {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)

			return
		}
		body := make([]byte, 256)
		n, _ := r.Body.Read(body)
		gotBody = string(body[:n])
		w.WriteHeader(http.StatusOK)
	}))

	if err := client.StopInstance(context.Background(), "5"); err != nil {
		t.Fatalf("StopInstance: %v", err)
	}
	if !strings.Contains(gotBody, "stopped") {
		t.Errorf("StopInstance body %q should contain 'stopped'", gotBody)
	}
}

func TestStopInstance500(t *testing.T) {
	client, _ := newTestClient(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "cannot stop", http.StatusInternalServerError)
	}))

	if err := client.StopInstance(context.Background(), "5"); err == nil {
		t.Fatal("expected error for 500, got nil")
	}
}

// ---- DestroyInstance ----

func TestDestroyInstance(t *testing.T) {
	var gotMethod string

	client, _ := newTestClient(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotMethod = r.Method
		w.WriteHeader(http.StatusOK)
	}))

	if err := client.DestroyInstance(context.Background(), "7"); err != nil {
		t.Fatalf("DestroyInstance: %v", err)
	}
	if gotMethod != http.MethodDelete {
		t.Errorf("method = %q, want DELETE", gotMethod)
	}
}

func TestDestroyInstance500(t *testing.T) {
	client, _ := newTestClient(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "cannot destroy", http.StatusInternalServerError)
	}))

	if err := client.DestroyInstance(context.Background(), "7"); err == nil {
		t.Fatal("expected error for 500, got nil")
	}
}
