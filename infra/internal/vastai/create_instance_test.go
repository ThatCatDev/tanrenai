package vastai

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// TestCreateInstanceSuccess verifies CreateInstance sends a PUT and polls until it finds a running instance.
func TestCreateInstanceSuccess(t *testing.T) {
	listCallCount := 0

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")

		switch {
		case r.Method == http.MethodPut && strings.Contains(r.URL.Path, "/asks/"):
			// PUT /asks/<offerID>/ — create the instance
			json.NewEncoder(w).Encode(map[string]any{"success": true})
		case r.Method == http.MethodGet && strings.Contains(r.URL.Path, "/instances"):
			listCallCount++
			// Return a running instance on the first poll
			json.NewEncoder(w).Encode(map[string]any{
				"instances": []Instance{
					{ID: 42, Status: "running", GPUName: "RTX 4090", SSHHost: "1.2.3.4", SSHPort: 22},
				},
			})
		default:
			http.Error(w, "not found", http.StatusNotFound)
		}
	}))
	defer srv.Close()

	client := newTestClient("test-key", srv)

	// Override the sleep by using a very fast test (the mock returns immediately)
	// We can't easily skip the time.Sleep(3*time.Second) in CreateInstance, so we
	// use t.Parallel() and accept the real sleep in a sub-test only when needed.
	// For the happy path we test via a subtly different approach: use a context
	// and let it time out before the sleep if needed. Instead, we just run it — the
	// sleep is only 3s per iteration and we expect success on iteration 1.
	// To avoid 3s per test, we note that CreateInstance sleeps 3s then polls.
	// We skip actual timing here since the mock always returns immediately.

	inst, err := client.CreateInstance(context.Background(), 999, CreateOpts{
		DiskGB: 50,
		Image:  "ubuntu:22.04",
	})
	if err != nil {
		t.Fatalf("CreateInstance() error: %v", err)
	}
	if inst == nil {
		t.Fatal("CreateInstance() returned nil instance")
	}
	if inst.ID != 42 {
		t.Errorf("inst.ID = %d, want 42", inst.ID)
	}
	if inst.Status != "running" {
		t.Errorf("inst.Status = %q, want \"running\"", inst.Status)
	}
}

// TestCreateInstanceDefaultImage verifies the default image is used when opts.Image is empty.
func TestCreateInstanceDefaultImage(t *testing.T) {
	var gotBody map[string]any

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		switch r.Method {
		case http.MethodPut:
			json.NewDecoder(r.Body).Decode(&gotBody)
			json.NewEncoder(w).Encode(map[string]any{"success": true})
		case http.MethodGet:
			json.NewEncoder(w).Encode(map[string]any{
				"instances": []Instance{
					{ID: 1, Status: "running"},
				},
			})
		}
	}))
	defer srv.Close()

	client := newTestClient("key", srv)
	_, _ = client.CreateInstance(context.Background(), 123, CreateOpts{})

	if gotBody["image"] != "nvidia/cuda:12.4.1-devel-ubuntu22.04" {
		t.Errorf("default image = %v, want \"nvidia/cuda:12.4.1-devel-ubuntu22.04\"", gotBody["image"])
	}
	if gotBody["disk"] != 50.0 {
		t.Errorf("default disk = %v, want 50", gotBody["disk"])
	}
}

// TestCreateInstanceDefaultDisk verifies the default disk (50 GB) is used when opts.DiskGB is 0.
func TestCreateInstanceDefaultDisk(t *testing.T) {
	var gotBody map[string]any

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		switch r.Method {
		case http.MethodPut:
			json.NewDecoder(r.Body).Decode(&gotBody)
			json.NewEncoder(w).Encode(map[string]any{"success": true})
		case http.MethodGet:
			json.NewEncoder(w).Encode(map[string]any{
				"instances": []Instance{{ID: 1, Status: "running"}},
			})
		}
	}))
	defer srv.Close()

	client := newTestClient("key", srv)
	_, _ = client.CreateInstance(context.Background(), 456, CreateOpts{DiskGB: 0})

	if gotBody["disk"] != 50.0 {
		t.Errorf("disk = %v, want 50 (default)", gotBody["disk"])
	}
}

// TestCreateInstanceCustomDisk verifies a custom disk size is used.
func TestCreateInstanceCustomDisk(t *testing.T) {
	var gotBody map[string]any

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		switch r.Method {
		case http.MethodPut:
			json.NewDecoder(r.Body).Decode(&gotBody)
			json.NewEncoder(w).Encode(map[string]any{"success": true})
		case http.MethodGet:
			json.NewEncoder(w).Encode(map[string]any{
				"instances": []Instance{{ID: 1, Status: "running"}},
			})
		}
	}))
	defer srv.Close()

	client := newTestClient("key", srv)
	_, _ = client.CreateInstance(context.Background(), 456, CreateOpts{DiskGB: 200})

	if gotBody["disk"] != 200.0 {
		t.Errorf("disk = %v, want 200", gotBody["disk"])
	}
}

// TestCreateInstanceOnStart verifies onstart is included when opts.OnStart is set.
func TestCreateInstanceOnStart(t *testing.T) {
	var gotBody map[string]any

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		switch r.Method {
		case http.MethodPut:
			json.NewDecoder(r.Body).Decode(&gotBody)
			json.NewEncoder(w).Encode(map[string]any{"success": true})
		case http.MethodGet:
			json.NewEncoder(w).Encode(map[string]any{
				"instances": []Instance{{ID: 1, Status: "running"}},
			})
		}
	}))
	defer srv.Close()

	client := newTestClient("key", srv)
	_, _ = client.CreateInstance(context.Background(), 789, CreateOpts{
		OnStart: "echo hello",
	})

	if gotBody["onstart"] != "echo hello" {
		t.Errorf("onstart = %v, want \"echo hello\"", gotBody["onstart"])
	}
}

// TestCreateInstanceNoOnStart verifies onstart is NOT included when opts.OnStart is empty.
func TestCreateInstanceNoOnStart(t *testing.T) {
	var gotBody map[string]any

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		switch r.Method {
		case http.MethodPut:
			json.NewDecoder(r.Body).Decode(&gotBody)
			json.NewEncoder(w).Encode(map[string]any{"success": true})
		case http.MethodGet:
			json.NewEncoder(w).Encode(map[string]any{
				"instances": []Instance{{ID: 1, Status: "running"}},
			})
		}
	}))
	defer srv.Close()

	client := newTestClient("key", srv)
	_, _ = client.CreateInstance(context.Background(), 789, CreateOpts{})

	if _, ok := gotBody["onstart"]; ok {
		t.Error("onstart should not be in body when opts.OnStart is empty")
	}
}

// TestCreateInstancePUTError verifies CreateInstance returns an error when the PUT fails.
func TestCreateInstancePUTError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "forbidden", http.StatusForbidden)
	}))
	defer srv.Close()

	client := newTestClient("key", srv)
	_, err := client.CreateInstance(context.Background(), 999, CreateOpts{})
	if err == nil {
		t.Error("expected error when PUT returns 403")
	}
}

// TestCreateInstancePUTPathContainsOfferID verifies the correct endpoint path is used.
func TestCreateInstancePUTPathContainsOfferID(t *testing.T) {
	var gotPath string

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		switch r.Method {
		case http.MethodPut:
			gotPath = r.URL.Path
			json.NewEncoder(w).Encode(map[string]any{"success": true})
		case http.MethodGet:
			json.NewEncoder(w).Encode(map[string]any{
				"instances": []Instance{{ID: 1, Status: "running"}},
			})
		}
	}))
	defer srv.Close()

	client := newTestClient("key", srv)
	_, _ = client.CreateInstance(context.Background(), 12345, CreateOpts{})

	if !strings.Contains(gotPath, "12345") {
		t.Errorf("PUT path %q should contain offer ID 12345", gotPath)
	}
	if !strings.Contains(gotPath, "/asks/") {
		t.Errorf("PUT path %q should be under /asks/", gotPath)
	}
}

// TestCreateInstanceLoadingStatus verifies CreateInstance returns an instance with "loading" status.
func TestCreateInstanceLoadingStatus(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		switch r.Method {
		case http.MethodPut:
			json.NewEncoder(w).Encode(map[string]any{"success": true})
		case http.MethodGet:
			json.NewEncoder(w).Encode(map[string]any{
				"instances": []Instance{{ID: 7, Status: "loading", GPUName: "A100"}},
			})
		}
	}))
	defer srv.Close()

	client := newTestClient("key", srv)
	inst, err := client.CreateInstance(context.Background(), 42, CreateOpts{})
	if err != nil {
		t.Fatalf("CreateInstance() error: %v", err)
	}
	if inst.Status != "loading" {
		t.Errorf("inst.Status = %q, want \"loading\"", inst.Status)
	}
}

// TestCreateInstanceRequestBodyFields verifies required fields are present in the request body.
func TestCreateInstanceRequestBodyFields(t *testing.T) {
	var gotBody map[string]any

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		switch r.Method {
		case http.MethodPut:
			json.NewDecoder(r.Body).Decode(&gotBody)
			json.NewEncoder(w).Encode(map[string]any{"success": true})
		case http.MethodGet:
			json.NewEncoder(w).Encode(map[string]any{
				"instances": []Instance{{ID: 1, Status: "running"}},
			})
		}
	}))
	defer srv.Close()

	client := newTestClient("key", srv)
	_, _ = client.CreateInstance(context.Background(), 1, CreateOpts{})

	if gotBody["client_id"] != "me" {
		t.Errorf("client_id = %v, want \"me\"", gotBody["client_id"])
	}
	if !strings.Contains(gotBody["runtype"].(string), "ssh") {
		t.Errorf("runtype = %v, want to contain \"ssh\"", gotBody["runtype"])
	}
}
