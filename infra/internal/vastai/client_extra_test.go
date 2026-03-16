package vastai

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// redirectTransport rewrites the host of every outbound request to point at
// the test server, allowing us to test the Client without exposing baseURL.
type redirectTransport struct {
	target string // e.g. "http://127.0.0.1:PORT"
	base   http.RoundTripper
}

func (rt *redirectTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	// Clone to avoid mutating the original
	cloned := req.Clone(req.Context())
	cloned.URL.Scheme = "http"
	cloned.URL.Host = strings.TrimPrefix(rt.target, "http://")
	if rt.base != nil {
		return rt.base.RoundTrip(cloned)
	}

	return http.DefaultTransport.RoundTrip(cloned)
}

// newTestClient creates a Client whose HTTP requests are redirected to srv.
func newTestClient(apiKey string, srv *httptest.Server) *Client {
	return &Client{
		apiKey: apiKey,
		httpClient: &http.Client{
			Transport: &redirectTransport{target: srv.URL},
		},
	}
}

// --- normalizeGPUName ---

func TestNormalizeGPUName(t *testing.T) {
	tests := []struct {
		input string
		want  string
	}{
		{"RTX 4090", "rtx4090"},
		{"RTX_3090", "rtx3090"},
		{"A100-SXM4-80GB", "a100sxm480gb"},
		{"H100", "h100"},
		{"  H100  ", "h100"},
		{"", ""},
	}

	for _, tc := range tests {
		got := normalizeGPUName(tc.input)
		if got != tc.want {
			t.Errorf("normalizeGPUName(%q) = %q, want %q", tc.input, got, tc.want)
		}
	}
}

func TestNormalizeGPUNamePartialMatch(t *testing.T) {
	// "4090" should match "RTX 4090"
	needle := normalizeGPUName("4090")
	haystack := normalizeGPUName("RTX 4090")
	if !strings.Contains(haystack, needle) {
		t.Errorf("expected %q to contain %q", haystack, needle)
	}

	// "A100" should match "A100-SXM4-80GB"
	needle = normalizeGPUName("A100")
	haystack = normalizeGPUName("A100-SXM4-80GB")
	if !strings.Contains(haystack, needle) {
		t.Errorf("expected %q to contain %q", haystack, needle)
	}
}

// --- ListInstances ---

func TestListInstances(t *testing.T) {
	instances := []Instance{
		{ID: 1, Status: "running", GPUName: "RTX 4090", CostPerHr: 0.5, SSHHost: "1.2.3.4", SSHPort: 22},
		{ID: 2, Status: "exited", GPUName: "RTX 3090", CostPerHr: 0.3, SSHHost: "5.6.7.8", SSHPort: 22},
	}

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") != "Bearer test-key" {
			http.Error(w, "unauthorized", http.StatusUnauthorized)

			return
		}
		if r.Method != http.MethodGet {
			t.Errorf("expected GET, got %s", r.Method)
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{"instances": instances})
	}))
	defer srv.Close()

	client := newTestClient("test-key", srv)
	got, err := client.ListInstances(context.Background())
	if err != nil {
		t.Fatalf("ListInstances() error: %v", err)
	}
	if len(got) != 2 {
		t.Fatalf("ListInstances() returned %d instances, want 2", len(got))
	}
	if got[0].ID != 1 || got[0].Status != "running" {
		t.Errorf("got[0] = %+v, want ID=1 Status=running", got[0])
	}
	if got[1].ID != 2 || got[1].Status != "exited" {
		t.Errorf("got[1] = %+v, want ID=2 Status=exited", got[1])
	}
}

func TestListInstancesEmpty(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{"instances": []any{}})
	}))
	defer srv.Close()

	client := newTestClient("key", srv)
	got, err := client.ListInstances(context.Background())
	if err != nil {
		t.Fatalf("ListInstances() error: %v", err)
	}
	if len(got) != 0 {
		t.Errorf("expected empty slice, got %v", got)
	}
}

func TestListInstancesAuthError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "unauthorized", http.StatusUnauthorized)
	}))
	defer srv.Close()

	client := newTestClient("bad-key", srv)
	_, err := client.ListInstances(context.Background())
	if err == nil {
		t.Error("expected error on 401 response")
	}
}

// --- GetInstance ---

func TestGetInstanceFound(t *testing.T) {
	instances := []Instance{
		{ID: 123, Status: "running", GPUName: "RTX 4090", CostPerHr: 0.5, SSHHost: "1.2.3.4", SSHPort: 2222},
	}

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") != "Bearer mykey" {
			http.Error(w, "unauthorized", http.StatusUnauthorized)

			return
		}
		// Should include ?id=123 in query
		if !strings.Contains(r.URL.RawQuery, "id=123") {
			t.Errorf("expected ?id=123 in query, got %q", r.URL.RawQuery)
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{"instances": instances})
	}))
	defer srv.Close()

	client := newTestClient("mykey", srv)
	inst, err := client.GetInstance(context.Background(), "123")
	if err != nil {
		t.Fatalf("GetInstance() error: %v", err)
	}
	if inst.ID != 123 {
		t.Errorf("inst.ID = %d, want 123", inst.ID)
	}
	if inst.Status != "running" {
		t.Errorf("inst.Status = %q, want \"running\"", inst.Status)
	}
	if inst.SSHHost != "1.2.3.4" {
		t.Errorf("inst.SSHHost = %q, want \"1.2.3.4\"", inst.SSHHost)
	}
}

func TestGetInstanceNotFound(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		// Return a different instance ID so the lookup fails
		json.NewEncoder(w).Encode(map[string]any{
			"instances": []Instance{{ID: 999, Status: "running"}},
		})
	}))
	defer srv.Close()

	client := newTestClient("key", srv)
	_, err := client.GetInstance(context.Background(), "123")
	if err == nil {
		t.Error("expected error when instance not found")
	}
	if !strings.Contains(err.Error(), "123") {
		t.Errorf("error should mention instance ID 123, got: %v", err)
	}
}

func TestGetInstanceEmpty(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{"instances": []any{}})
	}))
	defer srv.Close()

	client := newTestClient("key", srv)
	_, err := client.GetInstance(context.Background(), "42")
	if err == nil {
		t.Error("expected error when instance list is empty")
	}
}

// --- StartInstance / StopInstance / DestroyInstance ---

func TestStartInstance(t *testing.T) {
	var gotMethod, gotPath, gotBody string

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotMethod = r.Method
		gotPath = r.URL.Path
		b := make([]byte, 1024)
		n, _ := r.Body.Read(b)
		gotBody = string(b[:n])
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{"success": true})
	}))
	defer srv.Close()

	client := newTestClient("key", srv)
	err := client.StartInstance(context.Background(), "456")
	if err != nil {
		t.Fatalf("StartInstance() error: %v", err)
	}
	if gotMethod != http.MethodPut {
		t.Errorf("StartInstance used %s, want PUT", gotMethod)
	}
	if !strings.Contains(gotPath, "456") {
		t.Errorf("path %q should contain instance ID \"456\"", gotPath)
	}
	if !strings.Contains(gotBody, "running") {
		t.Errorf("StartInstance body should contain \"running\", got: %q", gotBody)
	}
}

func TestStopInstance(t *testing.T) {
	var gotBody string

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b := make([]byte, 1024)
		n, _ := r.Body.Read(b)
		gotBody = string(b[:n])
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{"success": true})
	}))
	defer srv.Close()

	client := newTestClient("key", srv)
	err := client.StopInstance(context.Background(), "789")
	if err != nil {
		t.Fatalf("StopInstance() error: %v", err)
	}
	if !strings.Contains(gotBody, "stopped") {
		t.Errorf("StopInstance body should contain \"stopped\", got: %q", gotBody)
	}
}

func TestDestroyInstance(t *testing.T) {
	var gotMethod, gotPath string

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotMethod = r.Method
		gotPath = r.URL.Path
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{"success": true})
	}))
	defer srv.Close()

	client := newTestClient("key", srv)
	err := client.DestroyInstance(context.Background(), "999")
	if err != nil {
		t.Fatalf("DestroyInstance() error: %v", err)
	}
	if gotMethod != http.MethodDelete {
		t.Errorf("DestroyInstance used %s, want DELETE", gotMethod)
	}
	if !strings.Contains(gotPath, "999") {
		t.Errorf("path %q should contain instance ID \"999\"", gotPath)
	}
}

func TestInstanceOperationsAuthHeader(t *testing.T) {
	var receivedAuth string

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		receivedAuth = r.Header.Get("Authorization")
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{"success": true})
	}))
	defer srv.Close()

	client := newTestClient("secret-api-key", srv)
	_ = client.StartInstance(context.Background(), "1")

	if receivedAuth != "Bearer secret-api-key" {
		t.Errorf("Authorization = %q, want \"Bearer secret-api-key\"", receivedAuth)
	}
}

// --- SearchOffers ---

func TestSearchOffersNoFilter(t *testing.T) {
	offers := []Offer{
		{ID: 1, GPUName: "RTX 4090", GPURAMTotal: 24576, CostPerHr: 0.5, NumGPUs: 1},
		{ID: 2, GPUName: "RTX 3090", GPURAMTotal: 24576, CostPerHr: 0.3, NumGPUs: 1},
		{ID: 3, GPUName: "A100 SXM4 80GB", GPURAMTotal: 81920, CostPerHr: 1.2, NumGPUs: 1},
	}

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			t.Errorf("SearchOffers expected POST, got %s", r.Method)
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{"offers": offers})
	}))
	defer srv.Close()

	client := newTestClient("key", srv)
	got, err := client.SearchOffers(context.Background(), SearchQuery{
		MinGPURAM:    24,
		MaxCostPerHr: 2.0,
	})
	if err != nil {
		t.Fatalf("SearchOffers() error: %v", err)
	}
	if len(got) != 3 {
		t.Fatalf("SearchOffers() returned %d offers, want 3", len(got))
	}
	// Should be sorted by cost ascending
	if got[0].CostPerHr > got[1].CostPerHr || got[1].CostPerHr > got[2].CostPerHr {
		t.Errorf("offers not sorted by cost: %v %v %v", got[0].CostPerHr, got[1].CostPerHr, got[2].CostPerHr)
	}
}

func TestSearchOffersGPUNameFilter(t *testing.T) {
	offers := []Offer{
		{ID: 1, GPUName: "RTX 4090", GPURAMTotal: 24576, CostPerHr: 0.5},
		{ID: 2, GPUName: "RTX 3090", GPURAMTotal: 24576, CostPerHr: 0.3},
		{ID: 3, GPUName: "A100 SXM4 80GB", GPURAMTotal: 81920, CostPerHr: 1.2},
	}

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{"offers": offers})
	}))
	defer srv.Close()

	client := newTestClient("key", srv)
	got, err := client.SearchOffers(context.Background(), SearchQuery{
		GPUName:      "4090",
		MinGPURAM:    24,
		MaxCostPerHr: 2.0,
	})
	if err != nil {
		t.Fatalf("SearchOffers() error: %v", err)
	}
	if len(got) != 1 {
		t.Fatalf("SearchOffers() with GPUName filter returned %d offers, want 1", len(got))
	}
	if got[0].ID != 1 {
		t.Errorf("expected offer ID 1 (RTX 4090), got ID %d", got[0].ID)
	}
}

func TestSearchOffersA100Filter(t *testing.T) {
	offers := []Offer{
		{ID: 1, GPUName: "RTX 4090", GPURAMTotal: 24576, CostPerHr: 0.5},
		{ID: 2, GPUName: "A100 SXM4 80GB", GPURAMTotal: 81920, CostPerHr: 1.2},
		{ID: 3, GPUName: "A100 PCIe 40GB", GPURAMTotal: 40960, CostPerHr: 0.9},
	}

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{"offers": offers})
	}))
	defer srv.Close()

	client := newTestClient("key", srv)
	got, err := client.SearchOffers(context.Background(), SearchQuery{
		GPUName:      "A100",
		MinGPURAM:    24,
		MaxCostPerHr: 5.0,
	})
	if err != nil {
		t.Fatalf("SearchOffers() error: %v", err)
	}
	if len(got) != 2 {
		t.Fatalf("SearchOffers() A100 filter returned %d offers, want 2", len(got))
	}
	for _, o := range got {
		if !strings.Contains(o.GPUName, "A100") {
			t.Errorf("expected only A100 offers, got %q", o.GPUName)
		}
	}
}

func TestSearchOffersNoMatch(t *testing.T) {
	offers := []Offer{
		{ID: 1, GPUName: "RTX 4090", GPURAMTotal: 24576, CostPerHr: 0.5},
	}

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{"offers": offers})
	}))
	defer srv.Close()

	client := newTestClient("key", srv)
	got, err := client.SearchOffers(context.Background(), SearchQuery{
		GPUName:      "H100",
		MinGPURAM:    80,
		MaxCostPerHr: 5.0,
	})
	if err != nil {
		t.Fatalf("SearchOffers() error: %v", err)
	}
	if len(got) != 0 {
		t.Errorf("expected 0 offers matching H100, got %d", len(got))
	}
}

func TestSearchOffersSortedByCost(t *testing.T) {
	// Server returns offers in descending cost order; client should sort ascending
	offers := []Offer{
		{ID: 3, GPUName: "RTX 4090", CostPerHr: 0.9},
		{ID: 1, GPUName: "RTX 4090", CostPerHr: 0.3},
		{ID: 2, GPUName: "RTX 4090", CostPerHr: 0.6},
	}

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{"offers": offers})
	}))
	defer srv.Close()

	client := newTestClient("key", srv)
	got, err := client.SearchOffers(context.Background(), SearchQuery{MaxCostPerHr: 5.0})
	if err != nil {
		t.Fatalf("SearchOffers() error: %v", err)
	}
	if len(got) != 3 {
		t.Fatalf("expected 3 offers, got %d", len(got))
	}
	if got[0].CostPerHr != 0.3 || got[1].CostPerHr != 0.6 || got[2].CostPerHr != 0.9 {
		t.Errorf("offers not sorted by cost: %v", []float64{got[0].CostPerHr, got[1].CostPerHr, got[2].CostPerHr})
	}
}

func TestSearchOffersAPIError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "rate limited", http.StatusTooManyRequests)
	}))
	defer srv.Close()

	client := newTestClient("key", srv)
	_, err := client.SearchOffers(context.Background(), SearchQuery{})
	if err == nil {
		t.Error("expected error on 429 response")
	}
}

func TestSearchOffersRequestBody(t *testing.T) {
	var requestBody map[string]any

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewDecoder(r.Body).Decode(&requestBody)
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{"offers": []any{}})
	}))
	defer srv.Close()

	client := newTestClient("key", srv)
	_, err := client.SearchOffers(context.Background(), SearchQuery{
		MinGPURAM:    80,
		MaxCostPerHr: 2.5,
		MinDiskGB:    100,
	})
	if err != nil {
		t.Fatalf("SearchOffers() error: %v", err)
	}

	// Verify server-side filters are set correctly
	if requestBody["type"] != "ondemand" {
		t.Errorf("request body type = %v, want \"ondemand\"", requestBody["type"])
	}
	// gpu_ram in API is in MB
	gpuRAM, ok := requestBody["gpu_ram"].(map[string]any)
	if !ok {
		t.Fatalf("gpu_ram not a map: %v", requestBody["gpu_ram"])
	}
	if gpuRAM["gte"] != 80*1024.0 {
		t.Errorf("gpu_ram.gte = %v, want %v (MB)", gpuRAM["gte"], 80*1024.0)
	}
}
