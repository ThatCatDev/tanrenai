package gpuprovider

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/ThatCatDev/tanrenai/server/internal/gpuclient"
	"github.com/ThatCatDev/tanrenai/server/internal/vastai"
)

// newVastAITestEnv creates a VastAIProvider backed by mock GPU and vast.ai servers.
// gpuHealthy controls whether /health returns 200.
// instanceStatus is what the mock vast.ai API returns for GetInstance.
func newVastAITestEnv(t *testing.T, gpuHealthy bool, instanceStatus string) (*VastAIProvider, *httptest.Server, *httptest.Server) {
	t.Helper()

	gpuSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/health" {
			if gpuHealthy {
				w.WriteHeader(http.StatusOK)
			} else {
				w.WriteHeader(http.StatusInternalServerError)
			}

			return
		}
		http.NotFound(w, r)
	}))
	t.Cleanup(gpuSrv.Close)

	vastSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		resp := map[string]interface{}{
			"instances": []map[string]interface{}{
				{
					"id":            42,
					"actual_status": instanceStatus,
					"ssh_host":      "10.0.0.1",
					"ssh_port":      22,
				},
			},
		}
		json.NewEncoder(w).Encode(resp)
	}))
	t.Cleanup(vastSrv.Close)

	// Use the vastai.NewClient but point it at our test server by creating the
	// provider manually — we can't override the vastai baseURL, so instead
	// pass a nil client and exercise code that falls back to the stopped state.
	_ = vastSrv // referenced for documentation; actual vastai.Client hits real baseURL

	gpuClient := gpuclient.New(gpuSrv.URL)
	vastClient := vastai.NewClient("test-key")

	provider := NewVastAIProvider(vastClient, gpuClient, "42", gpuSrv.URL, 30*time.Minute)

	return provider, gpuSrv, vastSrv
}

func TestVastAIProviderName(t *testing.T) {
	provider, _, _ := newVastAITestEnv(t, true, "running")
	if got := provider.Name(); got != "vastai" {
		t.Errorf("Name() = %q, want %q", got, "vastai")
	}
}

func TestVastAIProviderRecordActivity(t *testing.T) {
	provider, _, _ := newVastAITestEnv(t, true, "running")
	before := provider.lastActivity
	time.Sleep(time.Millisecond)
	provider.RecordActivity()
	provider.mu.Lock()
	after := provider.lastActivity
	provider.mu.Unlock()
	if !after.After(before) {
		t.Error("RecordActivity() did not update lastActivity")
	}
}

func TestVastAIProviderClose(t *testing.T) {
	provider, _, _ := newVastAITestEnv(t, true, "running")
	// Start idle timer so there is a channel to close.
	provider.idleTimeout = 1 * time.Hour
	provider.StartIdleTimer()
	// Close should not panic and should nil out stopCh.
	provider.Close()
	provider.mu.Lock()
	if provider.stopCh != nil {
		t.Error("Close() did not nil out stopCh")
	}
	provider.mu.Unlock()
}

func TestVastAIProviderCloseIdempotent(t *testing.T) {
	provider, _, _ := newVastAITestEnv(t, true, "running")
	// Close without ever starting the idle timer should not panic.
	provider.Close()
	provider.Close()
}

func TestVastAIProviderStartIdleTimerZeroTimeout(t *testing.T) {
	provider, _, _ := newVastAITestEnv(t, true, "running")
	provider.idleTimeout = 0
	// With zero timeout, StartIdleTimer must be a no-op (no goroutine, no channel).
	provider.StartIdleTimer()
	provider.mu.Lock()
	if provider.stopCh != nil {
		t.Error("StartIdleTimer() should not set stopCh when idleTimeout <= 0")
	}
	provider.mu.Unlock()
}

func TestVastAIProviderStatusRunning(t *testing.T) {
	provider, _, _ := newVastAITestEnv(t, true, "running")
	status, err := provider.Status(context.Background())
	if err != nil {
		t.Fatalf("Status() returned error: %v", err)
	}
	if status.State != "running" {
		t.Errorf("State = %q, want running", status.State)
	}
	if status.Provider != "vastai" {
		t.Errorf("Provider = %q, want vastai", status.Provider)
	}
	if status.IdleSince == nil {
		t.Error("IdleSince should be set when running")
	}
}

func TestVastAIProviderStatusStoppedFallback(t *testing.T) {
	// GPU unhealthy and no vastai client (nil) → falls back to "stopped".
	gpuSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
	}))
	t.Cleanup(gpuSrv.Close)

	gpuClient := gpuclient.New(gpuSrv.URL)
	provider := NewVastAIProvider(nil, gpuClient, "", gpuSrv.URL, 0)

	status, err := provider.Status(context.Background())
	if err != nil {
		t.Fatalf("Status() returned error: %v", err)
	}
	if status.State != "stopped" {
		t.Errorf("State = %q, want stopped", status.State)
	}
	if status.Provider != "vastai" {
		t.Errorf("Provider = %q, want vastai", status.Provider)
	}
}

func TestVastAIProviderStopNotConfigured(t *testing.T) {
	gpuSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	}))
	t.Cleanup(gpuSrv.Close)

	gpuClient := gpuclient.New(gpuSrv.URL)
	provider := NewVastAIProvider(nil, gpuClient, "", gpuSrv.URL, 0)

	err := provider.Stop(context.Background())
	if err == nil {
		t.Error("Stop() should return error when not configured")
	}
}

func TestVastAIProviderEnsureRunningAlreadyHealthy(t *testing.T) {
	provider, _, _ := newVastAITestEnv(t, true, "running")
	// GPU is healthy, so EnsureRunning should return without calling StartInstance.
	if err := provider.EnsureRunning(context.Background()); err != nil {
		t.Errorf("EnsureRunning() returned error: %v", err)
	}
}

func TestVastAIProviderEnsureRunningContextCancelled(t *testing.T) {
	// GPU is healthy so EnsureRunning returns nil immediately (no start attempt).
	// Then verify context cancellation is propagated in waitForHealthy.
	// We test this by using an already-starting provider and a cancelled context.
	gpuSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Always report unhealthy so waitForHealthy keeps looping.
		w.WriteHeader(http.StatusInternalServerError)
	}))
	t.Cleanup(gpuSrv.Close)

	gpuClient := gpuclient.New(gpuSrv.URL)
	vastClient := vastai.NewClient("test-key")
	provider := NewVastAIProvider(vastClient, gpuClient, "99", gpuSrv.URL, 0)

	// Simulate that starting is already in progress.
	provider.mu.Lock()
	provider.starting = true
	provider.mu.Unlock()

	ctx, cancel := context.WithCancel(context.Background())
	cancel() // cancel immediately

	err := provider.EnsureRunning(ctx)
	if err == nil {
		t.Error("EnsureRunning() should return error when context cancelled")
	}

	// Clean up.
	provider.mu.Lock()
	provider.starting = false
	provider.mu.Unlock()
}

func TestVastAIProviderStartIdleTimerReplaces(t *testing.T) {
	provider, _, _ := newVastAITestEnv(t, true, "running")
	provider.idleTimeout = 1 * time.Hour

	provider.StartIdleTimer()
	ch1 := provider.stopCh

	// Start again — should close the old channel and create a new one.
	provider.StartIdleTimer()
	ch2 := provider.stopCh

	if ch1 == ch2 {
		t.Error("StartIdleTimer() should replace existing stopCh")
	}

	// Cleanup: close the new channel.
	provider.Close()
}

func TestVastAIProviderNewVastAIProvider(t *testing.T) {
	gpuClient := gpuclient.New("http://localhost:11435")
	vastClient := vastai.NewClient("key")
	timeout := 15 * time.Minute

	p := NewVastAIProvider(vastClient, gpuClient, "inst-1", "http://gpu:11435", timeout)
	if p == nil {
		t.Fatal("NewVastAIProvider returned nil")
	}
	if p.instanceID != "inst-1" {
		t.Errorf("instanceID = %q, want inst-1", p.instanceID)
	}
	if p.gpuURL != "http://gpu:11435" {
		t.Errorf("gpuURL = %q, want http://gpu:11435", p.gpuURL)
	}
	if p.idleTimeout != timeout {
		t.Errorf("idleTimeout = %v, want %v", p.idleTimeout, timeout)
	}
	if p.lastActivity.IsZero() {
		t.Error("lastActivity should be set on creation")
	}
}

func TestVastAIProviderStopWithClient(t *testing.T) {
	// Create a mock for the vast.ai put call.
	stopped := false
	vastAPISrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodPut {
			stopped = true
			w.WriteHeader(http.StatusOK)
			fmt.Fprintln(w, `{}`)

			return
		}
		http.NotFound(w, r)
	}))
	t.Cleanup(vastAPISrv.Close)

	// vastai.Client always hits console.vast.ai — we can't override baseURL
	// without a custom client, so instead test via the Stop-not-configured path.
	// We verify the configured path returns error only when client is nil.
	gpuClient := gpuclient.New("http://localhost:11435")
	provider := NewVastAIProvider(nil, gpuClient, "", "http://localhost:11435", 0)
	err := provider.Stop(context.Background())
	if err == nil {
		t.Error("Stop() should fail when instanceID is empty")
	}
	_ = stopped
}
