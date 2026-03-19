package gpuprovider

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"reflect"
	"testing"
	"time"
	"unsafe"

	"github.com/ThatCatDev/tanrenai/server/internal/gpuclient"
	"github.com/ThatCatDev/tanrenai/server/internal/vastai"
)

// vastaiRedirectTransport rewrites requests destined for the real vast.ai console
// to a local test server so tests never hit the network.
type vastaiRedirectTransport struct {
	targetBaseURL string
}

func (t *vastaiRedirectTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	// Clone the request so we don't mutate the original.
	clone := req.Clone(req.Context())
	clone.URL.Scheme = "http"
	clone.URL.Host = t.targetBaseURL[len("http://"):]

	return http.DefaultTransport.RoundTrip(clone)
}

// setVastaiHTTPClient injects a custom *http.Client into the unexported field of
// a *vastai.Client using reflect + unsafe. This is only safe in tests.
func setVastaiHTTPClient(c *vastai.Client, hc *http.Client) {
	v := reflect.ValueOf(c).Elem()
	f := v.FieldByName("httpClient")
	ptr := (**http.Client)(unsafe.Pointer(f.UnsafeAddr()))
	*ptr = hc
}

// newVastAITestEnvWithMock builds a full test environment where both the GPU
// server and the vast.ai API are backed by controllable httptest.Servers.
func newVastAITestEnvWithMock(
	t *testing.T,
	gpuHealthy bool,
	vastHandler http.HandlerFunc,
) (*VastAIProvider, *httptest.Server, *httptest.Server) {
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

	vastSrv := httptest.NewServer(vastHandler)
	t.Cleanup(vastSrv.Close)

	vastClient := vastai.NewClient("test-key")
	setVastaiHTTPClient(vastClient, &http.Client{
		Transport: &vastaiRedirectTransport{targetBaseURL: vastSrv.URL},
	})

	gpuClient := gpuclient.New(gpuSrv.URL)
	provider := NewVastAIProvider(vastClient, gpuClient, "42", gpuSrv.URL, 30*time.Minute)

	return provider, gpuSrv, vastSrv
}

// vastInstanceResponse returns a mock vast.ai instances JSON response.
func vastInstanceResponse(id int, status string) []byte {
	resp := map[string]interface{}{
		"instances": []map[string]interface{}{
			{
				"id":            id,
				"actual_status": status,
				"ssh_host":      "10.0.0.1",
				"ssh_port":      22,
			},
		},
	}
	b, _ := json.Marshal(resp)

	return b
}

// ---------------------------------------------------------------------------
// EnsureRunning — unhealthy GPU, StartInstance succeeds, then GPU becomes healthy
// ---------------------------------------------------------------------------

func TestVastAIProviderEnsureRunningStartsInstance(t *testing.T) {
	var startCalled bool

	// GPU starts unhealthy; after StartInstance is called we flip it to healthy
	// by using a counter in the handler.
	callCount := 0
	gpuSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/health" {
			callCount++
			// First call (initial health check) → unhealthy.
			// Second call (waitForHealthy ticker) → healthy.
			if callCount >= 2 {
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
		if r.Method == http.MethodPut {
			startCalled = true
			w.WriteHeader(http.StatusOK)
			fmt.Fprintln(w, `{}`)

			return
		}
		http.NotFound(w, r)
	}))
	t.Cleanup(vastSrv.Close)

	vastClient := vastai.NewClient("test-key")
	setVastaiHTTPClient(vastClient, &http.Client{
		Transport: &vastaiRedirectTransport{targetBaseURL: vastSrv.URL},
	})
	gpuClient := gpuclient.New(gpuSrv.URL)

	// Use a short waitForHealthy interval by overriding the provider's ticker via
	// a very short timeout; we rely on the GPU becoming healthy on the 2nd call.
	provider := NewVastAIProvider(vastClient, gpuClient, "42", gpuSrv.URL, 0)

	// waitForHealthy uses a 5-second ticker; that's too slow for a unit test.
	// Instead we test the path directly: mark starting=false (default), let
	// EnsureRunning call StartInstance, then confirm it proceeds to waitForHealthy.
	// We can't speed up the ticker from outside, so we exercise this by cancelling
	// the context quickly after StartInstance is called — we just need the code path hit.
	ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
	defer cancel()

	err := provider.EnsureRunning(ctx)
	// Either the GPU became healthy (no error) or the context timed out — both are
	// acceptable; what matters is that StartInstance was invoked.
	if !startCalled {
		t.Error("EnsureRunning did not call StartInstance when GPU was unhealthy")
	}
	// err could be nil or context.DeadlineExceeded — both indicate the code path ran.
	_ = err
}

// ---------------------------------------------------------------------------
// EnsureRunning — StartInstance returns an error
// ---------------------------------------------------------------------------

func TestVastAIProviderEnsureRunningStartInstanceError(t *testing.T) {
	gpuSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Always unhealthy so EnsureRunning tries to start the instance.
		w.WriteHeader(http.StatusInternalServerError)
	}))
	t.Cleanup(gpuSrv.Close)

	vastSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Simulate vast.ai returning an error for StartInstance.
		w.WriteHeader(http.StatusInternalServerError)
		fmt.Fprintln(w, `{"error": "quota exceeded"}`)
	}))
	t.Cleanup(vastSrv.Close)

	vastClient := vastai.NewClient("test-key")
	setVastaiHTTPClient(vastClient, &http.Client{
		Transport: &vastaiRedirectTransport{targetBaseURL: vastSrv.URL},
	})
	gpuClient := gpuclient.New(gpuSrv.URL)
	provider := NewVastAIProvider(vastClient, gpuClient, "42", gpuSrv.URL, 0)

	err := provider.EnsureRunning(context.Background())
	if err == nil {
		t.Error("EnsureRunning should return error when StartInstance fails")
	}
}

// ---------------------------------------------------------------------------
// EnsureRunning — concurrent start: second goroutine sees starting=true after lock re-check
// ---------------------------------------------------------------------------

func TestVastAIProviderEnsureRunningConcurrentStartingRecheck(t *testing.T) {
	gpuSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Always unhealthy so waitForHealthy keeps polling.
		w.WriteHeader(http.StatusInternalServerError)
	}))
	t.Cleanup(gpuSrv.Close)

	gpuClient := gpuclient.New(gpuSrv.URL)
	vastClient := vastai.NewClient("test-key")
	provider := NewVastAIProvider(vastClient, gpuClient, "42", gpuSrv.URL, 0)

	// Set starting = true AFTER the first health check but BEFORE the lock.
	// We simulate this by manually setting starting to true while p.starting is false,
	// which forces the "re-check after lock" branch (lines 65-68) to be exercised.
	//
	// The trick: set starting=true so EnsureRunning takes the re-check path.
	// Health check passes the first `if p.starting` guard (starting=false), then
	// gpu health fails, then we grab the lock — but at this point starting is true
	// because we pre-set it here.
	provider.mu.Lock()
	provider.starting = true
	provider.mu.Unlock()

	ctx, cancel := context.WithCancel(context.Background())
	cancel() // cancel immediately so waitForHealthy exits fast

	err := provider.EnsureRunning(ctx)
	// Should return context.Canceled from waitForHealthy.
	if !errors.Is(err, context.Canceled) {
		t.Errorf("expected context.Canceled, got: %v", err)
	}

	// Cleanup: reset starting.
	provider.mu.Lock()
	provider.starting = false
	provider.mu.Unlock()
}

// ---------------------------------------------------------------------------
// waitForHealthy — timeout path
// ---------------------------------------------------------------------------

func TestVastAIProviderWaitForHealthyTimeout(t *testing.T) {
	gpuSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
	}))
	t.Cleanup(gpuSrv.Close)

	gpuClient := gpuclient.New(gpuSrv.URL)
	vastClient := vastai.NewClient("test-key")
	provider := NewVastAIProvider(vastClient, gpuClient, "42", gpuSrv.URL, 0)

	// Use a context that is already cancelled to trigger the ctx.Done() path in
	// waitForHealthy immediately.
	ctx, cancel := context.WithTimeout(context.Background(), 1*time.Millisecond)
	defer cancel()
	time.Sleep(5 * time.Millisecond) // ensure the timeout fires

	err := provider.waitForHealthy(ctx)
	if err == nil {
		t.Fatal("waitForHealthy should return error on context timeout")
	}
}

// ---------------------------------------------------------------------------
// Status — GPU unhealthy, GetInstance succeeds via mock vast.ai
// ---------------------------------------------------------------------------

func TestVastAIProviderStatusViaVastAI(t *testing.T) {
	wantState := "loading"
	provider, _, _ := newVastAITestEnvWithMock(t, false,
		http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			w.Write(vastInstanceResponse(42, wantState))
		}),
	)

	status, err := provider.Status(context.Background())
	if err != nil {
		t.Fatalf("Status() returned error: %v", err)
	}
	if status.State != wantState {
		t.Errorf("State = %q, want %q", status.State, wantState)
	}
	if status.Provider != "vastai" {
		t.Errorf("Provider = %q, want vastai", status.Provider)
	}
}

// ---------------------------------------------------------------------------
// Status — GPU unhealthy, GetInstance returns error → fallback to "stopped"
// ---------------------------------------------------------------------------

func TestVastAIProviderStatusGetInstanceError(t *testing.T) {
	provider, _, _ := newVastAITestEnvWithMock(t, false,
		http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.WriteHeader(http.StatusInternalServerError)
			fmt.Fprintln(w, `{"error": "not found"}`)
		}),
	)

	status, err := provider.Status(context.Background())
	if err != nil {
		t.Fatalf("Status() returned error: %v", err)
	}
	if status.State != "stopped" {
		t.Errorf("State = %q, want stopped", status.State)
	}
}

// ---------------------------------------------------------------------------
// Stop — configured client, StopInstance succeeds
// ---------------------------------------------------------------------------

func TestVastAIProviderStopSuccess(t *testing.T) {
	var stopCalled bool
	provider, _, _ := newVastAITestEnvWithMock(t, true,
		http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.Method == http.MethodPut {
				stopCalled = true
				w.WriteHeader(http.StatusOK)
				fmt.Fprintln(w, `{}`)

				return
			}
			http.NotFound(w, r)
		}),
	)

	if err := provider.Stop(context.Background()); err != nil {
		t.Fatalf("Stop() returned error: %v", err)
	}
	if !stopCalled {
		t.Error("StopInstance was not called")
	}
}

// ---------------------------------------------------------------------------
// Stop — StopInstance returns error
// ---------------------------------------------------------------------------

func TestVastAIProviderStopError(t *testing.T) {
	provider, _, _ := newVastAITestEnvWithMock(t, true,
		http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.WriteHeader(http.StatusInternalServerError)
			fmt.Fprintln(w, `{"error": "forbidden"}`)
		}),
	)

	err := provider.Stop(context.Background())
	if err == nil {
		t.Error("Stop() should return error when StopInstance fails")
	}
}

// ---------------------------------------------------------------------------
// StartIdleTimer — verify that the goroutine exits via stopCh when Close is called
// ---------------------------------------------------------------------------

func TestVastAIProviderStartIdleTimerGoroutineExitsOnClose(t *testing.T) {
	gpuSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	}))
	t.Cleanup(gpuSrv.Close)

	vastClient := vastai.NewClient("test-key")
	gpuClient := gpuclient.New(gpuSrv.URL)
	provider := NewVastAIProvider(vastClient, gpuClient, "42", gpuSrv.URL, 1*time.Hour)

	provider.StartIdleTimer()

	// Close should signal the goroutine to exit. No panic is the success condition.
	provider.Close()

	// Confirm the channel is nil after close.
	provider.mu.Lock()
	ch := provider.stopCh
	provider.mu.Unlock()
	if ch != nil {
		t.Error("stopCh should be nil after Close()")
	}
}
