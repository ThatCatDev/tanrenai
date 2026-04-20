package apiclient

import (
	"context"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
)

func TestRefreshTransport_RetriesOn401(t *testing.T) {
	var calls atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		auth := r.Header.Get("Authorization")
		// First call: arrived with "stale". Reject.
		// Second call: should arrive with "fresh" after the transport refreshed.
		if auth == "Bearer fresh" {
			w.WriteHeader(http.StatusOK)
			_, _ = w.Write([]byte(`{"object":"list","data":[]}`))
			return
		}
		w.WriteHeader(http.StatusUnauthorized)
		_, _ = w.Write([]byte(`{"error":"expired"}`))
	}))
	defer srv.Close()

	c := New(srv.URL)
	c.SetAuthToken("stale")
	var refreshed atomic.Int32
	c.SetRefreshFunc(func() (string, error) {
		refreshed.Add(1)
		return "fresh", nil
	})

	resp, err := c.ListModels(context.Background())
	if err != nil {
		t.Fatalf("ListModels: %v", err)
	}
	if resp == nil {
		t.Fatal("nil response")
	}
	if refreshed.Load() != 1 {
		t.Errorf("refresh should have fired exactly once, got %d", refreshed.Load())
	}
	if calls.Load() != 2 {
		t.Errorf("server should have received 2 requests (initial + retry), got %d", calls.Load())
	}
	if tok := c.getAuthToken(); tok != "fresh" {
		t.Errorf("client token should be updated to 'fresh', got %q", tok)
	}
}

func TestRefreshTransport_DoesNotLoopWhenRefreshFails(t *testing.T) {
	var calls atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		w.WriteHeader(http.StatusUnauthorized)
	}))
	defer srv.Close()

	c := New(srv.URL)
	c.SetAuthToken("stale")
	c.SetRefreshFunc(func() (string, error) {
		return "", context.Canceled // any error
	})

	_, err := c.ListModels(context.Background())
	if err == nil {
		t.Fatal("expected 401 to surface when refresh fails")
	}
	// Initial + one fallback replay (transport surfaces the 401 via a final RoundTrip).
	if got := calls.Load(); got > 2 {
		t.Errorf("expected at most 2 server requests when refresh fails, got %d", got)
	}
}

func TestRefreshTransport_NoopWithoutRefreshFunc(t *testing.T) {
	var calls atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		w.WriteHeader(http.StatusUnauthorized)
	}))
	defer srv.Close()

	c := New(srv.URL) // no SetRefreshFunc
	c.SetAuthToken("stale")

	_, err := c.ListModels(context.Background())
	if err == nil {
		t.Fatal("expected 401 error")
	}
	if calls.Load() != 1 {
		t.Errorf("expected 1 request (no retry without refresh), got %d", calls.Load())
	}
}
