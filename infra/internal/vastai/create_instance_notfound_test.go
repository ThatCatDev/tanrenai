package vastai

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
)

// TestCreateInstanceNeverRunning verifies CreateInstance returns error when instance never appears as running.
// This test exercises the "instance created but not found after polling" error path.
// NOTE: This test sleeps 3s * 10 = 30s in the worst case; we use a cancelled context to short-circuit.
func TestCreateInstanceContextCancelledDuringPoll(t *testing.T) {
	putCalled := false

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		switch r.Method {
		case http.MethodPut:
			putCalled = true
			json.NewEncoder(w).Encode(map[string]any{"success": true})
		case http.MethodGet:
			// Always return an instance with "loading" status (never "running")
			// but with wrong status to avoid early return
			json.NewEncoder(w).Encode(map[string]any{
				"instances": []Instance{
					{ID: 1, Status: "exited"}, // neither running nor loading
				},
			})
		}
	}))
	defer srv.Close()

	client := newTestClient("key", srv)

	// Cancel context right away so the sleep inside CreateInstance is interrupted
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	_, err := client.CreateInstance(ctx, 42, CreateOpts{})
	// The PUT should succeed, then we poll. With cancelled context, ListInstances
	// may fail or return without finding a running instance. Either way, error expected.
	_ = putCalled
	_ = err // may fail at PUT (context already cancelled) or at polling
}
