package handlers

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/ThatCatDev/tanrenai/server/internal/gpuprovider"
)

// mockProvider implements gpuprovider.Provider for InstanceHandler tests.
type mockProvider struct {
	statusResult *gpuprovider.Status
	statusErr    error
	ensureRunErr error
	stopErr      error
}

func (m *mockProvider) Name() string { return "mock" }
func (m *mockProvider) EnsureRunning(_ context.Context) error {
	return m.ensureRunErr
}
func (m *mockProvider) RecordActivity() {}
func (m *mockProvider) Status(_ context.Context) (*gpuprovider.Status, error) {
	return m.statusResult, m.statusErr
}
func (m *mockProvider) Stop(_ context.Context) error {
	return m.stopErr
}
func (m *mockProvider) StartIdleTimer() {}
func (m *mockProvider) Close()          {}

func newInstanceHandler(p gpuprovider.Provider) *InstanceHandler {
	return &InstanceHandler{Provider: p}
}

// ---- Status ----

func TestInstanceStatus(t *testing.T) {
	now := time.Now()
	provider := &mockProvider{
		statusResult: &gpuprovider.Status{
			State:     "running",
			Provider:  "mock",
			GPUURL:    "http://localhost:11435",
			IdleSince: &now,
		},
	}
	h := newInstanceHandler(provider)

	req := httptest.NewRequest(http.MethodGet, "/api/instance/status", nil)
	rec := httptest.NewRecorder()

	h.Status(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", rec.Code, rec.Body.String())
	}
	if ct := rec.Header().Get("Content-Type"); ct != "application/json" {
		t.Errorf("Content-Type = %q, want application/json", ct)
	}

	var got gpuprovider.Status
	if err := json.Unmarshal(rec.Body.Bytes(), &got); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if got.State != "running" {
		t.Errorf("state = %q, want running", got.State)
	}
	if got.Provider != "mock" {
		t.Errorf("provider = %q, want mock", got.Provider)
	}
}

func TestInstanceStatusError(t *testing.T) {
	provider := &mockProvider{
		statusErr: fmt.Errorf("provider unavailable"),
	}
	h := newInstanceHandler(provider)

	req := httptest.NewRequest(http.MethodGet, "/api/instance/status", nil)
	rec := httptest.NewRecorder()

	h.Status(rec, req)

	if rec.Code != http.StatusInternalServerError {
		t.Fatalf("expected 500, got %d: %s", rec.Code, rec.Body.String())
	}
}

func TestInstanceStatusNilResult(t *testing.T) {
	// Status returns nil (no error) — should still encode successfully
	provider := &mockProvider{
		statusResult: nil,
		statusErr:    nil,
	}
	h := newInstanceHandler(provider)

	req := httptest.NewRequest(http.MethodGet, "/api/instance/status", nil)
	rec := httptest.NewRecorder()

	h.Status(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected 200 for nil status, got %d", rec.Code)
	}
}

// ---- Start ----

func TestInstanceStart(t *testing.T) {
	provider := &mockProvider{ensureRunErr: nil}
	h := newInstanceHandler(provider)

	req := httptest.NewRequest(http.MethodPost, "/api/instance/start", nil)
	rec := httptest.NewRecorder()

	h.Start(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", rec.Code, rec.Body.String())
	}

	var got map[string]string
	if err := json.Unmarshal(rec.Body.Bytes(), &got); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if got["status"] != "starting" {
		t.Errorf("status = %q, want starting", got["status"])
	}
}

func TestInstanceStartError(t *testing.T) {
	provider := &mockProvider{
		ensureRunErr: fmt.Errorf("failed to start instance"),
	}
	h := newInstanceHandler(provider)

	req := httptest.NewRequest(http.MethodPost, "/api/instance/start", nil)
	rec := httptest.NewRecorder()

	h.Start(rec, req)

	if rec.Code != http.StatusInternalServerError {
		t.Fatalf("expected 500, got %d: %s", rec.Code, rec.Body.String())
	}

	var errResp map[string]any
	if err := json.Unmarshal(rec.Body.Bytes(), &errResp); err != nil {
		t.Fatalf("decode error response: %v", err)
	}
}

// ---- Stop ----

func TestInstanceStop(t *testing.T) {
	provider := &mockProvider{stopErr: nil}
	h := newInstanceHandler(provider)

	req := httptest.NewRequest(http.MethodPost, "/api/instance/stop", nil)
	rec := httptest.NewRecorder()

	h.Stop(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", rec.Code, rec.Body.String())
	}

	var got map[string]string
	if err := json.Unmarshal(rec.Body.Bytes(), &got); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if got["status"] != "stopped" {
		t.Errorf("status = %q, want stopped", got["status"])
	}
}

func TestInstanceStopError(t *testing.T) {
	provider := &mockProvider{
		stopErr: fmt.Errorf("cannot stop instance"),
	}
	h := newInstanceHandler(provider)

	req := httptest.NewRequest(http.MethodPost, "/api/instance/stop", nil)
	rec := httptest.NewRecorder()

	h.Stop(rec, req)

	if rec.Code != http.StatusInternalServerError {
		t.Fatalf("expected 500, got %d: %s", rec.Code, rec.Body.String())
	}
}

// ---- Error response shape ----

func TestInstanceErrorResponseShape(t *testing.T) {
	provider := &mockProvider{
		statusErr: fmt.Errorf("something went wrong"),
	}
	h := newInstanceHandler(provider)

	req := httptest.NewRequest(http.MethodGet, "/api/instance/status", nil)
	rec := httptest.NewRecorder()

	h.Status(rec, req)

	if rec.Code != http.StatusInternalServerError {
		t.Fatalf("expected 500, got %d", rec.Code)
	}

	// The error response should be JSON with an "error" key.
	var errResp map[string]any
	if err := json.Unmarshal(rec.Body.Bytes(), &errResp); err != nil {
		t.Fatalf("decode error body: %v", err)
	}
	if _, ok := errResp["error"]; !ok {
		t.Errorf("error response missing 'error' key: %v", errResp)
	}
}
