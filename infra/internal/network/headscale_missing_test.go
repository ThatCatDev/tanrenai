package network

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"
)

// TestHeadscaleGenerateAuthKeyDecodeError tests error when response is invalid JSON.
func TestHeadscaleGenerateAuthKeyDecodeError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/api/v1/user" {
			w.Header().Set("Content-Type", "application/json")
			// Valid users response
			w.Write([]byte(`{"users":[{"id":"1","name":"tanrenai"}]}`))
		} else {
			// Invalid JSON for preauthkey response
			w.Header().Set("Content-Type", "application/json")
			w.Write([]byte(`{invalid json}`))
		}
	}))
	defer srv.Close()

	p := &HeadscaleProvider{
		baseURL:    srv.URL,
		apiKey:     "key",
		user:       "tanrenai",
		httpClient: srv.Client(),
	}

	_, err := p.GenerateAuthKey(context.Background())
	if err == nil {
		t.Error("expected error on invalid JSON preauthkey response")
	}
}

// TestHeadscaleGetUserIDDecodeError tests error when users response is invalid JSON.
func TestHeadscaleGetUserIDDecodeError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{invalid json}`))
	}))
	defer srv.Close()

	p := &HeadscaleProvider{
		baseURL:    srv.URL,
		apiKey:     "key",
		user:       "tanrenai",
		httpClient: srv.Client(),
	}

	_, err := p.getUserID(context.Background(), "tanrenai")
	if err == nil {
		t.Error("expected error on invalid JSON users response")
	}
}

// TestHeadscaleListNodesDecodeError tests error when nodes response is invalid JSON.
func TestHeadscaleListNodesDecodeError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{invalid json}`))
	}))
	defer srv.Close()

	p := &HeadscaleProvider{
		baseURL:    srv.URL,
		apiKey:     "key",
		user:       "user",
		httpClient: srv.Client(),
	}

	_, err := p.ListNodes(context.Background())
	if err == nil {
		t.Error("expected error on invalid JSON nodes response")
	}
}

// TestHeadscaleGetUserIDRequestError tests error when the HTTP request fails.
func TestHeadscaleGetUserIDStatusError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "internal error", http.StatusInternalServerError)
	}))
	defer srv.Close()

	p := &HeadscaleProvider{
		baseURL:    srv.URL,
		apiKey:     "key",
		user:       "tanrenai",
		httpClient: srv.Client(),
	}

	_, err := p.getUserID(context.Background(), "tanrenai")
	if err == nil {
		t.Error("expected error on 500 response")
	}
}
