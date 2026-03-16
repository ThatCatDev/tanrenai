package vastai

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestDestroyInstanceNotFound(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "not found", http.StatusNotFound)
	}))
	defer srv.Close()

	client := newTestClient("key", srv)
	err := client.DestroyInstance(context.Background(), "999")
	if err == nil {
		t.Error("expected error when DELETE returns 404")
	}
	if !strings.Contains(err.Error(), "404") {
		t.Errorf("error should mention 404, got: %v", err)
	}
}

func TestDestroyInstanceServerError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "internal error", http.StatusInternalServerError)
	}))
	defer srv.Close()

	client := newTestClient("key", srv)
	err := client.DestroyInstance(context.Background(), "123")
	if err == nil {
		t.Error("expected error when DELETE returns 500")
	}
}

func TestGetInstanceAPIError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "forbidden", http.StatusForbidden)
	}))
	defer srv.Close()

	client := newTestClient("key", srv)
	_, err := client.GetInstance(context.Background(), "1")
	if err == nil {
		t.Error("expected error when GET returns 403")
	}
}

func TestSearchOffersPostError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Return invalid JSON
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte("{invalid json response}"))
	}))
	defer srv.Close()

	client := newTestClient("key", srv)
	_, err := client.SearchOffers(context.Background(), SearchQuery{})
	if err == nil {
		t.Error("expected error on invalid JSON response")
	}
}

func TestGetInstanceInvalidJSON(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte("{bad json}"))
	}))
	defer srv.Close()

	client := newTestClient("key", srv)
	_, err := client.GetInstance(context.Background(), "1")
	if err == nil {
		t.Error("expected error on invalid JSON")
	}
}

func TestListInstancesInvalidJSON(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte("{bad json}"))
	}))
	defer srv.Close()

	client := newTestClient("key", srv)
	_, err := client.ListInstances(context.Background())
	if err == nil {
		t.Error("expected error on invalid JSON")
	}
}

func TestStartInstanceError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "forbidden", http.StatusForbidden)
	}))
	defer srv.Close()

	client := newTestClient("key", srv)
	err := client.StartInstance(context.Background(), "1")
	if err == nil {
		t.Error("expected error when StartInstance returns 403")
	}
}

func TestStopInstanceError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "unauthorized", http.StatusUnauthorized)
	}))
	defer srv.Close()

	client := newTestClient("key", srv)
	err := client.StopInstance(context.Background(), "2")
	if err == nil {
		t.Error("expected error when StopInstance returns 401")
	}
}
