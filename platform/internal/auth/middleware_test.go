package auth

import (
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestExtractBearerToken(t *testing.T) {
	tests := []struct {
		header string
		want   string
	}{
		{"Bearer abc123", "abc123"},
		{"bearer abc123", "abc123"},
		{"BEARER abc123", "abc123"},
		{"Bearer  spaced ", "spaced"},
		{"", ""},
		{"Basic abc123", ""},
		{"Bearerabc123", ""},
		{"Bearer", ""},
	}

	for _, tt := range tests {
		r := httptest.NewRequest(http.MethodGet, "/", nil)
		if tt.header != "" {
			r.Header.Set("Authorization", tt.header)
		}
		got := extractBearerToken(r)
		if got != tt.want {
			t.Errorf("extractBearerToken(%q) = %q, want %q", tt.header, got, tt.want)
		}
	}
}

func TestContextRoundTrip(t *testing.T) {
	// UserFromContext returns nil when no user set
	r := httptest.NewRequest(http.MethodGet, "/", nil)
	if user := UserFromContext(r.Context()); user != nil {
		t.Fatal("expected nil user from empty context")
	}
}
