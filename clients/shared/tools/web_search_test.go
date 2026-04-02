package tools

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// newTestDDGClient replaces ddgClient for tests using a redirect transport.
func newTestDDGClient(srv *httptest.Server) *http.Client {
	return &http.Client{
		Transport: &redirectTransport{target: srv.URL},
	}
}

// redirectTransport rewrites all requests to point at the test server.
type redirectTransport struct{ target string }

func (rt *redirectTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	cloned := req.Clone(req.Context())
	cloned.URL.Scheme = "http"
	cloned.URL.Host = strings.TrimPrefix(rt.target, "http://")

	return http.DefaultTransport.RoundTrip(cloned)
}

// ddgHTML returns a minimal DuckDuckGo lite HTML page with n results.
func ddgHTML(results []struct{ title, href, snippet string }) string {
	var sb strings.Builder
	sb.WriteString(`<html><body><table>`)
	for _, r := range results {
		sb.WriteString(`<tr><td><a class="result-link" href="`)
		sb.WriteString(r.href)
		sb.WriteString(`">`)
		sb.WriteString(r.title)
		sb.WriteString(`</a></td></tr>`)
		if r.snippet != "" {
			sb.WriteString(`<tr><td class="result-snippet">`)
			sb.WriteString(r.snippet)
			sb.WriteString(`</td></tr>`)
		}
	}
	sb.WriteString(`</table></body></html>`)

	return sb.String()
}

func TestWebSearchExecute_InvalidArgs(t *testing.T) {
	tool := &WebSearchTool{}
	result, err := tool.Execute(context.Background(), `{not json}`)
	if err != nil {
		t.Fatal(err)
	}
	if !result.IsError {
		t.Fatal("expected error for invalid JSON")
	}
}

func TestWebSearchExecute_EmptyQuery(t *testing.T) {
	tool := &WebSearchTool{}
	result, err := tool.Execute(context.Background(), `{"query":""}`)
	if err != nil {
		t.Fatal(err)
	}
	if !result.IsError {
		t.Fatal("expected error for empty query")
	}
	if !strings.Contains(result.Output, "query is required") {
		t.Errorf("expected 'query is required', got: %s", result.Output)
	}
}

func TestWebSearchExecute_Success(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/html")
		html := ddgHTML([]struct{ title, href, snippet string }{
			{"Go Programming", "https://go.dev", "The Go programming language."},
			{"Go Tour", "https://tour.golang.org", "A tour of Go."},
		})
		_, _ = w.Write([]byte(html))
	}))
	defer srv.Close()

	orig := ddgClient
	ddgClient = newTestDDGClient(srv)
	defer func() { ddgClient = orig }()

	tool := &WebSearchTool{}
	result, err := tool.Execute(context.Background(), `{"query":"golang","max_results":5}`)
	if err != nil {
		t.Fatal(err)
	}
	if result.IsError {
		t.Fatalf("unexpected error: %s", result.Output)
	}
	if !strings.Contains(result.Output, "golang") {
		t.Errorf("expected query in output, got: %s", result.Output)
	}
	if !strings.Contains(result.Output, "Go Programming") {
		t.Errorf("expected title in output, got: %s", result.Output)
	}
	if !strings.Contains(result.Output, "https://go.dev") {
		t.Errorf("expected URL in output, got: %s", result.Output)
	}
	if !strings.Contains(result.Output, "The Go programming language") {
		t.Errorf("expected snippet in output, got: %s", result.Output)
	}
}

func TestWebSearchExecute_NoResults(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/html")
		_, _ = w.Write([]byte(`<html><body><table></table></body></html>`))
	}))
	defer srv.Close()

	orig := ddgClient
	ddgClient = newTestDDGClient(srv)
	defer func() { ddgClient = orig }()

	tool := &WebSearchTool{}
	result, err := tool.Execute(context.Background(), `{"query":"xyzzy_no_results_42"}`)
	if err != nil {
		t.Fatal(err)
	}
	if result.IsError {
		t.Fatalf("unexpected error: %s", result.Output)
	}
	if !strings.Contains(result.Output, "No results found") {
		t.Errorf("expected 'No results found', got: %s", result.Output)
	}
}

func TestWebSearchExecute_DefaultMaxResults(t *testing.T) {
	// When max_results <= 0 it should default to 5 without panicking.
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/html")
		_, _ = w.Write([]byte(`<html><body><table></table></body></html>`))
	}))
	defer srv.Close()

	orig := ddgClient
	ddgClient = newTestDDGClient(srv)
	defer func() { ddgClient = orig }()

	tool := &WebSearchTool{}
	result, err := tool.Execute(context.Background(), `{"query":"test"}`)
	if err != nil {
		t.Fatal(err)
	}
	// No error from the tool itself — zero results is fine.
	_ = result
}

func TestWebSearchExecute_HTTPError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "server error", http.StatusInternalServerError)
	}))
	defer srv.Close()

	orig := ddgClient
	ddgClient = newTestDDGClient(srv)
	defer func() { ddgClient = orig }()

	tool := &WebSearchTool{}
	result, err := tool.Execute(context.Background(), `{"query":"test"}`)
	if err != nil {
		t.Fatal(err)
	}
	if !result.IsError {
		t.Fatal("expected error result for HTTP 500")
	}
	if !strings.Contains(result.Output, "web search failed") {
		t.Errorf("expected 'web search failed', got: %s", result.Output)
	}
}

func TestWebSearchExecute_ResultWithoutSnippet(t *testing.T) {
	// Ensure result with no snippet still renders correctly.
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/html")
		html := ddgHTML([]struct{ title, href, snippet string }{
			{"Result Without Snippet", "https://example.com", ""},
		})
		_, _ = w.Write([]byte(html))
	}))
	defer srv.Close()

	orig := ddgClient
	ddgClient = newTestDDGClient(srv)
	defer func() { ddgClient = orig }()

	tool := &WebSearchTool{}
	result, err := tool.Execute(context.Background(), `{"query":"test","max_results":1}`)
	if err != nil {
		t.Fatal(err)
	}
	if result.IsError {
		t.Fatalf("unexpected error: %s", result.Output)
	}
	if !strings.Contains(result.Output, "Result Without Snippet") {
		t.Errorf("expected title, got: %s", result.Output)
	}
}

func TestWebSearchExecute_CancelledContext(t *testing.T) {
	// A cancelled context should cause searchDuckDuckGo to fail with an error.
	ctx, cancel := context.WithCancel(context.Background())
	cancel() // cancel immediately

	tool := &WebSearchTool{}
	result, err := tool.Execute(ctx, `{"query":"test"}`)
	if err != nil {
		t.Fatal(err)
	}
	// The tool returns an error result (not a Go error) for network failures.
	if !result.IsError {
		t.Fatal("expected error result for cancelled context")
	}
}

func TestWebSearchMetadata(t *testing.T) {
	tool := &WebSearchTool{}
	if tool.Name() != "web_search" {
		t.Errorf("unexpected name: %s", tool.Name())
	}
	if tool.Description() == "" {
		t.Error("expected non-empty description")
	}
	params := tool.Parameters()
	if len(params) == 0 {
		t.Error("expected non-empty parameters")
	}
}
