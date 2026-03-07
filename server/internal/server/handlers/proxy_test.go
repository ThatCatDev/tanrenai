package handlers

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/ThatCatDev/tanrenai/server/internal/gpuclient"
	"github.com/ThatCatDev/tanrenai/server/internal/gpuprovider"
	"github.com/ThatCatDev/tanrenai/server/pkg/api"
)

// fakeProvider implements gpuprovider.Provider for testing.
type fakeProvider struct {
	running bool
}

func (f *fakeProvider) Name() string                              { return "fake" }
func (f *fakeProvider) EnsureRunning(_ context.Context) error {
	if !f.running {
		return fmt.Errorf("gpu not running")
	}
	return nil
}
func (f *fakeProvider) RecordActivity()                           {}
func (f *fakeProvider) Status(_ context.Context) (*gpuprovider.Status, error) { return nil, nil }
func (f *fakeProvider) Stop(_ context.Context) error              { return nil }
func (f *fakeProvider) StartIdleTimer()                           {}
func (f *fakeProvider) Close()                                    {}

// newProxyTestHandler creates a ProxyHandler backed by a mock GPU httptest.Server.
// The caller provides a handler that simulates the GPU server's behavior.
func newProxyTestHandler(t *testing.T, gpuHandler http.Handler) (*ProxyHandler, *httptest.Server) {
	t.Helper()
	gpu := httptest.NewServer(gpuHandler)
	t.Cleanup(gpu.Close)

	client := gpuclient.New(gpu.URL)
	h := &ProxyHandler{
		GPUClient: client,
		Provider:  &fakeProvider{running: true},
	}
	return h, gpu
}

func TestChatCompletions(t *testing.T) {
	wantResp := api.ChatCompletionResponse{
		ID:      "chatcmpl-123",
		Object:  "chat.completion",
		Created: 1700000000,
		Model:   "test-model",
		Choices: []api.Choice{
			{
				Index:        0,
				Message:      api.Message{Role: "assistant", Content: "Hello!"},
				FinishReason: "stop",
			},
		},
		Usage: &api.Usage{PromptTokens: 5, CompletionTokens: 2, TotalTokens: 7},
	}

	handler, _ := newProxyTestHandler(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/chat/completions" {
			http.Error(w, "not found", http.StatusNotFound)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(wantResp)
	}))

	body := `{"model":"test-model","messages":[{"role":"user","content":"Hi"}]}`
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()

	handler.ChatCompletions(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", rec.Code, rec.Body.String())
	}

	var got api.ChatCompletionResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &got); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if got.ID != wantResp.ID {
		t.Errorf("ID = %q, want %q", got.ID, wantResp.ID)
	}
	if len(got.Choices) != 1 || got.Choices[0].Message.Content != "Hello!" {
		t.Errorf("unexpected choices: %+v", got.Choices)
	}
}

func TestChatCompletionsStreaming(t *testing.T) {
	sseData := "data: {\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hi\"}}]}\n\ndata: [DONE]\n\n"

	handler, _ := newProxyTestHandler(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/chat/completions" {
			http.Error(w, "not found", http.StatusNotFound)
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(sseData))
	}))

	body := `{"model":"test-model","messages":[{"role":"user","content":"Hi"}],"stream":true}`
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()

	handler.ChatCompletions(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", rec.Code, rec.Body.String())
	}
	if ct := rec.Header().Get("Content-Type"); ct != "text/event-stream" {
		t.Errorf("Content-Type = %q, want text/event-stream", ct)
	}
	if !strings.Contains(rec.Body.String(), "data: [DONE]") {
		t.Errorf("response body missing SSE terminator, got: %s", rec.Body.String())
	}
}

func TestTokenize(t *testing.T) {
	handler, _ := newProxyTestHandler(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/tokenize" {
			http.Error(w, "not found", http.StatusNotFound)
			return
		}
		// The gpuclient.Tokenize returns len(tokens), so we return 5 tokens.
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string][]int{"tokens": {1, 2, 3, 4, 5}})
	}))

	body := `{"content":"hello world test"}`
	req := httptest.NewRequest(http.MethodPost, "/tokenize", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()

	handler.Tokenize(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", rec.Code, rec.Body.String())
	}

	var got map[string]int
	if err := json.Unmarshal(rec.Body.Bytes(), &got); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if got["count"] != 5 {
		t.Errorf("count = %d, want 5", got["count"])
	}
}

func TestListModels(t *testing.T) {
	wantResp := api.ModelListResponse{
		Object: "list",
		Data: []api.ModelInfo{
			{ID: "model-a", Object: "model", Created: 1700000000, OwnedBy: "test"},
			{ID: "model-b", Object: "model", Created: 1700000001, OwnedBy: "test"},
		},
	}

	handler, _ := newProxyTestHandler(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/models" {
			http.Error(w, "not found", http.StatusNotFound)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(wantResp)
	}))

	req := httptest.NewRequest(http.MethodGet, "/v1/models", nil)
	rec := httptest.NewRecorder()

	handler.ListModels(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", rec.Code, rec.Body.String())
	}

	var got api.ModelListResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &got); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if len(got.Data) != 2 {
		t.Errorf("got %d models, want 2", len(got.Data))
	}
	if got.Data[0].ID != "model-a" {
		t.Errorf("first model ID = %q, want model-a", got.Data[0].ID)
	}
}

func TestLoadModel(t *testing.T) {
	handler, _ := newProxyTestHandler(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/load" {
			http.Error(w, "not found", http.StatusNotFound)
			return
		}
		var req map[string]string
		json.NewDecoder(r.Body).Decode(&req)
		if req["model"] != "my-model" {
			http.Error(w, "wrong model", http.StatusBadRequest)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(`{"status":"ok"}`))
	}))

	body := `{"model":"my-model"}`
	req := httptest.NewRequest(http.MethodPost, "/api/load", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()

	handler.LoadModel(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", rec.Code, rec.Body.String())
	}

	var got map[string]string
	if err := json.Unmarshal(rec.Body.Bytes(), &got); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if got["status"] != "ok" {
		t.Errorf("status = %q, want ok", got["status"])
	}
}

func TestRawProxy(t *testing.T) {
	handler, _ := newProxyTestHandler(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/some/custom/path" {
			http.Error(w, "not found", http.StatusNotFound)
			return
		}
		body, _ := io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(fmt.Sprintf(`{"echo":%q,"method":%q}`, string(body), r.Method)))
	}))

	req := httptest.NewRequest(http.MethodPost, "/some/custom/path", strings.NewReader(`{"key":"value"}`))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()

	handler.RawProxy(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", rec.Code, rec.Body.String())
	}
	if ct := rec.Header().Get("Content-Type"); ct != "application/json" {
		t.Errorf("Content-Type = %q, want application/json", ct)
	}

	var got map[string]string
	if err := json.Unmarshal(rec.Body.Bytes(), &got); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if got["method"] != "POST" {
		t.Errorf("method = %q, want POST", got["method"])
	}
}

func TestGPUDown(t *testing.T) {
	// Use a provider that reports GPU as not running.
	handler := &ProxyHandler{
		GPUClient: gpuclient.New("http://127.0.0.1:1"), // unreachable
		Provider:  &fakeProvider{running: false},
	}

	body := `{"model":"test","messages":[{"role":"user","content":"hi"}]}`
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()

	handler.ChatCompletions(rec, req)

	if rec.Code != http.StatusServiceUnavailable {
		t.Fatalf("expected 503, got %d: %s", rec.Code, rec.Body.String())
	}

	var errResp api.ErrorResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &errResp); err != nil {
		t.Fatalf("decode error response: %v", err)
	}
	if errResp.Error.Code != "gpu_unavailable" {
		t.Errorf("error code = %q, want gpu_unavailable", errResp.Error.Code)
	}
}

func TestBadJSON(t *testing.T) {
	handler, _ := newProxyTestHandler(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		t.Fatal("GPU server should not be called for bad JSON")
	}))

	// Send invalid JSON to ChatCompletions.
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{invalid json`))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()

	handler.ChatCompletions(rec, req)

	if rec.Code != http.StatusBadRequest {
		t.Fatalf("expected 400, got %d: %s", rec.Code, rec.Body.String())
	}

	var errResp api.ErrorResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &errResp); err != nil {
		t.Fatalf("decode error response: %v", err)
	}
	if errResp.Error.Code != "invalid_request" {
		t.Errorf("error code = %q, want invalid_request", errResp.Error.Code)
	}
}

func TestBadJSONTokenize(t *testing.T) {
	handler, _ := newProxyTestHandler(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		t.Fatal("GPU server should not be called for bad JSON")
	}))

	req := httptest.NewRequest(http.MethodPost, "/tokenize", strings.NewReader(`not json`))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()

	handler.Tokenize(rec, req)

	if rec.Code != http.StatusBadRequest {
		t.Fatalf("expected 400, got %d: %s", rec.Code, rec.Body.String())
	}
}

func TestBadJSONLoadModel(t *testing.T) {
	handler, _ := newProxyTestHandler(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		t.Fatal("GPU server should not be called for bad JSON")
	}))

	req := httptest.NewRequest(http.MethodPost, "/api/load", strings.NewReader(`{bad`))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()

	handler.LoadModel(rec, req)

	if rec.Code != http.StatusBadRequest {
		t.Fatalf("expected 400, got %d: %s", rec.Code, rec.Body.String())
	}
}
