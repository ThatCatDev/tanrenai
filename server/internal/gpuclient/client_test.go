package gpuclient

import (
	"context"
	"encoding/json"
	"io"
	"math"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/ThatCatDev/tanrenai/server/pkg/api"
)

// newTestServer creates an httptest.Server with the given handler and returns
// a Client pointing at it. The server is closed when the test ends.
func newTestServer(t *testing.T, handler http.Handler) (*Client, *httptest.Server) {
	t.Helper()
	srv := httptest.NewServer(handler)
	t.Cleanup(srv.Close)

	return New(srv.URL), srv
}

// ---- New / BaseURL ----

func TestNew(t *testing.T) {
	c := New("http://example.com:11435")
	if c == nil {
		t.Fatal("New returned nil")
	}
	if c.httpClient == nil {
		t.Error("httpClient is nil")
	}
	if c.streamClient == nil {
		t.Error("streamClient is nil")
	}
}

func TestBaseURL(t *testing.T) {
	c := New("http://gpu.local:11435")
	if c.BaseURL() != "http://gpu.local:11435" {
		t.Errorf("BaseURL() = %q, want http://gpu.local:11435", c.BaseURL())
	}
}

// ---- ChatCompletion ----

func TestChatCompletion(t *testing.T) {
	want := api.ChatCompletionResponse{
		ID:     "chatcmpl-1",
		Object: "chat.completion",
		Model:  "test-model",
		Choices: []api.Choice{
			{Index: 0, Message: api.Message{Role: "assistant", Content: "Hello!"}, FinishReason: "stop"},
		},
		Usage: &api.Usage{PromptTokens: 3, CompletionTokens: 2, TotalTokens: 5},
	}

	client, _ := newTestServer(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/chat/completions" {
			http.Error(w, "not found", http.StatusNotFound)

			return
		}
		if r.Method != http.MethodPost {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)

			return
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(want)
	}))

	req := &api.ChatCompletionRequest{
		Model:    "test-model",
		Messages: []api.Message{{Role: "user", Content: "Hi"}},
	}
	got, err := client.ChatCompletion(context.Background(), req)
	if err != nil {
		t.Fatalf("ChatCompletion: %v", err)
	}
	if got.ID != want.ID {
		t.Errorf("ID = %q, want %q", got.ID, want.ID)
	}
	if len(got.Choices) != 1 || got.Choices[0].Message.Content != "Hello!" {
		t.Errorf("unexpected choices: %+v", got.Choices)
	}
}

func TestChatCompletion500(t *testing.T) {
	client, _ := newTestServer(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "internal error", http.StatusInternalServerError)
	}))

	req := &api.ChatCompletionRequest{
		Model:    "test-model",
		Messages: []api.Message{{Role: "user", Content: "Hi"}},
	}
	_, err := client.ChatCompletion(context.Background(), req)
	if err == nil {
		t.Fatal("expected error for 500, got nil")
	}
}

func TestChatCompletionBadJSON(t *testing.T) {
	client, _ := newTestServer(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(`{not valid json`))
	}))

	req := &api.ChatCompletionRequest{
		Model:    "test-model",
		Messages: []api.Message{{Role: "user", Content: "Hi"}},
	}
	_, err := client.ChatCompletion(context.Background(), req)
	if err == nil {
		t.Fatal("expected error for bad JSON response, got nil")
	}
}

func TestChatCompletionConnectionRefused(t *testing.T) {
	// Port 1 is effectively unreachable; use a short timeout context so the test
	// completes quickly rather than waiting for the 5-minute http.Client timeout.
	client := New("http://127.0.0.1:1")

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	req := &api.ChatCompletionRequest{
		Model:    "test-model",
		Messages: []api.Message{{Role: "user", Content: "Hi"}},
	}
	_, err := client.ChatCompletion(ctx, req)
	if err == nil {
		t.Fatal("expected error for connection refused, got nil")
	}
}

// ---- StreamCompletionRaw ----

func TestStreamCompletionRaw(t *testing.T) {
	sseData := "data: {\"id\":\"1\",\"choices\":[]}\n\ndata: [DONE]\n\n"

	client, _ := newTestServer(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(sseData))
	}))

	req := &api.ChatCompletionRequest{
		Model:    "test-model",
		Messages: []api.Message{{Role: "user", Content: "Hi"}},
	}
	body, err := client.StreamCompletionRaw(context.Background(), req)
	if err != nil {
		t.Fatalf("StreamCompletionRaw: %v", err)
	}
	defer body.Close()

	buf := make([]byte, 1024)
	n, _ := body.Read(buf)
	if n == 0 {
		t.Error("expected non-empty stream body")
	}
}

func TestStreamCompletionRaw500(t *testing.T) {
	client, _ := newTestServer(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "gpu error", http.StatusInternalServerError)
	}))

	req := &api.ChatCompletionRequest{
		Model:    "test-model",
		Messages: []api.Message{{Role: "user", Content: "Hi"}},
	}
	_, err := client.StreamCompletionRaw(context.Background(), req)
	if err == nil {
		t.Fatal("expected error for 500, got nil")
	}
}

// ---- Tokenize ----

func TestTokenize(t *testing.T) {
	client, _ := newTestServer(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/tokenize" {
			http.Error(w, "not found", http.StatusNotFound)

			return
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string][]int{"tokens": {1, 2, 3, 4, 5}})
	}))

	count, err := client.Tokenize(context.Background(), "hello world")
	if err != nil {
		t.Fatalf("Tokenize: %v", err)
	}
	if count != 5 {
		t.Errorf("Tokenize count = %d, want 5", count)
	}
}

func TestTokenize500(t *testing.T) {
	client, _ := newTestServer(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "tokenize error", http.StatusInternalServerError)
	}))

	_, err := client.Tokenize(context.Background(), "text")
	if err == nil {
		t.Fatal("expected error for 500, got nil")
	}
}

func TestTokenizeBadJSON(t *testing.T) {
	client, _ := newTestServer(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(`{bad json`))
	}))

	_, err := client.Tokenize(context.Background(), "text")
	if err == nil {
		t.Fatal("expected error for bad JSON, got nil")
	}
}

// ---- Embed ----

func TestEmbed(t *testing.T) {
	embedding := []float32{0.1, 0.2, 0.3, 0.4}

	client, _ := newTestServer(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/embeddings" {
			http.Error(w, "not found", http.StatusNotFound)

			return
		}
		resp := api.EmbeddingResponse{
			Data: []api.EmbeddingData{
				{Embedding: embedding, Index: 0},
			},
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(resp)
	}))

	got, err := client.Embed(context.Background(), "hello world")
	if err != nil {
		t.Fatalf("Embed: %v", err)
	}
	if len(got) != len(embedding) {
		t.Errorf("Embed returned %d values, want %d", len(got), len(embedding))
	}
}

func TestEmbedEmptyData(t *testing.T) {
	client, _ := newTestServer(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		resp := api.EmbeddingResponse{Data: []api.EmbeddingData{}}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(resp)
	}))

	_, err := client.Embed(context.Background(), "text")
	if err == nil {
		t.Fatal("expected error for empty embedding data, got nil")
	}
}

func TestEmbed500(t *testing.T) {
	client, _ := newTestServer(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "embed error", http.StatusInternalServerError)
	}))

	_, err := client.Embed(context.Background(), "text")
	if err == nil {
		t.Fatal("expected error for 500, got nil")
	}
}

func TestEmbedBadJSON(t *testing.T) {
	client, _ := newTestServer(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(`{bad`))
	}))

	_, err := client.Embed(context.Background(), "text")
	if err == nil {
		t.Fatal("expected error for bad JSON, got nil")
	}
}

// ---- LoadModel ----

func TestLoadModel(t *testing.T) {
	client, _ := newTestServer(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/load" {
			http.Error(w, "not found", http.StatusNotFound)

			return
		}
		resp := api.LoadResponse{Status: "loaded", Model: "my-model", CtxSize: 4096}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(resp)
	}))

	got, err := client.LoadModel(context.Background(), "my-model")
	if err != nil {
		t.Fatalf("LoadModel: %v", err)
	}
	if got.Status != "loaded" {
		t.Errorf("Status = %q, want loaded", got.Status)
	}
	if got.Model != "my-model" {
		t.Errorf("Model = %q, want my-model", got.Model)
	}
	if got.CtxSize != 4096 {
		t.Errorf("CtxSize = %d, want 4096", got.CtxSize)
	}
}

func TestLoadModel500(t *testing.T) {
	client, _ := newTestServer(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "load error", http.StatusInternalServerError)
	}))

	_, err := client.LoadModel(context.Background(), "model")
	if err == nil {
		t.Fatal("expected error for 500, got nil")
	}
}

// ---- ListModels ----

func TestListModels(t *testing.T) {
	want := api.ModelListResponse{
		Object: "list",
		Data: []api.ModelInfo{
			{ID: "model-a", Object: "model", Created: 1700000000, OwnedBy: "test"},
			{ID: "model-b", Object: "model", Created: 1700000001, OwnedBy: "test"},
		},
	}

	client, _ := newTestServer(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/models" {
			http.Error(w, "not found", http.StatusNotFound)

			return
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(want)
	}))

	got, err := client.ListModels(context.Background())
	if err != nil {
		t.Fatalf("ListModels: %v", err)
	}
	if len(got.Data) != 2 {
		t.Errorf("got %d models, want 2", len(got.Data))
	}
	if got.Data[0].ID != "model-a" {
		t.Errorf("first model ID = %q, want model-a", got.Data[0].ID)
	}
}

func TestListModels500(t *testing.T) {
	client, _ := newTestServer(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "list error", http.StatusInternalServerError)
	}))

	_, err := client.ListModels(context.Background())
	if err == nil {
		t.Fatal("expected error for 500, got nil")
	}
}

// ---- Health ----

func TestHealth(t *testing.T) {
	client, _ := newTestServer(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/health" {
			http.Error(w, "not found", http.StatusNotFound)

			return
		}
		w.WriteHeader(http.StatusOK)
	}))

	if err := client.Health(context.Background()); err != nil {
		t.Errorf("Health: %v", err)
	}
}

func TestHealth500(t *testing.T) {
	client, _ := newTestServer(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "unhealthy", http.StatusInternalServerError)
	}))

	if err := client.Health(context.Background()); err == nil {
		t.Fatal("expected error for 500, got nil")
	}
}

func TestHealthConnectionRefused(t *testing.T) {
	client := New("http://127.0.0.1:1")
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	if err := client.Health(ctx); err == nil {
		t.Fatal("expected error for unreachable server, got nil")
	}
}

// ---- PullModelStream ----

func TestPullModelStream(t *testing.T) {
	client, _ := newTestServer(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/pull" {
			http.Error(w, "not found", http.StatusNotFound)

			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		w.Write([]byte("data: {\"status\":\"downloading\"}\n\n"))
	}))

	body, err := client.PullModelStream(context.Background(), "http://example.com/model.gguf", "")
	if err != nil {
		t.Fatalf("PullModelStream: %v", err)
	}
	defer body.Close()
}

func TestPullModelStream_ForwardsName(t *testing.T) {
	gotBody := make(chan []byte, 1)
	client, _ := newTestServer(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		gotBody <- b
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		w.Write([]byte("data: {\"status\":\"downloading\"}\n\n"))
	}))

	body, err := client.PullModelStream(context.Background(), "http://example.com/x.gguf", "Qwen3.6-35B-A3B-Q4_K_M")
	if err != nil {
		t.Fatalf("PullModelStream: %v", err)
	}
	defer body.Close()

	b := <-gotBody
	var got struct {
		URL  string `json:"url"`
		Name string `json:"name"`
	}
	if err := json.Unmarshal(b, &got); err != nil {
		t.Fatalf("unmarshal request body: %v", err)
	}
	if got.Name != "Qwen3.6-35B-A3B-Q4_K_M" {
		t.Errorf("Name = %q, want Qwen3.6-35B-A3B-Q4_K_M", got.Name)
	}
}

func TestPullModelStream500(t *testing.T) {
	client, _ := newTestServer(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "pull error", http.StatusInternalServerError)
	}))

	_, err := client.PullModelStream(context.Background(), "http://example.com/model.gguf", "")
	if err == nil {
		t.Fatal("expected error for 500, got nil")
	}
}

// ---- RawRequest ----

func TestRawRequest(t *testing.T) {
	client, _ := newTestServer(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/custom/path" {
			http.Error(w, "not found", http.StatusNotFound)

			return
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(`{"ok":true}`))
	}))

	resp, err := client.RawRequest(context.Background(), http.MethodGet, "/custom/path", nil)
	if err != nil {
		t.Fatalf("RawRequest: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		t.Errorf("status = %d, want 200", resp.StatusCode)
	}
}

// ---- normalizeVector ----

func TestNormalizeVector(t *testing.T) {
	v := []float32{3, 4} // magnitude = 5
	normalizeVector(v)

	const eps = 1e-6
	if math.Abs(float64(v[0])-0.6) > eps {
		t.Errorf("v[0] = %f, want ~0.6", v[0])
	}
	if math.Abs(float64(v[1])-0.8) > eps {
		t.Errorf("v[1] = %f, want ~0.8", v[1])
	}
}

func TestNormalizeVectorUnitLength(t *testing.T) {
	v := []float32{1, 0, 0, 0}
	normalizeVector(v)

	if v[0] != 1.0 {
		t.Errorf("unit vector[0] = %f, want 1.0", v[0])
	}
}

func TestNormalizeVectorZeroVector(t *testing.T) {
	v := []float32{0, 0, 0}
	// Should not panic or divide by zero
	normalizeVector(v)

	for i, x := range v {
		if x != 0 {
			t.Errorf("v[%d] = %f, want 0 (zero vector should be unchanged)", i, x)
		}
	}
}

func TestNormalizeVectorResultIsUnit(t *testing.T) {
	v := []float32{1, 2, 3, 4, 5}
	normalizeVector(v)

	var sum float64
	for _, x := range v {
		sum += float64(x) * float64(x)
	}
	norm := math.Sqrt(sum)
	const eps = 1e-5
	if math.Abs(norm-1.0) > eps {
		t.Errorf("normalized vector magnitude = %f, want ~1.0", norm)
	}
}
