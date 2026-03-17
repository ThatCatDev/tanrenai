package apiclient

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/ThatCatDev/tanrenai/shared/pkg/api"
)

func TestNew(t *testing.T) {
	c := New("http://localhost:8080")
	if c.BaseURL() != "http://localhost:8080" {
		t.Fatalf("expected base URL http://localhost:8080, got %s", c.BaseURL())
	}
	if c.httpClient.Timeout != 2*time.Minute {
		t.Fatalf("expected 2 minute timeout, got %v", c.httpClient.Timeout)
	}
	if c.streamClient.Timeout != 0 {
		t.Fatal("expected no timeout on stream client")
	}
}

func TestSetBaseURL(t *testing.T) {
	c := New("http://old:8080")
	c.SetBaseURL("http://new:9090")
	if c.BaseURL() != "http://new:9090" {
		t.Fatalf("expected updated URL, got %s", c.BaseURL())
	}
}

func TestChatCompletion(t *testing.T) {
	want := &api.ChatCompletionResponse{
		ID:     "chatcmpl-123",
		Object: "chat.completion",
		Model:  "test-model",
		Choices: []api.Choice{
			{
				Index: 0,
				Message: api.Message{
					Role:    "assistant",
					Content: "Hello, world!",
				},
				FinishReason: "stop",
			},
		},
	}

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/chat/completions" {
			t.Errorf("unexpected path: %s", r.URL.Path)
		}
		if r.Method != http.MethodPost {
			t.Errorf("unexpected method: %s", r.Method)
		}
		if r.Header.Get("Content-Type") != "application/json" {
			t.Errorf("unexpected content type: %s", r.Header.Get("Content-Type"))
		}

		// Verify request body has stream=false
		var req api.ChatCompletionRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Fatalf("decode request: %v", err)
		}
		if req.Stream {
			t.Error("expected stream=false for non-streaming request")
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(want)
	}))
	defer srv.Close()

	c := New(srv.URL)
	got, err := c.ChatCompletion(context.Background(), &api.ChatCompletionRequest{
		Model:    "test-model",
		Messages: []api.Message{{Role: "user", Content: "Hi"}},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got.ID != want.ID {
		t.Errorf("ID = %q, want %q", got.ID, want.ID)
	}
	if got.Model != want.Model {
		t.Errorf("Model = %q, want %q", got.Model, want.Model)
	}
	if len(got.Choices) != 1 {
		t.Fatalf("expected 1 choice, got %d", len(got.Choices))
	}
	if got.Choices[0].Message.Content != "Hello, world!" {
		t.Errorf("Content = %q, want %q", got.Choices[0].Message.Content, "Hello, world!")
	}
	if got.Choices[0].FinishReason != "stop" {
		t.Errorf("FinishReason = %q, want %q", got.Choices[0].FinishReason, "stop")
	}
}

func TestMemorySearch(t *testing.T) {
	want := &api.MemorySearchResponse{
		Results: []api.MemorySearchResult{
			{
				Entry: api.MemoryEntry{
					ID:        "mem-1",
					UserMsg:   "What is Go?",
					AssistMsg: "Go is a programming language.",
				},
				SemanticScore: 0.95,
				KeywordScore:  0.8,
				CombinedScore: 0.9,
			},
		},
	}

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/memory/search" {
			t.Errorf("unexpected path: %s", r.URL.Path)
		}
		if r.Method != http.MethodPost {
			t.Errorf("unexpected method: %s", r.Method)
		}

		var req api.MemorySearchRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Fatalf("decode request: %v", err)
		}
		if req.Query != "Go language" {
			t.Errorf("Query = %q, want %q", req.Query, "Go language")
		}
		if req.Limit != 5 {
			t.Errorf("Limit = %d, want %d", req.Limit, 5)
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(want)
	}))
	defer srv.Close()

	c := New(srv.URL)
	got, err := c.MemorySearch(context.Background(), "Go language", 5)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(got.Results) != 1 {
		t.Fatalf("expected 1 result, got %d", len(got.Results))
	}
	if got.Results[0].Entry.ID != "mem-1" {
		t.Errorf("Entry.ID = %q, want %q", got.Results[0].Entry.ID, "mem-1")
	}
	if got.Results[0].CombinedScore != 0.9 {
		t.Errorf("CombinedScore = %f, want %f", got.Results[0].CombinedScore, 0.9)
	}
}

func TestMemoryStore(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/memory/store" {
			t.Errorf("unexpected path: %s", r.URL.Path)
		}
		if r.Method != http.MethodPost {
			t.Errorf("unexpected method: %s", r.Method)
		}

		var req api.MemoryStoreRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Fatalf("decode request: %v", err)
		}
		if req.UserMsg != "Hello" {
			t.Errorf("UserMsg = %q, want %q", req.UserMsg, "Hello")
		}
		if req.AssistMsg != "Hi there" {
			t.Errorf("AssistMsg = %q, want %q", req.AssistMsg, "Hi there")
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(api.MemoryStoreResponse{ID: "mem-42"})
	}))
	defer srv.Close()

	c := New(srv.URL)
	id, err := c.MemoryStore(context.Background(), "Hello", "Hi there")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if id != "mem-42" {
		t.Errorf("ID = %q, want %q", id, "mem-42")
	}
}

func TestListModels(t *testing.T) {
	want := &api.ModelListResponse{
		Object: "list",
		Data: []api.ModelInfo{
			{ID: "model-a", Object: "model", Created: 1700000000, OwnedBy: "test"},
			{ID: "model-b", Object: "model", Created: 1700000001, OwnedBy: "test"},
		},
	}

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/models" {
			t.Errorf("unexpected path: %s", r.URL.Path)
		}
		if r.Method != http.MethodGet {
			t.Errorf("unexpected method: %s", r.Method)
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(want)
	}))
	defer srv.Close()

	c := New(srv.URL)
	got, err := c.ListModels(context.Background())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got.Object != "list" {
		t.Errorf("Object = %q, want %q", got.Object, "list")
	}
	if len(got.Data) != 2 {
		t.Fatalf("expected 2 models, got %d", len(got.Data))
	}
	if got.Data[0].ID != "model-a" {
		t.Errorf("Data[0].ID = %q, want %q", got.Data[0].ID, "model-a")
	}
	if got.Data[1].ID != "model-b" {
		t.Errorf("Data[1].ID = %q, want %q", got.Data[1].ID, "model-b")
	}
}

func TestTokenize(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/tokenize" {
			t.Errorf("unexpected path: %s", r.URL.Path)
		}
		if r.Method != http.MethodPost {
			t.Errorf("unexpected method: %s", r.Method)
		}

		var req struct {
			Content string `json:"content"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Fatalf("decode request: %v", err)
		}
		if req.Content != "Hello world" {
			t.Errorf("Content = %q, want %q", req.Content, "Hello world")
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string][]int{
			"tokens": {1, 2, 3, 4, 5},
		})
	}))
	defer srv.Close()

	c := New(srv.URL)
	count, err := c.Tokenize(context.Background(), "Hello world")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if count != 5 {
		t.Errorf("token count = %d, want %d", count, 5)
	}
}

func TestErrorResponse(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
		w.Write([]byte("internal server error"))
	}))
	defer srv.Close()

	c := New(srv.URL)

	t.Run("ChatCompletion", func(t *testing.T) {
		_, err := c.ChatCompletion(context.Background(), &api.ChatCompletionRequest{
			Model:    "test",
			Messages: []api.Message{{Role: "user", Content: "hi"}},
		})
		if err == nil {
			t.Fatal("expected error, got nil")
		}
		if got := err.Error(); !contains(got, "500") {
			t.Errorf("error should contain status code 500, got: %s", got)
		}
	})

	t.Run("ListModels", func(t *testing.T) {
		_, err := c.ListModels(context.Background())
		if err == nil {
			t.Fatal("expected error, got nil")
		}
		if got := err.Error(); !contains(got, "500") {
			t.Errorf("error should contain status code 500, got: %s", got)
		}
	})

	t.Run("MemorySearch", func(t *testing.T) {
		_, err := c.MemorySearch(context.Background(), "test", 5)
		if err == nil {
			t.Fatal("expected error, got nil")
		}
		if got := err.Error(); !contains(got, "500") {
			t.Errorf("error should contain status code 500, got: %s", got)
		}
	})

	t.Run("MemoryStore", func(t *testing.T) {
		_, err := c.MemoryStore(context.Background(), "a", "b")
		if err == nil {
			t.Fatal("expected error, got nil")
		}
		if got := err.Error(); !contains(got, "500") {
			t.Errorf("error should contain status code 500, got: %s", got)
		}
	})

	t.Run("Tokenize", func(t *testing.T) {
		_, err := c.Tokenize(context.Background(), "test")
		if err == nil {
			t.Fatal("expected error, got nil")
		}
		if got := err.Error(); !contains(got, "500") {
			t.Errorf("error should contain status code 500, got: %s", got)
		}
	})

	t.Run("MemoryDelete", func(t *testing.T) {
		err := c.MemoryDelete(context.Background(), "some-id")
		if err == nil {
			t.Fatal("expected error, got nil")
		}
		if got := err.Error(); !contains(got, "500") {
			t.Errorf("error should contain status code 500, got: %s", got)
		}
	})

	t.Run("MemoryClear", func(t *testing.T) {
		err := c.MemoryClear(context.Background())
		if err == nil {
			t.Fatal("expected error, got nil")
		}
		if got := err.Error(); !contains(got, "500") {
			t.Errorf("error should contain status code 500, got: %s", got)
		}
	})
}

func TestTimeout(t *testing.T) {
	c := New("http://localhost:8080")
	if c.httpClient.Timeout != 2*time.Minute {
		t.Errorf("httpClient timeout = %v, want %v", c.httpClient.Timeout, 2*time.Minute)
	}
	if c.streamClient.Timeout != 0 {
		t.Errorf("streamClient timeout = %v, want 0 (no timeout)", c.streamClient.Timeout)
	}
}

func TestMemoryList(t *testing.T) {
	want := &api.MemoryListResponse{
		Entries: []api.MemoryEntry{
			{ID: "mem-1", UserMsg: "hi", AssistMsg: "hello"},
			{ID: "mem-2", UserMsg: "bye", AssistMsg: "goodbye"},
		},
		Total: 2,
	}

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/memory/list" {
			t.Errorf("unexpected path: %s", r.URL.Path)
		}
		if r.Method != http.MethodGet {
			t.Errorf("unexpected method: %s", r.Method)
		}
		if r.URL.Query().Get("limit") != "10" {
			t.Errorf("limit = %q, want %q", r.URL.Query().Get("limit"), "10")
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(want)
	}))
	defer srv.Close()

	c := New(srv.URL)
	got, err := c.MemoryList(context.Background(), 10)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got.Total != 2 {
		t.Errorf("Total = %d, want %d", got.Total, 2)
	}
	if len(got.Entries) != 2 {
		t.Fatalf("expected 2 entries, got %d", len(got.Entries))
	}
}

func TestMemoryDelete(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/memory/mem-99" {
			t.Errorf("unexpected path: %s", r.URL.Path)
		}
		if r.Method != http.MethodDelete {
			t.Errorf("unexpected method: %s", r.Method)
		}
		w.WriteHeader(http.StatusOK)
	}))
	defer srv.Close()

	c := New(srv.URL)
	err := c.MemoryDelete(context.Background(), "mem-99")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestMemoryClear(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/memory" {
			t.Errorf("unexpected path: %s", r.URL.Path)
		}
		if r.Method != http.MethodDelete {
			t.Errorf("unexpected method: %s", r.Method)
		}
		w.WriteHeader(http.StatusOK)
	}))
	defer srv.Close()

	c := New(srv.URL)
	err := c.MemoryClear(context.Background())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestMemoryCount(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/memory/count" {
			t.Errorf("unexpected path: %s", r.URL.Path)
		}
		if r.Method != http.MethodGet {
			t.Errorf("unexpected method: %s", r.Method)
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(api.MemoryCountResponse{Count: 42})
	}))
	defer srv.Close()

	c := New(srv.URL)
	count, err := c.MemoryCount(context.Background())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if count != 42 {
		t.Errorf("count = %d, want %d", count, 42)
	}
}

func TestInstanceStatus(t *testing.T) {
	want := &api.InstanceStatus{
		Status: "running",
		GPUURL: "http://gpu:11435",
	}

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/instance/status" {
			t.Errorf("unexpected path: %s", r.URL.Path)
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(want)
	}))
	defer srv.Close()

	c := New(srv.URL)
	got, err := c.InstanceStatus(context.Background())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got.Status != "running" {
		t.Errorf("Status = %q, want %q", got.Status, "running")
	}
	if got.GPUURL != "http://gpu:11435" {
		t.Errorf("GPUURL = %q, want %q", got.GPUURL, "http://gpu:11435")
	}
}

func TestStreamCompletion(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/chat/completions" {
			t.Errorf("unexpected path: %s", r.URL.Path)
		}

		var req api.ChatCompletionRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Fatalf("decode request: %v", err)
		}
		if !req.Stream {
			t.Error("expected stream=true for streaming request")
		}

		w.Header().Set("Content-Type", "text/event-stream")
		flusher, _ := w.(http.Flusher)

		chunks := []string{
			`data: {"id":"c1","model":"m","choices":[{"index":0,"delta":{"role":"assistant","content":"Hello"}}]}`,
			`data: {"id":"c1","model":"m","choices":[{"index":0,"delta":{"content":" world"}}]}`,
			`data: [DONE]`,
		}
		for _, chunk := range chunks {
			w.Write([]byte(chunk + "\n\n"))
			flusher.Flush()
		}
	}))
	defer srv.Close()

	c := New(srv.URL)
	events, err := c.StreamCompletion(context.Background(), &api.ChatCompletionRequest{
		Model:    "m",
		Messages: []api.Message{{Role: "user", Content: "hi"}},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	var allContent string
	var doneReceived bool
	for ev := range events {
		if ev.Err != nil {
			t.Fatalf("unexpected stream error: %v", ev.Err)
		}
		if ev.Done {
			doneReceived = true

			continue
		}
		if ev.Chunk != nil {
			for _, choice := range ev.Chunk.Choices {
				allContent += choice.Delta.Content
			}
		}
	}
	if !doneReceived {
		t.Error("expected Done event")
	}
	if allContent != "Hello world" {
		t.Errorf("accumulated content = %q, want %q", allContent, "Hello world")
	}
}

func TestStreamCompletionServerError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusBadGateway)
		w.Write([]byte("bad gateway"))
	}))
	defer srv.Close()

	c := New(srv.URL)
	_, err := c.StreamCompletion(context.Background(), &api.ChatCompletionRequest{
		Model:    "m",
		Messages: []api.Message{{Role: "user", Content: "hi"}},
	})
	if err == nil {
		t.Fatal("expected error, got nil")
	}
	if !contains(err.Error(), "502") {
		t.Errorf("error should contain 502, got: %s", err.Error())
	}
}

func TestLoadModel(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/load" {
			t.Errorf("unexpected path: %s", r.URL.Path)
		}
		if r.Method != http.MethodPost {
			t.Errorf("unexpected method: %s", r.Method)
		}

		body, _ := io.ReadAll(r.Body)
		var req map[string]string
		json.Unmarshal(body, &req)
		if req["model"] != "test-model" {
			t.Errorf("model = %q, want %q", req["model"], "test-model")
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{"status": "loaded", "model": "test-model", "ctx_size": 4096})
	}))
	defer srv.Close()

	c := New(srv.URL)
	resp, err := c.LoadModel(context.Background(), "test-model")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if resp.CtxSize != 4096 {
		t.Errorf("ctx_size = %d, want 4096", resp.CtxSize)
	}
}

func TestCancelledContext(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	}))
	defer srv.Close()

	c := New(srv.URL)
	ctx, cancel := context.WithCancel(context.Background())
	cancel() // cancel immediately

	_, err := c.ChatCompletion(ctx, &api.ChatCompletionRequest{
		Model:    "test",
		Messages: []api.Message{{Role: "user", Content: "hi"}},
	})
	if err == nil {
		t.Fatal("expected error for cancelled context")
	}
}

func TestPullModel(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/pull" {
			t.Errorf("unexpected path: %s", r.URL.Path)
		}
		if r.Method != http.MethodPost {
			t.Errorf("unexpected method: %s", r.Method)
		}

		w.Header().Set("Content-Type", "text/event-stream")
		flusher, _ := w.(http.Flusher)

		events := []string{
			`data: {"status":"downloading","progress":50}`,
			`data: {"status":"downloaded","progress":100}`,
			`data: [DONE]`,
		}
		for _, ev := range events {
			w.Write([]byte(ev + "\n\n"))
			flusher.Flush()
		}
	}))
	defer srv.Close()

	c := New(srv.URL)
	ch, err := c.PullModel(context.Background(), "http://example.com/model.gguf")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	var events []api.PullEvent
	for ev := range ch {
		if ev.Err != nil {
			t.Fatalf("unexpected event error: %v", ev.Err)
		}
		events = append(events, ev.Event)
		if ev.Done {
			break
		}
	}
	// Drain channel
	for range ch {
	}

	if len(events) == 0 {
		t.Fatal("expected at least one event")
	}
	if events[0].Status != "downloading" {
		t.Errorf("first event status = %q, want %q", events[0].Status, "downloading")
	}
}

func TestPullModelError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
		w.Write([]byte("bad request"))
	}))
	defer srv.Close()

	c := New(srv.URL)
	_, err := c.PullModel(context.Background(), "http://example.com/model.gguf")
	if err == nil {
		t.Fatal("expected error, got nil")
	}
	if !contains(err.Error(), "400") {
		t.Errorf("error should contain 400, got: %s", err.Error())
	}
}

func TestPullModelInvalidJSON(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		flusher, _ := w.(http.Flusher)
		w.Write([]byte("data: {invalid json}\n\n"))
		flusher.Flush()
	}))
	defer srv.Close()

	c := New(srv.URL)
	ch, err := c.PullModel(context.Background(), "http://example.com/model.gguf")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	var gotErr error
	for ev := range ch {
		if ev.Err != nil {
			gotErr = ev.Err
		}
	}
	if gotErr == nil {
		t.Error("expected parse error from invalid JSON event")
	}
}

func TestInstanceStart(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/instance/start" {
			t.Errorf("unexpected path: %s", r.URL.Path)
		}
		if r.Method != http.MethodPost {
			t.Errorf("unexpected method: %s", r.Method)
		}
		w.WriteHeader(http.StatusOK)
	}))
	defer srv.Close()

	c := New(srv.URL)
	err := c.InstanceStart(context.Background())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestInstanceStartError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
		w.Write([]byte("server error"))
	}))
	defer srv.Close()

	c := New(srv.URL)
	err := c.InstanceStart(context.Background())
	if err == nil {
		t.Fatal("expected error, got nil")
	}
	if !contains(err.Error(), "500") {
		t.Errorf("error should contain 500, got: %s", err.Error())
	}
}

func TestInstanceStop(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/instance/stop" {
			t.Errorf("unexpected path: %s", r.URL.Path)
		}
		if r.Method != http.MethodPost {
			t.Errorf("unexpected method: %s", r.Method)
		}
		w.WriteHeader(http.StatusOK)
	}))
	defer srv.Close()

	c := New(srv.URL)
	err := c.InstanceStop(context.Background())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestInstanceStopError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusServiceUnavailable)
		w.Write([]byte("unavailable"))
	}))
	defer srv.Close()

	c := New(srv.URL)
	err := c.InstanceStop(context.Background())
	if err == nil {
		t.Fatal("expected error, got nil")
	}
	if !contains(err.Error(), "503") {
		t.Errorf("error should contain 503, got: %s", err.Error())
	}
}

func TestGetJSONError(t *testing.T) {
	// Test getJSON error path via MemoryList, MemoryCount, InstanceStatus, ListModels
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusForbidden)
		w.Write([]byte("forbidden"))
	}))
	defer srv.Close()

	c := New(srv.URL)

	t.Run("MemoryList", func(t *testing.T) {
		_, err := c.MemoryList(context.Background(), 5)
		if err == nil {
			t.Fatal("expected error")
		}
		if !contains(err.Error(), "403") {
			t.Errorf("expected 403 in error, got: %s", err.Error())
		}
	})

	t.Run("MemoryCount", func(t *testing.T) {
		_, err := c.MemoryCount(context.Background())
		if err == nil {
			t.Fatal("expected error")
		}
		if !contains(err.Error(), "403") {
			t.Errorf("expected 403 in error, got: %s", err.Error())
		}
	})

	t.Run("InstanceStatus", func(t *testing.T) {
		_, err := c.InstanceStatus(context.Background())
		if err == nil {
			t.Fatal("expected error")
		}
		if !contains(err.Error(), "403") {
			t.Errorf("expected 403 in error, got: %s", err.Error())
		}
	})

	t.Run("LoadModel", func(t *testing.T) {
		_, err := c.LoadModel(context.Background(), "mymodel")
		if err == nil {
			t.Fatal("expected error")
		}
		if !contains(err.Error(), "403") {
			t.Errorf("expected 403 in error, got: %s", err.Error())
		}
	})
}

func TestConnErrorCancelledContext(t *testing.T) {
	// Use a server that blocks so we can cancel the context
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// block until client disconnects
		<-r.Context().Done()
	}))
	defer srv.Close()

	c := New(srv.URL)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	_, err := c.MemoryList(ctx, 5)
	if err == nil {
		t.Fatal("expected error for cancelled context")
	}
}

func contains(s, substr string) bool {
	return len(s) >= len(substr) && searchSubstring(s, substr)
}

func searchSubstring(s, substr string) bool {
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return true
		}
	}

	return false
}
