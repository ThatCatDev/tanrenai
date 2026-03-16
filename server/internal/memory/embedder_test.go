package memory

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/ThatCatDev/tanrenai/server/internal/gpuclient"
	"github.com/ThatCatDev/tanrenai/server/pkg/api"
)

func TestNewRemoteEmbedFunc(t *testing.T) {
	wantVec := []float32{0.1, 0.2, 0.3, 0.4}

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/embeddings" {
			http.NotFound(w, r)

			return
		}
		resp := api.EmbeddingResponse{
			Data: []api.EmbeddingData{
				{Embedding: wantVec, Index: 0},
			},
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(resp)
	}))
	defer srv.Close()

	client := gpuclient.New(srv.URL)
	embedFunc := NewRemoteEmbedFunc(client)
	if embedFunc == nil {
		t.Fatal("NewRemoteEmbedFunc returned nil")
	}

	vec, err := embedFunc(context.Background(), "hello world")
	if err != nil {
		t.Fatalf("embedFunc returned error: %v", err)
	}
	if len(vec) == 0 {
		t.Fatal("embedFunc returned empty vector")
	}
}

func TestNewRemoteEmbedFuncError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
	}))
	defer srv.Close()

	client := gpuclient.New(srv.URL)
	embedFunc := NewRemoteEmbedFunc(client)

	_, err := embedFunc(context.Background(), "hello")
	if err == nil {
		t.Fatal("embedFunc expected error on server error, got nil")
	}
}

func TestNewRemoteEmbedFuncEmptyData(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		resp := api.EmbeddingResponse{
			Data: []api.EmbeddingData{},
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(resp)
	}))
	defer srv.Close()

	client := gpuclient.New(srv.URL)
	embedFunc := NewRemoteEmbedFunc(client)

	_, err := embedFunc(context.Background(), "hello")
	if err == nil {
		t.Fatal("embedFunc expected error for empty data, got nil")
	}
}
