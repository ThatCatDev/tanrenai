package server

import (
	"log"
	"net/http"

	"github.com/ThatCatDev/tanrenai/server/internal/server/handlers"
)

func (s *Server) registerRoutes(mux *http.ServeMux) {
	// Health
	mux.HandleFunc("GET /health", handlers.Health)

	// Proxy to GPU server: chat completions
	proxy := &handlers.ProxyHandler{
		GPUClient: s.gpuClient,
		Provider:  s.provider,
	}
	mux.HandleFunc("POST /v1/chat/completions", proxy.ChatCompletions)
	mux.HandleFunc("POST /tokenize", proxy.Tokenize)
	mux.HandleFunc("GET /v1/models", proxy.ListModels)
	mux.HandleFunc("POST /api/load", proxy.LoadModel)
	mux.HandleFunc("POST /api/pull", proxy.PullModel)

	// Finetune proxy to GPU server
	mux.HandleFunc("POST /v1/finetune/prepare", proxy.RawProxy)
	mux.HandleFunc("POST /v1/finetune/train", proxy.RawProxy)
	mux.HandleFunc("GET /v1/finetune/status/", proxy.RawProxy)
	mux.HandleFunc("POST /v1/finetune/merge", proxy.RawProxy)
	mux.HandleFunc("GET /v1/finetune/runs", proxy.RawProxy)
	mux.HandleFunc("DELETE /v1/finetune/runs/", proxy.RawProxy)

	// Memory endpoints (only active if memory store is set)
	if s.memStore != nil {
		mem := &handlers.MemoryHandler{MemStore: s.memStore}
		mux.HandleFunc("POST /v1/memory/search", mem.Search)
		mux.HandleFunc("POST /v1/memory/store", mem.Store)
		mux.HandleFunc("GET /v1/memory/list", mem.List)
		mux.HandleFunc("DELETE /v1/memory/{id}", mem.Delete)
		mux.HandleFunc("DELETE /v1/memory", mem.Clear)
		mux.HandleFunc("GET /v1/memory/count", mem.Count)
	}

	// Instance management (always registered — provider handles local vs vastai)
	inst := &handlers.InstanceHandler{Provider: s.provider}
	mux.HandleFunc("GET /api/instance/status", inst.Status)
	mux.HandleFunc("POST /api/instance/start", inst.Start)
	mux.HandleFunc("POST /api/instance/stop", inst.Stop)
}

func withLogging(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		log.Printf("%s %s", r.Method, r.URL.Path)
		next.ServeHTTP(w, r)
	})
}

func withCORS(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization")
		if r.Method == http.MethodOptions {
			w.WriteHeader(http.StatusOK)
			return
		}
		next.ServeHTTP(w, r)
	})
}
