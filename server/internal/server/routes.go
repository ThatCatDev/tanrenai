package server

import (
	"log"
	"net/http"

	"github.com/thatcatdev/tanrenai/server/internal/runner"
	"github.com/thatcatdev/tanrenai/server/internal/server/handlers"
)

func (s *Server) registerRoutes(mux *http.ServeMux) {
	mux.HandleFunc("GET /health", handlers.Health)
	mux.HandleFunc("GET /v1/models", s.handleModels)
	mux.HandleFunc("POST /v1/chat/completions", s.handleChatCompletions)
	mux.HandleFunc("POST /v1/agent/completions", s.handleAgentCompletions)
	mux.HandleFunc("POST /api/load", s.handleLoadModel)
}

func (s *Server) handleModels(w http.ResponseWriter, r *http.Request) {
	h := &handlers.ModelsHandler{Store: s.store}
	h.ServeHTTP(w, r)
}

func (s *Server) handleChatCompletions(w http.ResponseWriter, r *http.Request) {
	h := &handlers.ChatHandler{
		GetRunner: func() runner.Runner { return s.runner },
		LoadFunc:  s.LoadModel,
	}
	h.ServeHTTP(w, r)
}

func (s *Server) handleAgentCompletions(w http.ResponseWriter, r *http.Request) {
	h := &handlers.AgentHandler{
		GetRunner: func() runner.Runner { return s.runner },
		LoadFunc:  s.LoadModel,
		Registry:  s.toolRegistry,
	}
	h.ServeHTTP(w, r)
}

func (s *Server) handleLoadModel(w http.ResponseWriter, r *http.Request) {
	h := &handlers.LoadHandler{LoadFunc: s.LoadModel}
	h.ServeHTTP(w, r)
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
		w.Header().Set("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization")
		if r.Method == http.MethodOptions {
			w.WriteHeader(http.StatusOK)
			return
		}
		next.ServeHTTP(w, r)
	})
}
