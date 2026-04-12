package server

import (
	"encoding/json"
	"net/http"
)

func (s *Server) registerRoutes(mux *http.ServeMux) {
	// Public
	mux.HandleFunc("GET /health", s.handleHealth)

	// TODO: Add OIDC auth middleware and protected routes in Phase 2
	// For now, register placeholder routes

	// User endpoints (will require auth)
	mux.HandleFunc("GET /api/user/me", s.handleNotImplemented)
	mux.HandleFunc("PUT /api/user/settings", s.handleNotImplemented)
	mux.HandleFunc("POST /api/user/vastai-key", s.handleNotImplemented)
	mux.HandleFunc("DELETE /api/user/vastai-key", s.handleNotImplemented)

	// Instance endpoints (will require auth)
	mux.HandleFunc("GET /api/instance/status", s.handleNotImplemented)
	mux.HandleFunc("POST /api/instance/provision", s.handleNotImplemented)
	mux.HandleFunc("POST /api/instance/destroy", s.handleNotImplemented)
	mux.HandleFunc("GET /api/instance/cost", s.handleNotImplemented)
}

func (s *Server) handleHealth(w http.ResponseWriter, _ *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
}

func (s *Server) handleNotImplemented(w http.ResponseWriter, _ *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusNotImplemented)
	_ = json.NewEncoder(w).Encode(map[string]string{"error": "not_implemented", "message": "this endpoint is not yet implemented"})
}
