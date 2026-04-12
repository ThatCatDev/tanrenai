package server

import (
	"encoding/json"
	"net/http"

	"github.com/ThatCatDev/tanrenai/platform/internal/server/handlers"
)

func (s *Server) registerRoutes(mux *http.ServeMux) {
	// Public
	mux.HandleFunc("GET /health", s.handleHealth)

	userHandler := &handlers.UserHandler{
		DB:            s.db,
		EncryptionKey: s.cfg.EncryptionKey,
	}
	instHandler := &handlers.InstanceHandler{
		DB: s.db,
	}

	if s.auth != nil {
		// Authenticated routes
		mux.Handle("GET /api/user/me", s.auth.WrapFunc(userHandler.Me))
		mux.Handle("PUT /api/user/settings", s.auth.WrapFunc(userHandler.UpdateSettings))
		mux.Handle("POST /api/user/vastai-key", s.auth.WrapFunc(userHandler.SetVastaiKey))
		mux.Handle("DELETE /api/user/vastai-key", s.auth.WrapFunc(userHandler.DeleteVastaiKey))

		mux.Handle("GET /api/instance/status", s.auth.WrapFunc(instHandler.Status))
		mux.Handle("POST /api/instance/provision", s.auth.WrapFunc(instHandler.Provision))
		mux.Handle("POST /api/instance/destroy", s.auth.WrapFunc(instHandler.Destroy))
		mux.Handle("GET /api/instance/cost", s.auth.WrapFunc(instHandler.Cost))
	} else {
		// No OIDC configured — register routes without auth (development mode)
		mux.HandleFunc("GET /api/user/me", userHandler.Me)
		mux.HandleFunc("PUT /api/user/settings", userHandler.UpdateSettings)
		mux.HandleFunc("POST /api/user/vastai-key", userHandler.SetVastaiKey)
		mux.HandleFunc("DELETE /api/user/vastai-key", userHandler.DeleteVastaiKey)

		mux.HandleFunc("GET /api/instance/status", instHandler.Status)
		mux.HandleFunc("POST /api/instance/provision", instHandler.Provision)
		mux.HandleFunc("POST /api/instance/destroy", instHandler.Destroy)
		mux.HandleFunc("GET /api/instance/cost", instHandler.Cost)
	}
}

func (s *Server) handleHealth(w http.ResponseWriter, _ *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
}
