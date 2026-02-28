package handlers

import (
	"encoding/json"
	"net/http"

	"github.com/ThatCatDev/tanrenai/server/internal/gpuprovider"
)

// InstanceHandler handles GPU instance management endpoints.
type InstanceHandler struct {
	Provider gpuprovider.Provider
}

// Status handles GET /api/instance/status.
func (h *InstanceHandler) Status(w http.ResponseWriter, r *http.Request) {
	status, err := h.Provider.Status(r.Context())
	if err != nil {
		writeError(w, http.StatusInternalServerError, "instance_error", err.Error())
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(status)
}

// Start handles POST /api/instance/start.
func (h *InstanceHandler) Start(w http.ResponseWriter, r *http.Request) {
	if err := h.Provider.EnsureRunning(r.Context()); err != nil {
		writeError(w, http.StatusInternalServerError, "instance_error", err.Error())
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"status": "starting"})
}

// Stop handles POST /api/instance/stop.
func (h *InstanceHandler) Stop(w http.ResponseWriter, r *http.Request) {
	if err := h.Provider.Stop(r.Context()); err != nil {
		writeError(w, http.StatusInternalServerError, "instance_error", err.Error())
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"status": "stopped"})
}
