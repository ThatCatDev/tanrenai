package handlers

import (
	"encoding/json"
	"net/http"

	"github.com/ThatCatDev/tanrenai/platform/internal/auth"
	"github.com/ThatCatDev/tanrenai/platform/internal/database"
)

// UserHandler handles user-related API endpoints.
type UserHandler struct {
	DB            *database.DB
	EncryptionKey string
}

// Me returns the current user's profile.
func (h *UserHandler) Me(w http.ResponseWriter, r *http.Request) {
	user := auth.UserFromContext(r.Context())
	if user == nil {
		writeJSON(w, http.StatusUnauthorized, map[string]string{"error": "not authenticated"})
		return
	}

	writeJSON(w, http.StatusOK, map[string]any{
		"id":               user.ID,
		"email":            user.Email,
		"name":             user.Name,
		"has_vastai_key":   user.VastaiKeyEnc != nil,
		"idle_timeout_min": user.IdleTimeoutMin,
		"max_cost_per_hr":  user.MaxCostPerHr,
		"preferred_gpu":    user.PreferredGPU,
	})
}

// UpdateSettings updates the user's configurable settings.
func (h *UserHandler) UpdateSettings(w http.ResponseWriter, r *http.Request) {
	user := auth.UserFromContext(r.Context())
	if user == nil {
		writeJSON(w, http.StatusUnauthorized, map[string]string{"error": "not authenticated"})
		return
	}

	var req struct {
		IdleTimeoutMin int     `json:"idle_timeout_min"`
		MaxCostPerHr   float64 `json:"max_cost_per_hr"`
		PreferredGPU   string  `json:"preferred_gpu"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "invalid request body"})
		return
	}

	if req.IdleTimeoutMin < 15 {
		req.IdleTimeoutMin = 15
	}
	if req.MaxCostPerHr <= 0 {
		req.MaxCostPerHr = 1.0
	}

	if err := h.DB.UpdateUserSettings(r.Context(), user.ID, req.IdleTimeoutMin, req.MaxCostPerHr, req.PreferredGPU); err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]string{"error": "failed to update settings"})
		return
	}

	writeJSON(w, http.StatusOK, map[string]string{"status": "ok"})
}

// SetVastaiKey stores an encrypted vast.ai API key for the user.
func (h *UserHandler) SetVastaiKey(w http.ResponseWriter, r *http.Request) {
	user := auth.UserFromContext(r.Context())
	if user == nil {
		writeJSON(w, http.StatusUnauthorized, map[string]string{"error": "not authenticated"})
		return
	}

	var req struct {
		APIKey string `json:"api_key"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil || req.APIKey == "" {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "api_key is required"})
		return
	}

	encrypted, err := database.Encrypt([]byte(req.APIKey), h.EncryptionKey)
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]string{"error": "encryption failed"})
		return
	}

	if err := h.DB.SetVastaiKey(r.Context(), user.ID, encrypted); err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]string{"error": "failed to store key"})
		return
	}

	writeJSON(w, http.StatusOK, map[string]string{"status": "ok"})
}

// DeleteVastaiKey removes the user's stored vast.ai API key.
func (h *UserHandler) DeleteVastaiKey(w http.ResponseWriter, r *http.Request) {
	user := auth.UserFromContext(r.Context())
	if user == nil {
		writeJSON(w, http.StatusUnauthorized, map[string]string{"error": "not authenticated"})
		return
	}

	if err := h.DB.DeleteVastaiKey(r.Context(), user.ID); err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]string{"error": "failed to delete key"})
		return
	}

	writeJSON(w, http.StatusOK, map[string]string{"status": "ok"})
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}
