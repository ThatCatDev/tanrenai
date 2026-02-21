package handlers

import (
	"context"
	"encoding/json"
	"net/http"

	"github.com/thatcatdev/tanrenai/server/internal/models"
	"github.com/thatcatdev/tanrenai/server/pkg/api"
)

// ModelsHandler handles GET /v1/models.
type ModelsHandler struct {
	Store *models.Store
}

func (h *ModelsHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	available := h.Store.List()

	data := make([]api.ModelInfo, 0, len(available))
	for _, m := range available {
		data = append(data, api.ModelInfo{
			ID:      m.Name,
			Object:  "model",
			Created: m.ModifiedAt,
			OwnedBy: "local",
		})
	}

	resp := api.ModelListResponse{
		Object: "list",
		Data:   data,
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

// LoadHandler handles POST /api/load.
type LoadHandler struct {
	LoadFunc func(ctx context.Context, model string) error
}

func (h *LoadHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Model string `json:"model"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid_request", "failed to parse request body")
		return
	}

	if req.Model == "" {
		writeError(w, http.StatusBadRequest, "invalid_request", "model field is required")
		return
	}

	if err := h.LoadFunc(r.Context(), req.Model); err != nil {
		writeError(w, http.StatusInternalServerError, "model_error", err.Error())
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"status": "loaded", "model": req.Model})
}

func writeError(w http.ResponseWriter, status int, errType, message string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(api.ErrorResponse{
		Error: api.ErrorDetail{
			Message: message,
			Type:    errType,
		},
	})
}
