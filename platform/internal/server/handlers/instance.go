package handlers

import (
	"encoding/json"
	"net/http"

	"github.com/ThatCatDev/tanrenai/platform/internal/auth"
	"github.com/ThatCatDev/tanrenai/platform/internal/database"
	"github.com/ThatCatDev/tanrenai/platform/internal/instance"
)

// InstanceHandler handles instance management API endpoints.
type InstanceHandler struct {
	DB      *database.DB
	Manager *instance.Manager
}

// Status returns the current user's active instance status.
func (h *InstanceHandler) Status(w http.ResponseWriter, r *http.Request) {
	user := auth.UserFromContext(r.Context())
	if user == nil {
		writeJSON(w, http.StatusUnauthorized, map[string]string{"error": "not authenticated"})
		return
	}

	inst, err := h.DB.GetActiveInstance(r.Context(), user.ID)
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]string{"error": "failed to get instance"})
		return
	}

	if inst == nil {
		writeJSON(w, http.StatusOK, map[string]any{
			"status":   "none",
			"provider": "vastai",
		})
		return
	}

	writeJSON(w, http.StatusOK, map[string]any{
		"status":          inst.Status,
		"provision_state": inst.ProvisionState,
		"gpu_name":        inst.GPUName,
		"gpu_url":         inst.GPUURL,
		"cost_per_hr":     inst.CostPerHr,
		"model_loaded":    inst.ModelLoaded,
		"created_at":      inst.CreatedAt,
		"last_activity":   inst.LastActivity,
	})
}

// Provision triggers instance provisioning for a given model size.
func (h *InstanceHandler) Provision(w http.ResponseWriter, r *http.Request) {
	user := auth.UserFromContext(r.Context())
	if user == nil {
		writeJSON(w, http.StatusUnauthorized, map[string]string{"error": "not authenticated"})
		return
	}

	var req struct {
		ModelSize    string  `json:"model_size"`
		MaxCostPerHr float64 `json:"max_cost_per_hr,omitempty"`
		GPUName      string  `json:"gpu_name,omitempty"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil || req.ModelSize == "" {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "model_size is required"})
		return
	}

	// Override user settings if provided in request
	if req.MaxCostPerHr > 0 {
		user.MaxCostPerHr = req.MaxCostPerHr
	}
	if req.GPUName != "" {
		user.PreferredGPU = req.GPUName
	}

	gpuURL, err := h.Manager.EnsureRunning(r.Context(), user, req.ModelSize)
	if err != nil {
		writeJSON(w, http.StatusServiceUnavailable, map[string]string{
			"error":   "provision_failed",
			"message": err.Error(),
		})
		return
	}

	writeJSON(w, http.StatusOK, map[string]any{
		"status":  "running",
		"gpu_url": gpuURL,
	})
}

// Destroy destroys the user's active instance.
func (h *InstanceHandler) Destroy(w http.ResponseWriter, r *http.Request) {
	user := auth.UserFromContext(r.Context())
	if user == nil {
		writeJSON(w, http.StatusUnauthorized, map[string]string{"error": "not authenticated"})
		return
	}

	if err := h.Manager.Destroy(r.Context(), user); err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]string{
			"error":   "destroy_failed",
			"message": err.Error(),
		})
		return
	}

	writeJSON(w, http.StatusOK, map[string]string{"status": "destroyed"})
}

// Cost returns cost info for the user's active instance.
func (h *InstanceHandler) Cost(w http.ResponseWriter, r *http.Request) {
	user := auth.UserFromContext(r.Context())
	if user == nil {
		writeJSON(w, http.StatusUnauthorized, map[string]string{"error": "not authenticated"})
		return
	}

	inst, err := h.DB.GetActiveInstance(r.Context(), user.ID)
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]string{"error": "failed to get instance"})
		return
	}

	if inst == nil {
		writeJSON(w, http.StatusOK, map[string]any{
			"cost_per_hr":   0,
			"running_hours": 0,
			"total_cost":    0,
			"gpu_name":      "",
		})
		return
	}

	runningHours := inst.LastActivity.Sub(inst.CreatedAt).Hours()
	if runningHours < 0 {
		runningHours = 0
	}

	writeJSON(w, http.StatusOK, map[string]any{
		"cost_per_hr":   inst.CostPerHr,
		"running_hours": runningHours,
		"total_cost":    inst.CostPerHr * runningHours,
		"gpu_name":      inst.GPUName,
	})
}
