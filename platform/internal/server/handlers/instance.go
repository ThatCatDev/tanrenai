package handlers

import (
	"net/http"

	"github.com/ThatCatDev/tanrenai/platform/internal/auth"
	"github.com/ThatCatDev/tanrenai/platform/internal/database"
)

// InstanceHandler handles instance management API endpoints.
type InstanceHandler struct {
	DB *database.DB
	// Manager will be added in Phase 4
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

// Provision triggers instance provisioning. Will be implemented in Phase 4.
func (h *InstanceHandler) Provision(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusNotImplemented, map[string]string{"error": "not_implemented", "message": "provisioning not yet implemented"})
}

// Destroy destroys the user's active instance. Will be fully implemented in Phase 4.
func (h *InstanceHandler) Destroy(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusNotImplemented, map[string]string{"error": "not_implemented", "message": "destroy not yet implemented"})
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
			"cost_per_hr":    0,
			"running_hours":  0,
			"total_cost":     0,
			"gpu_name":       "",
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
