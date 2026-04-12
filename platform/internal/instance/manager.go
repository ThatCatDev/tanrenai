package instance

import (
	"context"
	"fmt"
	"log/slog"
	"net/http"
	"sync"
	"time"

	"github.com/ThatCatDev/tanrenai/platform/internal/database"
)

// Manager manages per-user GPU instance lifecycles.
type Manager struct {
	db          *database.DB
	provisioner *Provisioner
	timers      map[string]*IdleTimer
	mu          sync.RWMutex
}

// NewManager creates a new instance manager.
func NewManager(db *database.DB, provisioner *Provisioner) *Manager {
	return &Manager{
		db:          db,
		provisioner: provisioner,
		timers:      make(map[string]*IdleTimer),
	}
}

// EnsureRunning checks if the user has a running GPU instance, provisioning one if needed.
// Returns the GPU URL for the running instance.
func (m *Manager) EnsureRunning(ctx context.Context, user *database.User, modelSize string) (string, error) {
	// Check for existing active instance
	inst, err := m.db.GetActiveInstance(ctx, user.ID)
	if err != nil {
		return "", fmt.Errorf("check active instance: %w", err)
	}

	if inst != nil && inst.Status == "running" && inst.GPUURL != "" {
		// Health check existing instance
		if m.healthCheck(ctx, inst.GPUURL) {
			return inst.GPUURL, nil
		}
		slog.Warn("active instance unhealthy", "user", user.Email, "gpu_url", inst.GPUURL)
	}

	if inst != nil && inst.Status == "provisioning" {
		// Already provisioning — wait for it or return current state
		if inst.GPUURL != "" && m.healthCheck(ctx, inst.GPUURL) {
			_ = m.db.UpdateInstanceStatus(ctx, inst.ID, "running", "ready")
			return inst.GPUURL, nil
		}
		return "", fmt.Errorf("instance is still provisioning (state: %s)", inst.ProvisionState)
	}

	// Need to provision
	if modelSize == "" {
		return "", fmt.Errorf("no running instance and no model specified for provisioning")
	}

	inst, err = m.provisioner.Provision(ctx, user, modelSize)
	if err != nil {
		return "", err
	}

	// Start idle timer for this user
	m.startIdleTimer(user.ID)

	return inst.GPUURL, nil
}

// RecordActivity updates the last activity timestamp for a user's instance.
func (m *Manager) RecordActivity(ctx context.Context, userID string) {
	inst, err := m.db.GetActiveInstance(ctx, userID)
	if err != nil || inst == nil {
		return
	}
	_ = m.db.UpdateInstanceActivity(ctx, inst.ID)
}

// GetStatus returns the status of a user's active instance.
func (m *Manager) GetStatus(ctx context.Context, userID string) (*database.Instance, error) {
	return m.db.GetActiveInstance(ctx, userID)
}

// Destroy destroys a user's active instance.
func (m *Manager) Destroy(ctx context.Context, user *database.User) error {
	inst, err := m.db.GetActiveInstance(ctx, user.ID)
	if err != nil {
		return err
	}
	if inst == nil {
		return fmt.Errorf("no active instance")
	}

	// Stop idle timer
	m.stopIdleTimer(user.ID)

	return m.provisioner.Destroy(ctx, user, inst)
}

// Close stops all idle timers.
func (m *Manager) Close() {
	m.mu.Lock()
	defer m.mu.Unlock()
	for _, timer := range m.timers {
		timer.Stop()
	}
	m.timers = make(map[string]*IdleTimer)
}

func (m *Manager) startIdleTimer(userID string) {
	m.mu.Lock()
	defer m.mu.Unlock()

	if existing, ok := m.timers[userID]; ok {
		existing.Stop()
	}

	timer := NewIdleTimer(userID, m.db, m.provisioner)
	timer.Start()
	m.timers[userID] = timer
}

func (m *Manager) stopIdleTimer(userID string) {
	m.mu.Lock()
	defer m.mu.Unlock()

	if timer, ok := m.timers[userID]; ok {
		timer.Stop()
		delete(m.timers, userID)
	}
}

func (m *Manager) healthCheck(ctx context.Context, gpuURL string) bool {
	hctx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()

	req, err := http.NewRequestWithContext(hctx, http.MethodGet, gpuURL+"/health", nil)
	if err != nil {
		return false
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return false
	}
	_ = resp.Body.Close()
	return resp.StatusCode == http.StatusOK
}
