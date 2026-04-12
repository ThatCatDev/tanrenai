package database

import (
	"context"
	"fmt"
	"time"

	"github.com/jackc/pgx/v5"
)

// Instance represents a GPU instance managed by the platform.
type Instance struct {
	ID              string
	UserID          string
	VastaiInstanceID int
	Status          string // pending, provisioning, running, destroying, destroyed
	GPUName         string
	GPUURL          string
	SSHHost         string
	SSHPort         int
	CostPerHr       float64
	ModelLoaded     string
	ProvisionState  string // searching, creating, booting, ready
	CreatedAt       time.Time
	LastActivity    time.Time
	DestroyedAt     *time.Time
}

// CreateInstance creates a new instance record.
func (db *DB) CreateInstance(ctx context.Context, inst *Instance) error {
	err := db.Pool.QueryRow(ctx, `
		INSERT INTO instances (user_id, vastai_instance_id, status, gpu_name, gpu_url, ssh_host, ssh_port, cost_per_hr, model_loaded, provision_state)
		VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
		RETURNING id, created_at, last_activity
	`, inst.UserID, inst.VastaiInstanceID, inst.Status, inst.GPUName, inst.GPUURL,
		inst.SSHHost, inst.SSHPort, inst.CostPerHr, inst.ModelLoaded, inst.ProvisionState,
	).Scan(&inst.ID, &inst.CreatedAt, &inst.LastActivity)
	if err != nil {
		return fmt.Errorf("create instance: %w", err)
	}
	return nil
}

// GetActiveInstance returns the user's active (non-destroyed) instance, if any.
func (db *DB) GetActiveInstance(ctx context.Context, userID string) (*Instance, error) {
	var inst Instance
	err := db.Pool.QueryRow(ctx, `
		SELECT id, user_id, vastai_instance_id, status, gpu_name, gpu_url, ssh_host, ssh_port,
		       cost_per_hr, model_loaded, provision_state, created_at, last_activity, destroyed_at
		FROM instances
		WHERE user_id = $1 AND status NOT IN ('destroyed')
		ORDER BY created_at DESC
		LIMIT 1
	`, userID).Scan(
		&inst.ID, &inst.UserID, &inst.VastaiInstanceID, &inst.Status,
		&inst.GPUName, &inst.GPUURL, &inst.SSHHost, &inst.SSHPort,
		&inst.CostPerHr, &inst.ModelLoaded, &inst.ProvisionState,
		&inst.CreatedAt, &inst.LastActivity, &inst.DestroyedAt,
	)
	if err == pgx.ErrNoRows {
		return nil, nil
	}
	if err != nil {
		return nil, fmt.Errorf("get active instance: %w", err)
	}
	return &inst, nil
}

// UpdateInstanceStatus updates the status and provision_state of an instance.
func (db *DB) UpdateInstanceStatus(ctx context.Context, id, status, provisionState string) error {
	_, err := db.Pool.Exec(ctx, `
		UPDATE instances SET status = $1, provision_state = $2 WHERE id = $3
	`, status, provisionState, id)
	if err != nil {
		return fmt.Errorf("update instance status: %w", err)
	}
	return nil
}

// UpdateInstanceGPU updates the GPU details after provisioning.
func (db *DB) UpdateInstanceGPU(ctx context.Context, id string, vastaiID int, gpuName, gpuURL string, costPerHr float64) error {
	_, err := db.Pool.Exec(ctx, `
		UPDATE instances SET vastai_instance_id = $1, gpu_name = $2, gpu_url = $3, cost_per_hr = $4
		WHERE id = $5
	`, vastaiID, gpuName, gpuURL, costPerHr, id)
	if err != nil {
		return fmt.Errorf("update instance GPU: %w", err)
	}
	return nil
}

// UpdateInstanceActivity updates the last_activity timestamp.
func (db *DB) UpdateInstanceActivity(ctx context.Context, id string) error {
	_, err := db.Pool.Exec(ctx, `
		UPDATE instances SET last_activity = NOW() WHERE id = $1
	`, id)
	if err != nil {
		return fmt.Errorf("update instance activity: %w", err)
	}
	return nil
}

// DestroyInstance marks an instance as destroyed.
func (db *DB) DestroyInstance(ctx context.Context, id string) error {
	_, err := db.Pool.Exec(ctx, `
		UPDATE instances SET status = 'destroyed', destroyed_at = NOW() WHERE id = $1
	`, id)
	if err != nil {
		return fmt.Errorf("destroy instance: %w", err)
	}
	return nil
}
