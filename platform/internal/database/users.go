package database

import (
	"context"
	"fmt"
	"time"
)

// User represents a platform user.
type User struct {
	ID              string
	Email           string
	Name            string
	VastaiKeyEnc    []byte  // encrypted vast.ai API key
	IdleTimeoutMin  int
	MaxCostPerHr    float64
	PreferredGPU    string
	CreatedAt       time.Time
	UpdatedAt       time.Time
}

// CreateOrUpdateUser creates a user if they don't exist, or updates their name/email.
// Called on every authenticated request to keep user info in sync with OIDC.
func (db *DB) CreateOrUpdateUser(ctx context.Context, id, email, name string) (*User, error) {
	var u User
	err := db.Pool.QueryRow(ctx, `
		INSERT INTO users (id, email, name)
		VALUES ($1, $2, $3)
		ON CONFLICT (id) DO UPDATE SET
			email = EXCLUDED.email,
			name = EXCLUDED.name,
			updated_at = NOW()
		RETURNING id, email, name, vastai_api_key_enc, idle_timeout_min, max_cost_per_hr, preferred_gpu, created_at, updated_at
	`, id, email, name).Scan(
		&u.ID, &u.Email, &u.Name, &u.VastaiKeyEnc,
		&u.IdleTimeoutMin, &u.MaxCostPerHr, &u.PreferredGPU,
		&u.CreatedAt, &u.UpdatedAt,
	)
	if err != nil {
		return nil, fmt.Errorf("create or update user: %w", err)
	}
	return &u, nil
}

// GetUser returns a user by ID.
func (db *DB) GetUser(ctx context.Context, id string) (*User, error) {
	var u User
	err := db.Pool.QueryRow(ctx, `
		SELECT id, email, name, vastai_api_key_enc, idle_timeout_min, max_cost_per_hr, preferred_gpu, created_at, updated_at
		FROM users WHERE id = $1
	`, id).Scan(
		&u.ID, &u.Email, &u.Name, &u.VastaiKeyEnc,
		&u.IdleTimeoutMin, &u.MaxCostPerHr, &u.PreferredGPU,
		&u.CreatedAt, &u.UpdatedAt,
	)
	if err != nil {
		return nil, fmt.Errorf("get user: %w", err)
	}
	return &u, nil
}

// SetVastaiKey stores an encrypted vast.ai API key for a user.
func (db *DB) SetVastaiKey(ctx context.Context, userID string, encryptedKey []byte) error {
	_, err := db.Pool.Exec(ctx, `
		UPDATE users SET vastai_api_key_enc = $1, updated_at = NOW() WHERE id = $2
	`, encryptedKey, userID)
	if err != nil {
		return fmt.Errorf("set vastai key: %w", err)
	}
	return nil
}

// DeleteVastaiKey removes the stored vast.ai API key for a user.
func (db *DB) DeleteVastaiKey(ctx context.Context, userID string) error {
	_, err := db.Pool.Exec(ctx, `
		UPDATE users SET vastai_api_key_enc = NULL, updated_at = NOW() WHERE id = $1
	`, userID)
	if err != nil {
		return fmt.Errorf("delete vastai key: %w", err)
	}
	return nil
}

// UpdateUserSettings updates a user's configurable settings.
func (db *DB) UpdateUserSettings(ctx context.Context, userID string, idleMin int, maxCost float64, gpu string) error {
	_, err := db.Pool.Exec(ctx, `
		UPDATE users SET idle_timeout_min = $1, max_cost_per_hr = $2, preferred_gpu = $3, updated_at = NOW()
		WHERE id = $4
	`, idleMin, maxCost, gpu, userID)
	if err != nil {
		return fmt.Errorf("update user settings: %w", err)
	}
	return nil
}
