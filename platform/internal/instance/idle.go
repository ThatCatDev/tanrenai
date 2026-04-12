package instance

import (
	"context"
	"log/slog"
	"sync"
	"time"

	"github.com/ThatCatDev/tanrenai/platform/internal/database"
)

// IdleTimer monitors a user's instance and destroys it after the configured idle timeout.
type IdleTimer struct {
	userID      string
	db          *database.DB
	provisioner *Provisioner
	stopCh      chan struct{}
	mu          sync.Mutex
}

// NewIdleTimer creates a new idle timer for a user.
func NewIdleTimer(userID string, db *database.DB, provisioner *Provisioner) *IdleTimer {
	return &IdleTimer{
		userID:      userID,
		db:          db,
		provisioner: provisioner,
	}
}

// Start begins monitoring the user's instance for idle timeout.
func (t *IdleTimer) Start() {
	t.mu.Lock()
	if t.stopCh != nil {
		close(t.stopCh)
	}
	t.stopCh = make(chan struct{})
	stopCh := t.stopCh
	t.mu.Unlock()

	go func() {
		ticker := time.NewTicker(1 * time.Minute)
		defer ticker.Stop()

		for {
			select {
			case <-stopCh:
				return
			case <-ticker.C:
				t.check()
			}
		}
	}()
}

// Stop stops the idle timer.
func (t *IdleTimer) Stop() {
	t.mu.Lock()
	defer t.mu.Unlock()
	if t.stopCh != nil {
		close(t.stopCh)
		t.stopCh = nil
	}
}

func (t *IdleTimer) check() {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	user, err := t.db.GetUser(ctx, t.userID)
	if err != nil {
		slog.Error("idle check: get user", "error", err, "user_id", t.userID)
		return
	}

	inst, err := t.db.GetActiveInstance(ctx, t.userID)
	if err != nil || inst == nil {
		return
	}

	if inst.Status != "running" {
		return
	}

	idleTimeout := time.Duration(user.IdleTimeoutMin) * time.Minute
	idle := time.Since(inst.LastActivity)

	if idle < idleTimeout {
		return
	}

	slog.Info("instance idle, destroying",
		"user", user.Email,
		"idle", idle.Round(time.Second),
		"timeout", idleTimeout)

	if err := t.provisioner.Destroy(ctx, user, inst); err != nil {
		slog.Error("idle destroy failed", "error", err, "user_id", t.userID)
	}

	// Stop this timer — it will be restarted when the user provisions again
	t.Stop()
}
