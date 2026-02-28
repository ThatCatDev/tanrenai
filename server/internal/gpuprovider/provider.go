package gpuprovider

import (
	"context"
	"time"
)

// Provider abstracts GPU backend lifecycle management.
// Implementations include LocalProvider (static GPU URL) and VastAIProvider (vast.ai instance).
type Provider interface {
	Name() string
	EnsureRunning(ctx context.Context) error
	RecordActivity()
	Status(ctx context.Context) (*Status, error)
	Stop(ctx context.Context) error
	StartIdleTimer()
	Close()
}

// Status represents the current GPU provider status.
type Status struct {
	State    string     `json:"status"`
	Provider string     `json:"provider"`
	GPUURL   string     `json:"gpu_url,omitempty"`
	IdleSince *time.Time `json:"idle_since,omitempty"`
}
