package gpuprovider

import (
	"context"

	"github.com/ThatCatDev/tanrenai/server/internal/gpuclient"
)

// LocalProvider is a GPU provider for a locally-reachable GPU server.
// EnsureRunning pings health; Stop and idle timer are no-ops.
type LocalProvider struct {
	gpuClient *gpuclient.Client
}

// NewLocalProvider creates a provider that points at a static GPU URL.
func NewLocalProvider(gpuClient *gpuclient.Client) *LocalProvider {
	return &LocalProvider{gpuClient: gpuClient}
}

func (p *LocalProvider) Name() string { return "local" }

func (p *LocalProvider) EnsureRunning(ctx context.Context) error {
	return p.gpuClient.Health(ctx)
}

func (p *LocalProvider) RecordActivity() {}

func (p *LocalProvider) Status(ctx context.Context) (*Status, error) {
	state := "stopped"
	if err := p.gpuClient.Health(ctx); err == nil {
		state = "running"
	}
	return &Status{
		State:    state,
		Provider: "local",
		GPUURL:   p.gpuClient.BaseURL(),
	}, nil
}

func (p *LocalProvider) Stop(ctx context.Context) error { return nil }
func (p *LocalProvider) StartIdleTimer()                {}
func (p *LocalProvider) Close()                         {}
