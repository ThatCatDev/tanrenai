package gpuprovider

import (
	"context"
	"fmt"
	"log"
	"sync"
	"time"

	"github.com/ThatCatDev/tanrenai/server/internal/gpuclient"
	"github.com/ThatCatDev/tanrenai/server/internal/vastai"
)

// VastAIProvider manages a vast.ai GPU instance lifecycle.
type VastAIProvider struct {
	client       *vastai.Client
	gpuClient    *gpuclient.Client
	instanceID   string
	gpuURL       string
	idleTimeout  time.Duration
	lastActivity time.Time

	mu       sync.Mutex
	stopCh   chan struct{}
	starting bool
}

// NewVastAIProvider creates a provider backed by a vast.ai instance.
func NewVastAIProvider(client *vastai.Client, gpuClient *gpuclient.Client, instanceID, gpuURL string, idleTimeout time.Duration) *VastAIProvider {
	return &VastAIProvider{
		client:       client,
		gpuClient:    gpuClient,
		instanceID:   instanceID,
		gpuURL:       gpuURL,
		idleTimeout:  idleTimeout,
		lastActivity: time.Now(),
	}
}

func (p *VastAIProvider) Name() string { return "vastai" }

// RecordActivity resets the idle timer.
func (p *VastAIProvider) RecordActivity() {
	p.mu.Lock()
	p.lastActivity = time.Now()
	p.mu.Unlock()
}

// EnsureRunning starts the instance if stopped and waits until the GPU server is healthy.
func (p *VastAIProvider) EnsureRunning(ctx context.Context) error {
	p.mu.Lock()
	if p.starting {
		p.mu.Unlock()
		return p.waitForHealthy(ctx)
	}
	p.mu.Unlock()

	if err := p.gpuClient.Health(ctx); err == nil {
		return nil
	}

	p.mu.Lock()
	p.starting = true
	p.mu.Unlock()
	defer func() {
		p.mu.Lock()
		p.starting = false
		p.mu.Unlock()
	}()

	log.Printf("Starting vast.ai instance %s...", p.instanceID)
	if err := p.client.StartInstance(ctx, p.instanceID); err != nil {
		return fmt.Errorf("start instance: %w", err)
	}

	return p.waitForHealthy(ctx)
}

func (p *VastAIProvider) waitForHealthy(ctx context.Context) error {
	ticker := time.NewTicker(5 * time.Second)
	defer ticker.Stop()

	timeout := time.After(5 * time.Minute)

	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-timeout:
			return fmt.Errorf("timeout waiting for GPU server to become healthy")
		case <-ticker.C:
			if err := p.gpuClient.Health(ctx); err == nil {
				log.Printf("GPU server is healthy")
				return nil
			}
		}
	}
}

// Status returns the current instance status.
func (p *VastAIProvider) Status(ctx context.Context) (*Status, error) {
	if err := p.gpuClient.Health(ctx); err == nil {
		p.mu.Lock()
		idle := p.lastActivity
		p.mu.Unlock()
		return &Status{
			State:     "running",
			Provider:  "vastai",
			GPUURL:    p.gpuURL,
			IdleSince: &idle,
		}, nil
	}

	if p.client != nil && p.instanceID != "" {
		inst, err := p.client.GetInstance(ctx, p.instanceID)
		if err == nil {
			return &Status{
				State:    inst.Status,
				Provider: "vastai",
				GPUURL:   p.gpuURL,
			}, nil
		}
	}

	return &Status{State: "stopped", Provider: "vastai"}, nil
}

// Stop stops the GPU instance.
func (p *VastAIProvider) Stop(ctx context.Context) error {
	if p.client == nil || p.instanceID == "" {
		return fmt.Errorf("vast.ai not configured")
	}
	log.Printf("Stopping vast.ai instance %s...", p.instanceID)
	return p.client.StopInstance(ctx, p.instanceID)
}

// StartIdleTimer starts a goroutine that stops the instance after idleTimeout of no requests.
func (p *VastAIProvider) StartIdleTimer() {
	if p.idleTimeout <= 0 {
		return
	}

	p.mu.Lock()
	if p.stopCh != nil {
		close(p.stopCh)
	}
	p.stopCh = make(chan struct{})
	stopCh := p.stopCh
	p.mu.Unlock()

	go func() {
		ticker := time.NewTicker(1 * time.Minute)
		defer ticker.Stop()

		for {
			select {
			case <-stopCh:
				return
			case <-ticker.C:
				p.mu.Lock()
				idle := time.Since(p.lastActivity)
				p.mu.Unlock()

				if idle >= p.idleTimeout {
					log.Printf("Instance idle for %v, stopping...", idle.Round(time.Second))
					ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
					if err := p.Stop(ctx); err != nil {
						log.Printf("Failed to stop idle instance: %v", err)
					}
					cancel()
					return
				}
			}
		}
	}()
}

// Close stops the idle timer.
func (p *VastAIProvider) Close() {
	p.mu.Lock()
	if p.stopCh != nil {
		close(p.stopCh)
		p.stopCh = nil
	}
	p.mu.Unlock()
}
