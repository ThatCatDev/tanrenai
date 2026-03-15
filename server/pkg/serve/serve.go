// Package serve provides a public API for starting the tanrenai backend server in-process.
package serve

import (
	"context"
	"fmt"
	"log/slog"
	"time"

	"github.com/ThatCatDev/tanrenai/server/internal/config"
	"github.com/ThatCatDev/tanrenai/server/internal/gpuclient"
	"github.com/ThatCatDev/tanrenai/server/internal/gpuprovider"
	"github.com/ThatCatDev/tanrenai/server/internal/memory"
	"github.com/ThatCatDev/tanrenai/server/internal/server"
	"github.com/ThatCatDev/tanrenai/server/internal/vastai"
)

// Config holds the public configuration for the backend server.
type Config struct {
	Host           string
	Port           int
	GPUURL         string
	MemoryEnabled  bool
	MemoryDir      string
	VastaiAPIKey   string // vast.ai API key
	VastaiInstance string // vast.ai instance ID
	IdleTimeout    string // auto-stop after inactivity (e.g. "20m")
}

// Start starts the backend server and blocks until ctx is cancelled.
func Start(ctx context.Context, cfg Config) error {
	icfg := config.DefaultConfig()
	if cfg.Host != "" {
		icfg.Host = cfg.Host
	}
	if cfg.Port != 0 {
		icfg.Port = cfg.Port
	}
	if cfg.GPUURL != "" {
		icfg.GPUURL = cfg.GPUURL
	}
	icfg.MemoryEnabled = cfg.MemoryEnabled
	if cfg.MemoryDir != "" {
		icfg.MemoryDir = cfg.MemoryDir
	}
	if cfg.VastaiAPIKey != "" {
		icfg.VastaiAPIKey = cfg.VastaiAPIKey
	}
	if cfg.VastaiInstance != "" {
		icfg.VastaiInstance = cfg.VastaiInstance
	}
	if cfg.IdleTimeout != "" {
		icfg.IdleTimeout = cfg.IdleTimeout
	}

	if err := config.EnsureDirs(icfg); err != nil {
		return fmt.Errorf("ensure dirs: %w", err)
	}

	gpu := gpuclient.New(icfg.GPUURL)

	var memStore memory.Store
	if icfg.MemoryEnabled {
		embedFunc := memory.NewRemoteEmbedFunc(gpu)
		store, err := memory.NewChromemStore(icfg.MemoryDir, embedFunc)
		if err != nil {
			return fmt.Errorf("memory store: %w", err)
		}
		memStore = store
		slog.Info("memory store initialized", "dir", icfg.MemoryDir)
	}

	var provider gpuprovider.Provider
	if icfg.VastaiAPIKey != "" && icfg.VastaiInstance != "" {
		idleTimeout, err := time.ParseDuration(icfg.IdleTimeout)
		if err != nil {
			idleTimeout = 20 * time.Minute
		}
		vastClient := vastai.NewClient(icfg.VastaiAPIKey)
		provider = gpuprovider.NewVastAIProvider(vastClient, gpu, icfg.VastaiInstance, icfg.GPUURL, idleTimeout)
		slog.Info("GPU provider configured", "provider", "vastai", "instance_id", icfg.VastaiInstance, "idle_timeout", idleTimeout)
	} else {
		provider = gpuprovider.NewLocalProvider(gpu)
		slog.Info("GPU provider configured", "provider", "local", "gpu_url", icfg.GPUURL)
	}

	srv := server.New(icfg, gpu, memStore, provider)
	return srv.Start(ctx)
}
