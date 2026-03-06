// Package serve provides a public API for starting the tanrenai GPU server in-process.
package serve

import (
	"context"
	"fmt"

	"github.com/ThatCatDev/tanrenai/gpu/internal/config"
	"github.com/ThatCatDev/tanrenai/gpu/internal/server"
)

// Config holds the public configuration for the GPU server.
type Config struct {
	Host           string
	Port           int
	ModelsDir      string
	BinDir         string
	GPULayers      int
	CtxSize        int
	FlashAttention bool
}

// Start starts the GPU server and blocks until ctx is cancelled.
func Start(ctx context.Context, cfg Config) error {
	icfg := config.DefaultConfig()
	if cfg.Host != "" {
		icfg.Host = cfg.Host
	}
	if cfg.Port != 0 {
		icfg.Port = cfg.Port
	}
	if cfg.ModelsDir != "" {
		icfg.ModelsDir = cfg.ModelsDir
	}
	if cfg.BinDir != "" {
		icfg.BinDir = cfg.BinDir
	}
	if cfg.GPULayers != 0 {
		icfg.GPULayers = cfg.GPULayers
	}
	if cfg.CtxSize != 0 {
		icfg.CtxSize = cfg.CtxSize
	}
	icfg.FlashAttention = cfg.FlashAttention

	if err := config.EnsureDirs(); err != nil {
		return fmt.Errorf("ensure dirs: %w", err)
	}

	srv := server.New(icfg)
	return srv.Start(ctx)
}
