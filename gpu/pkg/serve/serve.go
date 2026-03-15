// Package serve provides a public API for starting the tanrenai GPU server in-process.
package serve

import (
	"context"
	"fmt"

	"github.com/ThatCatDev/tanrenai/gpu/internal/config"
	"github.com/ThatCatDev/tanrenai/gpu/internal/models"
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
	EmbeddingModel string
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
	if cfg.EmbeddingModel != "" {
		icfg.EmbeddingModel = cfg.EmbeddingModel
	}

	if err := config.EnsureDirs(); err != nil {
		return fmt.Errorf("ensure dirs: %w", err)
	}

	srv := server.New(icfg)

	// Start embedding subprocess if configured.
	if icfg.EmbeddingModel != "" {
		er, err := srv.StartEmbeddingSubprocess(ctx, icfg.EmbeddingModel)
		if err != nil {
			return fmt.Errorf("embedding subprocess: %w", err)
		}
		srv.SetEmbeddingRunner(er)
	}

	return srv.Start(ctx)
}

// ModelsDir returns the default directory where models are stored.
func ModelsDir() string {
	return config.ModelsDir()
}

// DownloadProgress is called periodically during a model download.
type DownloadProgress = models.DownloadProgress

// DownloadModel downloads a GGUF model from a URL to the models directory.
func DownloadModel(url, destDir string, progress DownloadProgress) (string, error) {
	return models.Download(url, destDir, progress)
}

// ResolveModel resolves a model name to its file path in the models directory.
func ResolveModel(name string) (string, error) {
	store := models.NewStore(config.ModelsDir())
	return store.Resolve(name)
}
