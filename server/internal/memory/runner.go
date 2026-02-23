package memory

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strconv"
	"time"

	"github.com/ThatCatDev/tanrenai/server/internal/runner"
)

// EmbeddingRunner manages a llama-server subprocess for embedding generation.
type EmbeddingRunner struct {
	sub     *runner.Subprocess
	baseURL string
}

// EmbeddingRunnerConfig configures the embedding runner.
type EmbeddingRunnerConfig struct {
	ModelPath string
	BinDir    string
	Port      int
	GPULayers int // 0 = CPU only (default for small embedding models)
}

// NewEmbeddingRunner creates and starts a llama-server for embeddings.
func NewEmbeddingRunner(ctx context.Context, cfg EmbeddingRunnerConfig) (*EmbeddingRunner, error) {
	args := []string{
		"--model", cfg.ModelPath,
		"--ctx-size", "512",
		"--host", "127.0.0.1",
		"--n-gpu-layers", strconv.Itoa(cfg.GPULayers),
		"--embeddings",
	}

	healthTimeout := 60 * time.Second

	sub, err := runner.NewSubprocess(runner.SubprocessConfig{
		BinDir:        cfg.BinDir,
		Args:          args,
		Port:          cfg.Port,
		Label:         "embedding",
		Quiet:         true,
		HealthTimeout: healthTimeout,
	})
	if err != nil {
		return nil, fmt.Errorf("configure embedding server: %w", err)
	}

	if err := sub.Start(ctx); err != nil {
		return nil, fmt.Errorf("start embedding server: %w", err)
	}

	return &EmbeddingRunner{
		sub:     sub,
		baseURL: sub.BaseURL(),
	}, nil
}

// BaseURL returns the base URL of the embedding server.
func (r *EmbeddingRunner) BaseURL() string {
	return r.baseURL
}

// ContextSize queries the embedding server's /props endpoint to discover the
// model's native context window. Returns 0 if detection fails.
func (r *EmbeddingRunner) ContextSize(ctx context.Context) int {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, r.baseURL+"/props", nil)
	if err != nil {
		return 0
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return 0
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return 0
	}
	var props struct {
		DefaultGeneration struct {
			NCtx int `json:"n_ctx"`
		} `json:"default_generation_settings"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&props); err != nil {
		return 0
	}
	return props.DefaultGeneration.NCtx
}

// Close stops the embedding server subprocess gracefully.
func (r *EmbeddingRunner) Close() error {
	if r.sub != nil {
		return r.sub.GracefulStop()
	}
	return nil
}
