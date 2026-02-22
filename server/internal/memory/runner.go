package memory

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"time"
)

// EmbeddingRunner manages a llama-server subprocess for embedding generation.
type EmbeddingRunner struct {
	cmd     *exec.Cmd
	port    int
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
	if cfg.Port == 0 {
		cfg.Port = 18081
	}

	binName := "llama-server"
	if runtime.GOOS == "windows" {
		binName = "llama-server.exe"
	}

	binPath := filepath.Join(cfg.BinDir, binName)
	if _, err := os.Stat(binPath); os.IsNotExist(err) {
		return nil, fmt.Errorf("llama-server not found at %s", binPath)
	}

	args := []string{
		"--model", cfg.ModelPath,
		"--port", strconv.Itoa(cfg.Port),
		"--ctx-size", "512",
		"--host", "127.0.0.1",
		"--n-gpu-layers", strconv.Itoa(cfg.GPULayers),
		"--embeddings",
	}

	cmd := exec.CommandContext(ctx, binPath, args...)
	cmd.Stdout = io.Discard
	cmd.Stderr = io.Discard
	cmd.Env = append(os.Environ(), "LD_LIBRARY_PATH="+cfg.BinDir)

	r := &EmbeddingRunner{
		cmd:     cmd,
		port:    cfg.Port,
		baseURL: fmt.Sprintf("http://127.0.0.1:%d", cfg.Port),
	}

	if err := cmd.Start(); err != nil {
		return nil, fmt.Errorf("failed to start embedding server: %w", err)
	}

	if err := r.waitForHealth(ctx, 60*time.Second); err != nil {
		r.Close()
		return nil, fmt.Errorf("embedding server failed to start: %w", err)
	}

	return r, nil
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

// Close stops the embedding server subprocess.
func (r *EmbeddingRunner) Close() error {
	if r.cmd != nil && r.cmd.Process != nil {
		if err := r.cmd.Process.Kill(); err != nil {
			return fmt.Errorf("failed to kill embedding server: %w", err)
		}
		r.cmd.Wait()
	}
	return nil
}

func (r *EmbeddingRunner) waitForHealth(ctx context.Context, timeout time.Duration) error {
	deadline := time.Now().Add(timeout)
	ticker := time.NewTicker(500 * time.Millisecond)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-ticker.C:
			if time.Now().After(deadline) {
				return fmt.Errorf("timeout waiting for embedding server after %s", timeout)
			}
			if r.healthCheck(ctx) == nil {
				return nil
			}
		}
	}
}

func (r *EmbeddingRunner) healthCheck(ctx context.Context) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, r.baseURL+"/health", nil)
	if err != nil {
		return err
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("health check returned %d", resp.StatusCode)
	}
	return nil
}
