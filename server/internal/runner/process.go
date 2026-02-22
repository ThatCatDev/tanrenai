package runner

import (
	"context"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"time"

	"github.com/ThatCatDev/tanrenai/server/pkg/api"
)

// ProcessRunner manages a llama-server subprocess for model inference.
type ProcessRunner struct {
	cmd       *exec.Cmd
	modelPath string
	modelName string
	opts      Options
	client    *Client
	baseURL   string
}

// NewProcessRunner creates a new ProcessRunner.
func NewProcessRunner() *ProcessRunner {
	return &ProcessRunner{}
}

func (r *ProcessRunner) Load(ctx context.Context, modelPath string, opts Options) error {
	r.modelPath = modelPath
	r.modelName = filepath.Base(modelPath)
	r.opts = opts
	r.baseURL = fmt.Sprintf("http://127.0.0.1:%d", opts.Port)
	r.client = NewClient(r.baseURL)

	binName := "llama-server"
	if runtime.GOOS == "windows" {
		binName = "llama-server.exe"
	}

	binPath := filepath.Join(opts.BinDir, binName)
	if _, err := os.Stat(binPath); os.IsNotExist(err) {
		return fmt.Errorf("llama-server not found at %s — download it with 'tanrenai setup'", binPath)
	}

	args := []string{
		"--model", modelPath,
		"--port", strconv.Itoa(opts.Port),
		"--ctx-size", strconv.Itoa(opts.CtxSize),
		"--host", "127.0.0.1",
	}

	if opts.GPULayers >= 0 {
		args = append(args, "--n-gpu-layers", strconv.Itoa(opts.GPULayers))
	} else {
		args = append(args, "--n-gpu-layers", "999")
	}

	if opts.Threads > 0 {
		args = append(args, "--threads", strconv.Itoa(opts.Threads))
	}

	if opts.FlashAttention {
		args = append(args, "--flash-attn", "on")
	}

	args = append(args, "--jinja")

	if opts.ChatTemplateFile != "" {
		args = append(args, "--chat-template-file", opts.ChatTemplateFile)
	}

	// Use background context for the subprocess — it must outlive the HTTP
	// request that triggered the load. The caller's ctx is only used for the
	// health-check wait below.
	r.cmd = exec.Command(binPath, args...)
	r.cmd.Stdout = os.Stdout
	r.cmd.Stderr = os.Stderr
	r.cmd.Env = append(os.Environ(), "LD_LIBRARY_PATH="+opts.BinDir)

	log.Printf("Starting llama-server: %s %v", binPath, args)

	if err := r.cmd.Start(); err != nil {
		return fmt.Errorf("failed to start llama-server: %w", err)
	}

	// Wait for the server to become healthy
	if err := r.waitForHealth(ctx, 120*time.Second); err != nil {
		r.Close()
		return fmt.Errorf("llama-server failed to start: %w", err)
	}

	log.Printf("llama-server ready on port %d with model %s", opts.Port, r.modelName)
	return nil
}

func (r *ProcessRunner) waitForHealth(ctx context.Context, timeout time.Duration) error {
	deadline := time.Now().Add(timeout)
	ticker := time.NewTicker(500 * time.Millisecond)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-ticker.C:
			if time.Now().After(deadline) {
				return fmt.Errorf("timeout waiting for llama-server to become ready after %s", timeout)
			}

			if err := r.Health(ctx); err == nil {
				return nil
			}
		}
	}
}

func (r *ProcessRunner) Health(ctx context.Context) error {
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

func (r *ProcessRunner) ChatCompletion(ctx context.Context, req *api.ChatCompletionRequest) (*api.ChatCompletionResponse, error) {
	req.Stream = false
	return r.client.ChatCompletion(ctx, req)
}

func (r *ProcessRunner) ChatCompletionStream(ctx context.Context, req *api.ChatCompletionRequest, w io.Writer) error {
	req.Stream = true
	return r.client.ChatCompletionStream(ctx, req, w)
}

func (r *ProcessRunner) Tokenize(ctx context.Context, text string) (int, error) {
	return r.client.Tokenize(ctx, text)
}

func (r *ProcessRunner) ModelName() string {
	return r.modelName
}

func (r *ProcessRunner) Close() error {
	if r.cmd != nil && r.cmd.Process != nil {
		log.Printf("Stopping llama-server (PID %d)", r.cmd.Process.Pid)
		if err := r.cmd.Process.Kill(); err != nil {
			return fmt.Errorf("failed to kill llama-server: %w", err)
		}
		r.cmd.Wait()
	}
	return nil
}
