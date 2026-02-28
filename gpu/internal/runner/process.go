package runner

import (
	"context"
	"fmt"
	"io"
	"log"
	"path/filepath"
	"strconv"
	"sync"
	"time"

	"github.com/ThatCatDev/tanrenai/gpu/pkg/api"
)

const maxRestartAttempts = 3

// ProcessRunner manages a llama-server subprocess for model inference.
type ProcessRunner struct {
	sub       *Subprocess
	modelPath string
	modelName string
	opts      Options
	client    *Client
	baseURL   string

	// Crash detection and auto-restart.
	mu           sync.Mutex
	restarts     int
	crashNotify  chan error // receives an error each time the process crashes
	stopMonitor  chan struct{}
}

// NewProcessRunner creates a new ProcessRunner.
func NewProcessRunner() *ProcessRunner {
	return &ProcessRunner{
		crashNotify: make(chan error, 4),
		stopMonitor: make(chan struct{}),
	}
}

// CrashNotify returns a channel that receives an error each time the
// subprocess crashes unexpectedly. The channel is buffered; if the consumer
// falls behind, notifications are dropped.
func (r *ProcessRunner) CrashNotify() <-chan error {
	return r.crashNotify
}

func (r *ProcessRunner) Load(ctx context.Context, modelPath string, opts Options) error {
	r.modelPath = modelPath
	r.modelName = filepath.Base(modelPath)
	r.opts = opts

	if err := r.startSubprocess(ctx); err != nil {
		return err
	}

	// Start crash monitoring goroutine.
	go r.monitorCrashes()

	return nil
}

// startSubprocess creates and starts the llama-server subprocess.
func (r *ProcessRunner) startSubprocess(ctx context.Context) error {
	args := r.buildArgs()

	healthTimeout := r.opts.HealthTimeout
	if healthTimeout == 0 {
		healthTimeout = 120 * time.Second
	}

	sub, err := NewSubprocess(SubprocessConfig{
		BinDir:        r.opts.BinDir,
		Args:          args,
		Port:          r.opts.Port,
		Label:         "llama-server",
		Quiet:         r.opts.Quiet,
		HealthTimeout: healthTimeout,
	})
	if err != nil {
		return err
	}

	if err := sub.Start(ctx); err != nil {
		return err
	}

	r.sub = sub
	r.baseURL = sub.BaseURL()
	r.client = NewClient(r.baseURL)
	// Update opts.Port so restarts reuse the same allocated port.
	r.opts.Port = sub.Port()

	log.Printf("llama-server ready on port %d with model %s", sub.Port(), r.modelName)
	return nil
}

func (r *ProcessRunner) buildArgs() []string {
	args := []string{
		"--model", r.modelPath,
		"--ctx-size", strconv.Itoa(r.opts.CtxSize),
		"--host", "127.0.0.1",
	}

	if r.opts.GPULayers >= 0 {
		args = append(args, "--n-gpu-layers", strconv.Itoa(r.opts.GPULayers))
	} else {
		args = append(args, "--n-gpu-layers", "999")
	}

	if r.opts.Threads > 0 {
		args = append(args, "--threads", strconv.Itoa(r.opts.Threads))
	}

	if r.opts.FlashAttention {
		args = append(args, "--flash-attn", "on")
	}

	args = append(args, "--jinja")

	if r.opts.ChatTemplateFile != "" {
		args = append(args, "--chat-template-file", r.opts.ChatTemplateFile)
	}

	return args
}

// monitorCrashes watches for unexpected process exits and restarts.
func (r *ProcessRunner) monitorCrashes() {
	for {
		select {
		case <-r.stopMonitor:
			return
		case <-r.sub.Done():
			if r.sub.WasStopped() {
				return
			}

			exitCode := r.sub.ExitCode()
			log.Printf("[llama-server] process crashed (exit code %d)", exitCode)

			r.mu.Lock()
			r.restarts++
			attempt := r.restarts
			r.mu.Unlock()

			crashErr := fmt.Errorf("llama-server crashed (exit code %d, restart %d/%d)", exitCode, attempt, maxRestartAttempts)

			// Notify consumers (non-blocking).
			select {
			case r.crashNotify <- crashErr:
			default:
			}

			if attempt > maxRestartAttempts {
				log.Printf("[llama-server] max restart attempts (%d) reached, giving up", maxRestartAttempts)
				return
			}

			log.Printf("[llama-server] restarting (attempt %d/%d)...", attempt, maxRestartAttempts)
			// Use a timeout context for restart health check.
			restartCtx, cancel := context.WithTimeout(context.Background(), r.sub.healthTimeout)
			if err := r.startSubprocess(restartCtx); err != nil {
				log.Printf("[llama-server] restart failed: %v", err)
				cancel()
				return
			}
			cancel()
			log.Printf("[llama-server] restart successful")
		}
	}
}

func (r *ProcessRunner) Health(ctx context.Context) error {
	if r.sub == nil {
		return fmt.Errorf("llama-server not started")
	}
	return r.sub.healthCheck(ctx)
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
	// Stop the crash monitor so it doesn't try to restart.
	select {
	case <-r.stopMonitor:
		// Already closed.
	default:
		close(r.stopMonitor)
	}

	if r.sub != nil {
		return r.sub.GracefulStop()
	}
	return nil
}
