package cmd

import (
	"context"
	"fmt"
	"log/slog"
	"net"
	"net/http"
	"time"

	gpuserve "github.com/ThatCatDev/tanrenai-gpu/pkg/serve"
	srvserve "github.com/ThatCatDev/tanrenai/server/pkg/serve"
)

type localOpts struct {
	GPULayers      int
	FlashAttention bool
	MemoryEnabled  bool
	EmbeddingModel string
	MemoryDir      string
	CPUMoE         bool
	NoKVOffload    bool
	FitVRAM        bool
}

// startLocalServers starts embedded GPU and backend servers on ephemeral ports.
// It returns the backend URL and a cleanup function that shuts both down.
func startLocalServers(ctx context.Context, opts localOpts, log *startupLog) (serverURL string, cleanup func(), err error) {
	gpuPort, err := freePort()
	if err != nil {
		return "", nil, fmt.Errorf("find free port for GPU server: %w", err)
	}
	srvPort, err := freePort()
	if err != nil {
		return "", nil, fmt.Errorf("find free port for backend server: %w", err)
	}

	srvCtx, srvCancel := context.WithCancel(ctx)

	// Start GPU server.
	gpuAddr := fmt.Sprintf("http://127.0.0.1:%d", gpuPort)
	gpuErrCh := make(chan error, 1)
	go func() {
		err := gpuserve.Start(srvCtx, gpuserve.Config{
			Host:           "127.0.0.1",
			Port:           gpuPort,
			GPULayers:      opts.GPULayers,
			FlashAttention: opts.FlashAttention,
			EmbeddingModel: opts.EmbeddingModel,
			CPUMoE:         opts.CPUMoE,
			NoKVOffload:    opts.NoKVOffload,
			FitVRAM:        opts.FitVRAM,
		})
		if err != nil {
			slog.Error("GPU server exited", "error", err)
		}
		gpuErrCh <- err
	}()

	if err := waitForHealth(srvCtx, gpuAddr, 30*time.Second); err != nil {
		srvCancel()

		return "", nil, fmt.Errorf("GPU server failed to start: %w", err)
	}
	log.Info(fmt.Sprintf("Local GPU server ready on :%d", gpuPort))

	// Start backend server.
	backendAddr := fmt.Sprintf("http://127.0.0.1:%d", srvPort)
	srvErrCh := make(chan error, 1)
	go func() {
		err := srvserve.Start(srvCtx, srvserve.Config{
			Host:          "127.0.0.1",
			Port:          srvPort,
			GPUURL:        gpuAddr,
			MemoryEnabled: opts.MemoryEnabled,
			MemoryDir:     opts.MemoryDir,
		})
		if err != nil {
			slog.Error("backend server exited", "error", err)
		}
		srvErrCh <- err
	}()

	if err := waitForHealth(srvCtx, backendAddr, 30*time.Second); err != nil {
		srvCancel()

		return "", nil, fmt.Errorf("backend server failed to start: %w", err)
	}
	log.Info(fmt.Sprintf("Local backend server ready on :%d", srvPort))

	cleanup = func() {
		srvCancel()
		// Drain error channels so goroutines can exit.
		<-gpuErrCh
		<-srvErrCh
	}

	return backendAddr, cleanup, nil
}

func freePort() (int, error) {
	l, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		return 0, err
	}
	port := l.Addr().(*net.TCPAddr).Port
	_ = l.Close()

	return port, nil
}

func waitForHealth(ctx context.Context, baseURL string, timeout time.Duration) error {
	deadline := time.After(timeout)
	for {
		select {
		case <-deadline:
			return fmt.Errorf("server at %s did not become healthy within %v", baseURL, timeout)
		case <-ctx.Done():
			return ctx.Err()
		default:
			resp, err := http.Get(baseURL + "/health")
			if err == nil {
				_ = resp.Body.Close()
				if resp.StatusCode == http.StatusOK {
					return nil
				}
			}
			time.Sleep(250 * time.Millisecond)
		}
	}
}
