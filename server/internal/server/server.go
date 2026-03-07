package server

import (
	"context"
	"fmt"
	"log/slog"
	"net"
	"net/http"
	"time"

	"github.com/ThatCatDev/tanrenai/server/internal/config"
	"github.com/ThatCatDev/tanrenai/server/internal/gpuclient"
	"github.com/ThatCatDev/tanrenai/server/internal/gpuprovider"
	"github.com/ThatCatDev/tanrenai/server/internal/memory"
)

// Server is the tanrenai backend HTTP API server.
type Server struct {
	cfg       *config.Config
	http      *http.Server
	gpuClient *gpuclient.Client
	memStore  memory.Store
	provider  gpuprovider.Provider
}

// New creates a new backend Server.
func New(cfg *config.Config, gpuClient *gpuclient.Client, memStore memory.Store, provider gpuprovider.Provider) *Server {
	s := &Server{
		cfg:       cfg,
		gpuClient: gpuClient,
		memStore:  memStore,
		provider:  provider,
	}

	mux := http.NewServeMux()
	s.registerRoutes(mux)

	s.http = &http.Server{
		Addr:    fmt.Sprintf("%s:%d", cfg.Host, cfg.Port),
		Handler: withLogging(withCORS(mux)),
	}

	return s
}

// Start starts the server and blocks until the context is cancelled.
func (s *Server) Start(ctx context.Context) error {
	ln, err := net.Listen("tcp", s.http.Addr)
	if err != nil {
		return fmt.Errorf("listen: %w", err)
	}

	slog.Info("Tanrenai backend listening", "addr", s.http.Addr, "gpu_url", s.cfg.GPUURL, "gpu_provider", s.provider.Name())
	if s.memStore != nil {
		slog.Info("memory store enabled", "dir", s.cfg.MemoryDir)
	}

	s.provider.StartIdleTimer()

	errCh := make(chan error, 1)
	go func() {
		errCh <- s.http.Serve(ln)
	}()

	select {
	case <-ctx.Done():
		slog.Info("shutting down server")
		shutdownCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		if err := s.http.Shutdown(shutdownCtx); err != nil {
			slog.Error("server shutdown error", "error", err)
		}
		s.provider.Close()
		if s.memStore != nil {
			s.memStore.Close()
		}
		return nil
	case err := <-errCh:
		return err
	}
}
