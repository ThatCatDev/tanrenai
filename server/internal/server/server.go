package server

import (
	"context"
	"fmt"
	"log"
	"net"
	"net/http"
	"time"

	"github.com/thatcatdev/tanrenai/server/internal/config"
	"github.com/thatcatdev/tanrenai/server/internal/models"
	"github.com/thatcatdev/tanrenai/server/internal/runner"
	"github.com/thatcatdev/tanrenai/server/internal/tools"
)

// Server is the tanrenai HTTP API server.
type Server struct {
	cfg          *config.Config
	http         *http.Server
	store        *models.Store
	runner       runner.Runner
	toolRegistry *tools.Registry
}

// New creates a new Server.
func New(cfg *config.Config) *Server {
	s := &Server{
		cfg:          cfg,
		store:        models.NewStore(cfg.ModelsDir),
		toolRegistry: tools.DefaultRegistry(),
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

	log.Printf("Tanrenai server listening on %s", s.http.Addr)
	log.Printf("Models dir: %s", s.cfg.ModelsDir)
	log.Printf("Bin dir: %s", s.cfg.BinDir)

	errCh := make(chan error, 1)
	go func() {
		errCh <- s.http.Serve(ln)
	}()

	select {
	case <-ctx.Done():
		log.Println("Shutting down server...")
		shutdownCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		if err := s.http.Shutdown(shutdownCtx); err != nil {
			log.Printf("Server shutdown error: %v", err)
		}
		if s.runner != nil {
			s.runner.Close()
		}
		return nil
	case err := <-errCh:
		return err
	}
}

// LoadModel loads a model by name into the runner.
func (s *Server) LoadModel(ctx context.Context, modelName string) error {
	modelPath, err := s.store.Resolve(modelName)
	if err != nil {
		return err
	}

	// Close existing runner if any
	if s.runner != nil {
		s.runner.Close()
		s.runner = nil
	}

	r := runner.NewProcessRunner()
	opts := runner.DefaultOptions()
	opts.BinDir = s.cfg.BinDir
	opts.GPULayers = s.cfg.GPULayers
	opts.CtxSize = s.cfg.CtxSize

	if err := r.Load(ctx, modelPath, opts); err != nil {
		return err
	}

	s.runner = r
	return nil
}
