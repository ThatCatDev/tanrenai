package server

import (
	"context"
	"fmt"
	"log"
	"net"
	"net/http"
	"time"

	"github.com/ThatCatDev/tanrenai/gpu/internal/config"
	"github.com/ThatCatDev/tanrenai/gpu/internal/models"
	"github.com/ThatCatDev/tanrenai/gpu/internal/runner"
	"github.com/ThatCatDev/tanrenai/gpu/internal/training"
)

// Server is the tanrenai GPU server — pure inference + training API.
type Server struct {
	cfg             *config.Config
	http            *http.Server
	store           *models.Store
	runner          runner.Runner
	embeddingRunner *EmbeddingSubprocess
	trainingManager *training.Manager
}

// EmbeddingSubprocess wraps an embedding server subprocess.
type EmbeddingSubprocess struct {
	Sub     *runner.Subprocess
	BaseURL string
}

// New creates a new GPU Server.
func New(cfg *config.Config) *Server {
	s := &Server{
		cfg:   cfg,
		store: models.NewStore(cfg.ModelsDir),
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

	log.Printf("Tanrenai GPU server listening on %s", s.http.Addr)
	log.Printf("Models dir: %s", s.cfg.ModelsDir)
	log.Printf("Bin dir: %s", s.cfg.BinDir)

	errCh := make(chan error, 1)
	go func() {
		errCh <- s.http.Serve(ln)
	}()

	select {
	case <-ctx.Done():
		log.Println("Shutting down GPU server...")
		shutdownCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		if err := s.http.Shutdown(shutdownCtx); err != nil {
			log.Printf("Server shutdown error: %v", err)
		}
		if s.runner != nil {
			s.runner.Close()
		}
		if s.embeddingRunner != nil {
			s.embeddingRunner.Sub.GracefulStop()
		}
		return nil
	case err := <-errCh:
		return err
	}
}

// SetTrainingManager sets the training manager for fine-tuning API endpoints.
func (s *Server) SetTrainingManager(m *training.Manager) {
	s.trainingManager = m
}

// SetEmbeddingRunner sets the embedding subprocess for the /v1/embeddings endpoint.
func (s *Server) SetEmbeddingRunner(er *EmbeddingSubprocess) {
	s.embeddingRunner = er
}

// StartEmbeddingSubprocess resolves the model and spawns a llama-server in embedding mode.
func (s *Server) StartEmbeddingSubprocess(ctx context.Context, modelName string) (*EmbeddingSubprocess, error) {
	modelPath, err := s.store.Resolve(modelName)
	if err != nil {
		return nil, err
	}

	args := []string{
		"--model", modelPath,
		"--embedding",
		"--ctx-size", "512",
		"--host", "127.0.0.1",
		"--n-gpu-layers", "999",
	}

	sub, err := runner.NewSubprocess(runner.SubprocessConfig{
		BinDir:        s.cfg.BinDir,
		Args:          args,
		Label:         "embedding",
		HealthTimeout: 60 * time.Second,
	})
	if err != nil {
		return nil, err
	}

	if err := sub.Start(ctx); err != nil {
		return nil, err
	}

	log.Printf("Embedding server ready on %s (model: %s)", sub.BaseURL(), modelName)
	return &EmbeddingSubprocess{Sub: sub, BaseURL: sub.BaseURL()}, nil
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
	opts.ChatTemplateFile = s.cfg.ChatTemplateFile
	opts.FlashAttention = s.cfg.FlashAttention
	opts.ReasoningFormat = s.cfg.ReasoningFormat

	if err := r.Load(ctx, modelPath, opts); err != nil {
		return err
	}

	s.runner = r
	return nil
}
