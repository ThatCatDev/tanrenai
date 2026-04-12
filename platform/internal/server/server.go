package server

import (
	"context"
	"fmt"
	"log/slog"
	"net"
	"net/http"
	"time"

	"github.com/ThatCatDev/tanrenai/platform/internal/auth"
	"github.com/ThatCatDev/tanrenai/platform/internal/config"
	"github.com/ThatCatDev/tanrenai/platform/internal/database"
	"github.com/ThatCatDev/tanrenai/platform/internal/instance"
)

// Server is the platform HTTP server.
type Server struct {
	cfg     config.Config
	db      *database.DB
	auth    *auth.Middleware
	manager *instance.Manager
}

// New creates a new platform server.
func New(cfg config.Config, db *database.DB) *Server {
	provisioner := instance.NewProvisioner(db, cfg.GPUDockerImage, cfg.EncryptionKey)
	manager := instance.NewManager(db, provisioner)

	return &Server{
		cfg:     cfg,
		db:      db,
		manager: manager,
	}
}

// Start starts the HTTP server and blocks until the context is cancelled.
func (s *Server) Start(ctx context.Context) error {
	// Initialize OIDC if configured
	if s.cfg.OIDCIssuer != "" {
		verifier, err := auth.NewOIDCVerifier(ctx, s.cfg.OIDCIssuer, s.cfg.OIDCClientID)
		if err != nil {
			return fmt.Errorf("initialize OIDC: %w", err)
		}
		s.auth = auth.NewMiddleware(verifier, s.db)
		slog.Info("OIDC authentication enabled", "issuer", s.cfg.OIDCIssuer)
	} else {
		slog.Warn("OIDC not configured — running without authentication")
	}

	mux := http.NewServeMux()
	s.registerRoutes(mux)

	handler := s.withCORS(s.withLogging(mux))

	addr := net.JoinHostPort(s.cfg.Host, fmt.Sprintf("%d", s.cfg.Port))
	srv := &http.Server{
		Addr:    addr,
		Handler: handler,
	}

	go func() {
		<-ctx.Done()
		slog.Info("shutting down server")
		s.manager.Close()
		shutdownCtx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		_ = srv.Shutdown(shutdownCtx)
	}()

	slog.Info("platform server starting", "addr", addr)
	if err := srv.ListenAndServe(); err != http.ErrServerClosed {
		return err
	}
	return nil
}

func (s *Server) withLogging(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		next.ServeHTTP(w, r)
		slog.Debug("request", "method", r.Method, "path", r.URL.Path, "duration", time.Since(start).Round(time.Millisecond))
	})
}

func (s *Server) withCORS(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Access-Control-Allow-Origin", s.cfg.FrontendOrigin)
		w.Header().Set("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization")
		w.Header().Set("Access-Control-Allow-Credentials", "true")

		if r.Method == http.MethodOptions {
			w.WriteHeader(http.StatusNoContent)
			return
		}

		next.ServeHTTP(w, r)
	})
}
