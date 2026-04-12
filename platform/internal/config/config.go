package config

import "os"

// Config holds all configuration for the platform service.
type Config struct {
	Host             string  // bind address
	Port             int     // listen port
	DatabaseURL      string  // postgres connection string
	OIDCIssuer       string  // OIDC issuer URL (e.g. Dex)
	OIDCClientID     string  // OIDC client ID
	OIDCClientSecret string  // OIDC client secret
	EncryptionKey    string  // 32-byte hex string for AES-256-GCM
	DefaultIdleMin   int     // default idle timeout in minutes
	DefaultMaxCost   float64 // default max $/hr for offers
	GPUDockerImage   string  // Docker image for GPU instances
	FrontendOrigin   string  // frontend URL for CORS
}

// Defaults returns a Config with sensible defaults, reading env vars.
func Defaults() Config {
	return Config{
		Host:           "0.0.0.0",
		Port:           3000,
		DatabaseURL:    envOr("DATABASE_URL", "postgres://tanrenai:tanrenai@localhost:5432/tanrenai?sslmode=disable"),
		OIDCIssuer:     envOr("OIDC_ISSUER", ""),
		OIDCClientID:   envOr("OIDC_CLIENT_ID", "tanrenai"),
		OIDCClientSecret: envOr("OIDC_CLIENT_SECRET", ""),
		EncryptionKey:  envOr("ENCRYPTION_KEY", ""),
		DefaultIdleMin: 60,
		DefaultMaxCost: 1.0,
		GPUDockerImage: envOr("GPU_DOCKER_IMAGE", "thatcatdev/tanrenai-gpu:latest"),
		FrontendOrigin: envOr("FRONTEND_ORIGIN", "http://localhost:5173"),
	}
}

func envOr(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}
