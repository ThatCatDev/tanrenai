package config

import (
	"os"
	"path/filepath"
	"runtime"
)

// Config holds the backend server configuration.
type Config struct {
	Host           string
	Port           int
	GPUURL         string // URL of the GPU server
	MemoryEnabled  bool
	MemoryDir      string
	VastaiAPIKey   string
	VastaiInstance string
	IdleTimeout    string // duration string, e.g. "20m"
}

// DefaultConfig returns a Config with sensible defaults.
func DefaultConfig() *Config {
	return &Config{
		Host:          "0.0.0.0",
		Port:          8080,
		GPUURL:        "http://localhost:11435",
		MemoryEnabled: false,
		MemoryDir:     MemoryDir(),
		IdleTimeout:   "20m",
	}
}

// DataDir returns the default data directory for tanrenai.
func DataDir() string {
	if dir := os.Getenv("TANRENAI_DATA_DIR"); dir != "" {
		return dir
	}
	if runtime.GOOS == "windows" {
		return filepath.Join(os.Getenv("LOCALAPPDATA"), "tanrenai")
	}
	home, _ := os.UserHomeDir()
	return filepath.Join(home, ".local", "share", "tanrenai")
}

// MemoryDir returns the directory where memory/RAG data is stored.
func MemoryDir() string {
	return filepath.Join(DataDir(), "memory")
}

// EnsureDirs creates the required directories if they don't exist.
func EnsureDirs(cfg *Config) error {
	dirs := []string{DataDir()}
	if cfg.MemoryEnabled {
		dirs = append(dirs, cfg.MemoryDir)
	}
	for _, dir := range dirs {
		if err := os.MkdirAll(dir, 0755); err != nil {
			return err
		}
	}
	return nil
}
