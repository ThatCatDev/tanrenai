package config

import (
	"os"
	"path/filepath"
	"runtime"
)

// DataDir returns the default data directory for tanrenai.
// Windows: %LOCALAPPDATA%\tanrenai
// Linux/Mac: ~/.local/share/tanrenai
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

// ModelsDir returns the directory where models are stored.
func ModelsDir() string {
	if dir := os.Getenv("TANRENAI_MODELS_DIR"); dir != "" {
		return dir
	}
	return filepath.Join(DataDir(), "models")
}

// BinDir returns the directory where llama-server binaries are stored.
func BinDir() string {
	return filepath.Join(DataDir(), "bin")
}

// EnsureDirs creates the required directories if they don't exist.
func EnsureDirs() error {
	dirs := []string{DataDir(), ModelsDir(), BinDir()}
	for _, dir := range dirs {
		if err := os.MkdirAll(dir, 0755); err != nil {
			return err
		}
	}
	return nil
}
