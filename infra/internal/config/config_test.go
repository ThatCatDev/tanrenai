package config

import (
	"os"
	"testing"
)

func TestDefaults(t *testing.T) {
	cfg := Defaults()

	if cfg.MinGPURAM != 24 {
		t.Errorf("MinGPURAM = %v, want 24", cfg.MinGPURAM)
	}
	if cfg.MaxCostPerHr != 1.0 {
		t.Errorf("MaxCostPerHr = %v, want 1.0", cfg.MaxCostPerHr)
	}
	if cfg.DiskGB != 50 {
		t.Errorf("DiskGB = %v, want 50", cfg.DiskGB)
	}
	if cfg.Network != "none" {
		t.Errorf("Network = %q, want \"none\"", cfg.Network)
	}
	if cfg.GPUPort != 11435 {
		t.Errorf("GPUPort = %d, want 11435", cfg.GPUPort)
	}
}

func TestDefaultsFromEnv(t *testing.T) {
	os.Setenv("VASTAI_API_KEY", "test-key-123")
	defer os.Unsetenv("VASTAI_API_KEY")

	os.Setenv("HEADSCALE_URL", "https://hs.example.com")
	defer os.Unsetenv("HEADSCALE_URL")

	cfg := Defaults()

	if cfg.VastaiAPIKey != "test-key-123" {
		t.Errorf("VastaiAPIKey = %q, want \"test-key-123\"", cfg.VastaiAPIKey)
	}
	if cfg.HeadscaleURL != "https://hs.example.com" {
		t.Errorf("HeadscaleURL = %q, want \"https://hs.example.com\"", cfg.HeadscaleURL)
	}
}
