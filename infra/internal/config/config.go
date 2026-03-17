package config

import (
	"fmt"
	"math"
	"os"
	"strings"
)

// Config holds all configuration for an infra deployment.
type Config struct {
	// vast.ai
	VastaiAPIKey   string
	VastaiInstance string  // existing instance ID, or empty to search+create
	GPUName        string  // filter offers by GPU name (e.g. "A100", "4090")
	MinGPURAM      float64 // minimum GPU RAM in GB for offer search
	MaxCostPerHr   float64 // maximum $/hr for offer search
	DiskGB         float64 // disk space in GB

	// Network tunnel
	Network       string // "headscale", "tailscale", or "none"
	HeadscaleURL  string
	HeadscaleAPI  string // API key for Headscale
	HeadscaleUser string // Headscale user name
	TailscaleAuth string // auth key for managed Tailscale

	// GPU server
	GPUPort   int
	ModelsDir string
	Model     string // model to pull on setup
}

// Defaults returns a Config with sensible defaults, reading env vars.
func Defaults() Config {
	return Config{
		VastaiAPIKey:   envOr("VASTAI_API_KEY", ""),
		VastaiInstance: envOr("VASTAI_INSTANCE_ID", ""),
		MinGPURAM:      24,
		MaxCostPerHr:   1.0,
		DiskGB:         50,
		Network:        "none",
		HeadscaleURL:   envOr("HEADSCALE_URL", ""),
		HeadscaleAPI:   envOr("HEADSCALE_API_KEY", ""),
		HeadscaleUser:  envOr("HEADSCALE_USER", ""),
		TailscaleAuth:  envOr("TAILSCALE_AUTH_KEY", ""),
		GPUPort:        11435,
	}
}

// VRAMForModelSize estimates the minimum GPU VRAM in GB needed to run a model
// of the given size (e.g. "8b", "27b", "72b", "120b") at Q4 quantization.
// Returns 0 if the string can't be parsed.
func VRAMForModelSize(size string) (float64, error) {
	s := strings.TrimSpace(strings.ToLower(size))
	s = strings.TrimSuffix(s, "b")

	var billions float64
	if _, err := fmt.Sscanf(s, "%f", &billions); err != nil {
		return 0, fmt.Errorf("invalid model size %q (expected e.g. 8b, 27b, 72b)", size)
	}

	// Q4 quantization: ~0.6 GB per billion params + 2 GB overhead for KV cache
	vram := billions*0.6 + 2
	// Round up to nearest whole GB
	return math.Ceil(vram), nil
}

// DiskForModelSize estimates the minimum disk space in GB needed to store a model
// of the given size at Q4 quantization, plus headroom for OS, build tools, etc.
func DiskForModelSize(size string) (float64, error) {
	vram, err := VRAMForModelSize(size)
	if err != nil {
		return 0, err
	}
	// Model file size ≈ VRAM needed, double it for headroom (partial downloads,
	// multiple models, build cache, OS, tools), minimum 100GB
	disk := vram*2 + 50
	if disk < 100 {
		disk = 100
	}
	// Round up to nearest 50GB
	disk = math.Ceil(disk/50) * 50

	return disk, nil
}

func envOr(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}

	return fallback
}
