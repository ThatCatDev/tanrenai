package vastai

import (
	"fmt"
	"math"
	"strings"
)

// VRAMForModelSize estimates the minimum GPU VRAM in GB needed to run a model
// of the given size (e.g. "8b", "27b", "72b", "120b") at Q4 quantization.
func VRAMForModelSize(size string) (float64, error) {
	s := strings.TrimSpace(strings.ToLower(size))
	s = strings.TrimSuffix(s, "b")

	var billions float64
	if _, err := fmt.Sscanf(s, "%f", &billions); err != nil {
		return 0, fmt.Errorf("invalid model size %q (expected e.g. 8b, 27b, 72b)", size)
	}

	// Q4 quantization: ~0.6 GB per billion params + 2 GB overhead for KV cache
	vram := billions*0.6 + 2
	return math.Ceil(vram), nil
}

// DiskForModelSize estimates the minimum disk space in GB needed to store a model
// of the given size at Q4 quantization, plus headroom for OS, build tools, etc.
func DiskForModelSize(size string) (float64, error) {
	vram, err := VRAMForModelSize(size)
	if err != nil {
		return 0, err
	}
	// Model file size ~ VRAM needed, double for headroom, minimum 100GB
	disk := vram*2 + 50
	if disk < 100 {
		disk = 100
	}
	// Round up to nearest 50GB
	disk = math.Ceil(disk/50) * 50
	return disk, nil
}
