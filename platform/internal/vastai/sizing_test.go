package vastai

import "testing"

func TestVRAMForModelSize(t *testing.T) {
	tests := []struct {
		size string
		want float64
	}{
		{"8b", 7},    // 8*0.6 + 2 = 6.8 → ceil = 7
		{"27b", 19},  // 27*0.6 + 2 = 18.2 → ceil = 19
		{"72b", 46},  // 72*0.6 + 2 = 45.2 → ceil = 46
		{"120b", 74}, // 120*0.6 + 2 = 74.0
	}
	for _, tt := range tests {
		got, err := VRAMForModelSize(tt.size)
		if err != nil {
			t.Errorf("VRAMForModelSize(%q) error: %v", tt.size, err)
			continue
		}
		if got != tt.want {
			t.Errorf("VRAMForModelSize(%q) = %v, want %v", tt.size, got, tt.want)
		}
	}
}

func TestVRAMForModelSizeInvalid(t *testing.T) {
	_, err := VRAMForModelSize("abc")
	if err == nil {
		t.Fatal("expected error for invalid model size")
	}
}

func TestDiskForModelSize(t *testing.T) {
	tests := []struct {
		size string
		want float64
	}{
		{"8b", 100},  // 7*2+50=64 → min 100
		{"72b", 150}, // 46*2+50=142 → ceil to 150
		{"120b", 200}, // 74*2+50=198 → ceil to 200
	}
	for _, tt := range tests {
		got, err := DiskForModelSize(tt.size)
		if err != nil {
			t.Errorf("DiskForModelSize(%q) error: %v", tt.size, err)
			continue
		}
		if got != tt.want {
			t.Errorf("DiskForModelSize(%q) = %v, want %v", tt.size, got, tt.want)
		}
	}
}
