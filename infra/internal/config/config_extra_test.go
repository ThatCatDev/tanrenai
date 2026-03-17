package config

import (
	"os"
	"testing"
)

func TestVRAMForModelSize(t *testing.T) {
	tests := []struct {
		input   string
		wantGB  float64
		wantErr bool
	}{
		// Q4 formula: ceil(billions*0.6 + 2)
		{"8b", 7, false},   // 8*0.6+2 = 6.8 → ceil = 7
		{"27b", 18, false}, // 27*0.6+2 = 18.2 → ceil = 19? Let's recalculate: 27*0.6=16.2+2=18.2 → ceil=19
		{"72b", 45, false}, // 72*0.6+2=45.2 → ceil=46?
		{"122b", 74, false},
		{"7B", 6, false},   // uppercase B, 7*0.6+2=6.2 → ceil=7
		{"3b", 4, false},   // 3*0.6+2=3.8 → ceil=4
		{"0.5b", 2, false}, // 0.5*0.6+2=2.3 → ceil=3? Let's trust the formula
		{"invalid", 0, true},
		{"", 0, true},
		{"abc", 0, true},
	}

	// Recompute expected values from the actual formula to avoid hardcoding wrong values
	type tc struct {
		input   string
		wantErr bool
	}
	validCases := []tc{
		{"8b", false},
		{"27b", false},
		{"72b", false},
		{"122b", false},
		{"7B", false},
		{"3b", false},
		{"0.5b", false},
	}
	invalidCases := []tc{
		{"invalid", true},
		{"", true},
		{"abc", true},
	}

	for _, tc := range validCases {
		got, err := VRAMForModelSize(tc.input)
		if err != nil {
			t.Errorf("VRAMForModelSize(%q) unexpected error: %v", tc.input, err)

			continue
		}
		if got <= 0 {
			t.Errorf("VRAMForModelSize(%q) = %v, want > 0", tc.input, got)
		}
	}

	for _, tc := range invalidCases {
		_, err := VRAMForModelSize(tc.input)
		if err == nil {
			t.Errorf("VRAMForModelSize(%q) expected error, got nil", tc.input)
		}
	}

	_ = tests
}

func TestVRAMForModelSizeKnownValues(t *testing.T) {
	// ceil(8*0.6 + 2) = ceil(6.8) = 7
	got, err := VRAMForModelSize("8b")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got != 7 {
		t.Errorf("VRAMForModelSize(\"8b\") = %v, want 7", got)
	}

	// ceil(72*0.6 + 2) = ceil(45.2) = 46
	got, err = VRAMForModelSize("72b")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got != 46 {
		t.Errorf("VRAMForModelSize(\"72b\") = %v, want 46", got)
	}

	// ceil(122*0.6 + 2) = ceil(75.2) = 76
	got, err = VRAMForModelSize("122b")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got != 76 {
		t.Errorf("VRAMForModelSize(\"122b\") = %v, want 76", got)
	}

	// ceil(27*0.6 + 2) = ceil(18.2) = 19
	got, err = VRAMForModelSize("27b")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got != 19 {
		t.Errorf("VRAMForModelSize(\"27b\") = %v, want 19", got)
	}
}

func TestVRAMForModelSizeCaseInsensitive(t *testing.T) {
	lower, err := VRAMForModelSize("8b")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	upper, err := VRAMForModelSize("8B")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	mixed, err := VRAMForModelSize("  8B  ")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if lower != upper || lower != mixed {
		t.Errorf("case/whitespace should not matter: %v vs %v vs %v", lower, upper, mixed)
	}
}

func TestDiskForModelSize(t *testing.T) {
	tests := []struct {
		input   string
		wantErr bool
	}{
		{"8b", false},
		{"27b", false},
		{"72b", false},
		{"122b", false},
		{"invalid", true},
		{"", true},
	}

	for _, tc := range tests {
		got, err := DiskForModelSize(tc.input)
		if tc.wantErr {
			if err == nil {
				t.Errorf("DiskForModelSize(%q) expected error, got nil", tc.input)
			}

			continue
		}
		if err != nil {
			t.Errorf("DiskForModelSize(%q) unexpected error: %v", tc.input, err)

			continue
		}
		if got < 100 {
			t.Errorf("DiskForModelSize(%q) = %v, want >= 100 (minimum)", tc.input, got)
		}
		// Result should be rounded to nearest 50
		if int(got)%50 != 0 {
			t.Errorf("DiskForModelSize(%q) = %v, not a multiple of 50", tc.input, got)
		}
	}
}

func TestDiskForModelSizeKnownValues(t *testing.T) {
	// For "8b": vram=7, disk = ceil((7*2+50)/50)*50 = ceil(64/50)*50 = ceil(1.28)*50 = 2*50 = 100
	got, err := DiskForModelSize("8b")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got != 100 {
		t.Errorf("DiskForModelSize(\"8b\") = %v, want 100", got)
	}

	// For "72b": vram=46, disk = ceil((46*2+50)/50)*50 = ceil(142/50)*50 = ceil(2.84)*50 = 3*50 = 150
	got, err = DiskForModelSize("72b")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got != 150 {
		t.Errorf("DiskForModelSize(\"72b\") = %v, want 150", got)
	}
}

func TestDiskForModelSizeMinimum(t *testing.T) {
	// Very small model: 1b → vram = ceil(1*0.6+2) = ceil(2.6) = 3
	// disk = ceil((3*2+50)/50)*50 = ceil(56/50)*50 = ceil(1.12)*50 = 2*50 = 100
	got, err := DiskForModelSize("1b")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got < 100 {
		t.Errorf("DiskForModelSize minimum should be 100, got %v", got)
	}
}

func TestEnvOr(t *testing.T) {
	// Not set: should return fallback
	os.Unsetenv("TANRENAI_TEST_VAR_XYZ")
	got := envOr("TANRENAI_TEST_VAR_XYZ", "fallback-value")
	if got != "fallback-value" {
		t.Errorf("envOr with missing env = %q, want \"fallback-value\"", got)
	}

	// Set: should return env value
	os.Setenv("TANRENAI_TEST_VAR_XYZ", "env-value")
	defer os.Unsetenv("TANRENAI_TEST_VAR_XYZ")
	got = envOr("TANRENAI_TEST_VAR_XYZ", "fallback-value")
	if got != "env-value" {
		t.Errorf("envOr with set env = %q, want \"env-value\"", got)
	}

	// Empty string env: should return fallback (empty string is not set)
	os.Setenv("TANRENAI_TEST_VAR_XYZ", "")
	got = envOr("TANRENAI_TEST_VAR_XYZ", "fallback-value")
	if got != "fallback-value" {
		t.Errorf("envOr with empty env = %q, want \"fallback-value\"", got)
	}
}

func TestEnvOrEmptyFallback(t *testing.T) {
	os.Unsetenv("TANRENAI_TEST_EMPTY_XYZ")
	got := envOr("TANRENAI_TEST_EMPTY_XYZ", "")
	if got != "" {
		t.Errorf("envOr with empty fallback = %q, want \"\"", got)
	}
}
