package ui

import (
	"os"
	"path/filepath"
	"testing"
)

func TestModelsDir(t *testing.T) {
	t.Run("from TANRENAI_MODELS_DIR", func(t *testing.T) {
		t.Setenv("TANRENAI_MODELS_DIR", "/custom/models")
		got := modelsDir()
		if got != "/custom/models" {
			t.Errorf("modelsDir() = %q, want /custom/models", got)
		}
	})

	t.Run("from TANRENAI_DATA_DIR", func(t *testing.T) {
		t.Setenv("TANRENAI_MODELS_DIR", "")
		t.Setenv("TANRENAI_DATA_DIR", "/custom/data")
		got := modelsDir()
		want := filepath.Join("/custom/data", "models")
		if got != want {
			t.Errorf("modelsDir() = %q, want %q", got, want)
		}
	})

	t.Run("default", func(t *testing.T) {
		t.Setenv("TANRENAI_MODELS_DIR", "")
		t.Setenv("TANRENAI_DATA_DIR", "")
		got := modelsDir()
		home, _ := os.UserHomeDir()
		want := filepath.Join(home, ".local", "share", "tanrenai", "models")
		if got != want {
			t.Errorf("modelsDir() = %q, want %q", got, want)
		}
	})
}

func TestScanLocalModels(t *testing.T) {
	tmpDir := t.TempDir()
	t.Setenv("TANRENAI_MODELS_DIR", tmpDir)

	// Create some test files
	os.WriteFile(filepath.Join(tmpDir, "llama-7b.gguf"), []byte("fake"), 0644)
	os.WriteFile(filepath.Join(tmpDir, "mistral.GGUF"), []byte("fake"), 0644)
	os.WriteFile(filepath.Join(tmpDir, "readme.txt"), []byte("not a model"), 0644)

	names := scanLocalModels()
	if len(names) != 2 {
		t.Fatalf("expected 2 models, got %d: %v", len(names), names)
	}

	found := map[string]bool{}
	for _, n := range names {
		found[n] = true
	}
	if !found["llama-7b"] {
		t.Error("expected llama-7b in results")
	}
	if !found["mistral"] {
		t.Error("expected mistral in results")
	}
}

func TestScanLocalModelsEmpty(t *testing.T) {
	tmpDir := t.TempDir()
	t.Setenv("TANRENAI_MODELS_DIR", tmpDir)

	names := scanLocalModels()
	if len(names) != 0 {
		t.Fatalf("expected 0 models, got %d", len(names))
	}
}

func TestScanLocalModelsMissingDir(t *testing.T) {
	t.Setenv("TANRENAI_MODELS_DIR", "/nonexistent/path/models")

	names := scanLocalModels()
	if len(names) != 0 {
		t.Fatalf("expected 0 models for missing dir, got %d", len(names))
	}
}
