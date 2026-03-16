package config

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestDefaultConfig(t *testing.T) {
	cfg := DefaultConfig()

	if cfg.Host != "0.0.0.0" {
		t.Errorf("Host = %q, want 0.0.0.0", cfg.Host)
	}
	if cfg.Port != 8080 {
		t.Errorf("Port = %d, want 8080", cfg.Port)
	}
	if cfg.GPUURL != "http://localhost:11435" {
		t.Errorf("GPUURL = %q, want http://localhost:11435", cfg.GPUURL)
	}
	if cfg.MemoryEnabled {
		t.Error("MemoryEnabled should default to false")
	}
	if cfg.IdleTimeout != "20m" {
		t.Errorf("IdleTimeout = %q, want 20m", cfg.IdleTimeout)
	}
	if cfg.MemoryDir == "" {
		t.Error("MemoryDir should not be empty in DefaultConfig")
	}
}

func TestDefaultConfigMemoryDirConsistency(t *testing.T) {
	cfg := DefaultConfig()
	// MemoryDir should be a subdirectory of DataDir
	dataDir := DataDir()
	if !strings.HasPrefix(cfg.MemoryDir, dataDir) {
		t.Errorf("MemoryDir %q should be under DataDir %q", cfg.MemoryDir, dataDir)
	}
}

func TestDataDirEnvVar(t *testing.T) {
	tmpDir := t.TempDir()
	t.Setenv("TANRENAI_DATA_DIR", tmpDir)

	got := DataDir()
	if got != tmpDir {
		t.Errorf("DataDir() = %q, want %q", got, tmpDir)
	}
}

func TestDataDirDefaultNoEnvVar(t *testing.T) {
	// Ensure env var is unset
	t.Setenv("TANRENAI_DATA_DIR", "")

	got := DataDir()
	if got == "" {
		t.Fatal("DataDir() returned empty string without env var")
	}
	// Should end with "tanrenai"
	base := filepath.Base(got)
	if base != "tanrenai" {
		t.Errorf("DataDir() base = %q, want tanrenai", base)
	}
}

func TestMemoryDir(t *testing.T) {
	tmpDir := t.TempDir()
	t.Setenv("TANRENAI_DATA_DIR", tmpDir)

	got := MemoryDir()
	want := filepath.Join(tmpDir, "memory")
	if got != want {
		t.Errorf("MemoryDir() = %q, want %q", got, want)
	}
}

func TestMemoryDirIsUnderDataDir(t *testing.T) {
	memDir := MemoryDir()
	dataDir := DataDir()

	if !strings.HasPrefix(memDir, dataDir) {
		t.Errorf("MemoryDir %q should be a subdirectory of DataDir %q", memDir, dataDir)
	}

	// Should be exactly <DataDir>/memory
	want := filepath.Join(dataDir, "memory")
	if memDir != want {
		t.Errorf("MemoryDir() = %q, want %q", memDir, want)
	}
}

func TestEnsureDirsMemoryDisabled(t *testing.T) {
	tmpDir := t.TempDir()
	t.Setenv("TANRENAI_DATA_DIR", tmpDir)

	cfg := &Config{
		MemoryEnabled: false,
		MemoryDir:     filepath.Join(tmpDir, "memory"),
	}

	if err := EnsureDirs(cfg); err != nil {
		t.Fatalf("EnsureDirs: %v", err)
	}

	// DataDir should exist
	if _, err := os.Stat(tmpDir); err != nil {
		t.Errorf("DataDir not created: %v", err)
	}

	// MemoryDir should NOT be created (memory disabled)
	memDir := filepath.Join(tmpDir, "memory")
	if _, err := os.Stat(memDir); err == nil {
		t.Error("MemoryDir should not have been created when MemoryEnabled=false")
	}
}

func TestEnsureDirsMemoryEnabled(t *testing.T) {
	tmpDir := t.TempDir()
	t.Setenv("TANRENAI_DATA_DIR", tmpDir)

	memDir := filepath.Join(tmpDir, "memory")
	cfg := &Config{
		MemoryEnabled: true,
		MemoryDir:     memDir,
	}

	if err := EnsureDirs(cfg); err != nil {
		t.Fatalf("EnsureDirs: %v", err)
	}

	// Both DataDir and MemoryDir should exist
	if _, err := os.Stat(tmpDir); err != nil {
		t.Errorf("DataDir not created: %v", err)
	}
	if _, err := os.Stat(memDir); err != nil {
		t.Errorf("MemoryDir not created: %v", err)
	}
}

func TestEnsureDirsCreatesNestedDirs(t *testing.T) {
	tmpDir := t.TempDir()
	// Use a nested path that doesn't exist yet
	deepDir := filepath.Join(tmpDir, "a", "b", "c", "tanrenai")
	t.Setenv("TANRENAI_DATA_DIR", deepDir)

	memDir := filepath.Join(deepDir, "memory")
	cfg := &Config{
		MemoryEnabled: true,
		MemoryDir:     memDir,
	}

	if err := EnsureDirs(cfg); err != nil {
		t.Fatalf("EnsureDirs: %v", err)
	}

	if _, err := os.Stat(deepDir); err != nil {
		t.Errorf("nested DataDir not created: %v", err)
	}
	if _, err := os.Stat(memDir); err != nil {
		t.Errorf("nested MemoryDir not created: %v", err)
	}
}

func TestEnsureDirsIdempotent(t *testing.T) {
	tmpDir := t.TempDir()
	t.Setenv("TANRENAI_DATA_DIR", tmpDir)

	cfg := &Config{
		MemoryEnabled: true,
		MemoryDir:     filepath.Join(tmpDir, "memory"),
	}

	// Call twice — should not error
	if err := EnsureDirs(cfg); err != nil {
		t.Fatalf("first EnsureDirs: %v", err)
	}
	if err := EnsureDirs(cfg); err != nil {
		t.Fatalf("second EnsureDirs: %v", err)
	}
}
