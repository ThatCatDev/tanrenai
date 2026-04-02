package cmd

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/ThatCatDev/tanrenai/client/internal/chatctx"
	"github.com/ThatCatDev/tanrenai/shared/pkg/api"
)

// ── truncate ──────────────────────────────────────────────────────────

func TestTruncate(t *testing.T) {
	tests := []struct {
		input string
		max   int
		want  string
	}{
		{"hello", 10, "hello"},
		{"hello", 5, "hello"},
		{"hello world", 5, "hello..."},
		{"", 5, ""},
		{"ab", 1, "a..."},
	}
	for _, tt := range tests {
		got := truncate(tt.input, tt.max)
		if got != tt.want {
			t.Errorf("truncate(%q, %d) = %q, want %q", tt.input, tt.max, got, tt.want)
		}
	}
}

// ── formatBytes ───────────────────────────────────────────────────────

func TestFormatBytes(t *testing.T) {
	tests := []struct {
		input int64
		want  string
	}{
		{0, "0 B"},
		{500, "500 B"},
		{1024 * 1024, "1.0 MB"},
		{5 * 1024 * 1024, "5.0 MB"},
		{1024 * 1024 * 1024, "1.0 GB"},
		{int64(2.5 * 1024 * 1024 * 1024), "2.5 GB"},
	}
	for _, tt := range tests {
		got := formatBytes(tt.input)
		if got != tt.want {
			t.Errorf("formatBytes(%d) = %q, want %q", tt.input, got, tt.want)
		}
	}
}

// ── tanrenaiDataDir ───────────────────────────────────────────────────

func TestTanrenaiDataDir_EnvOverride(t *testing.T) {
	t.Setenv("TANRENAI_DATA_DIR", "/custom/data")
	got := tanrenaiDataDir()
	if got != "/custom/data" {
		t.Errorf("got %q, want /custom/data", got)
	}
}

func TestTanrenaiDataDir_Default(t *testing.T) {
	t.Setenv("TANRENAI_DATA_DIR", "")
	got := tanrenaiDataDir()
	if !strings.Contains(got, "tanrenai") {
		t.Errorf("expected path containing 'tanrenai', got %q", got)
	}
}

// ── projectMemoryDir ──────────────────────────────────────────────────

func TestProjectMemoryDir_ContainsHash(t *testing.T) {
	t.Setenv("TANRENAI_DATA_DIR", "/tmp/test-tanrenai")
	dir := projectMemoryDir()
	if !strings.HasPrefix(dir, "/tmp/test-tanrenai/memory/") {
		t.Errorf("unexpected dir: %q", dir)
	}
	// Hash part should be 16 hex chars
	hash := filepath.Base(dir)
	if len(hash) != 16 {
		t.Errorf("hash part %q should be 16 chars", hash)
	}
}

// ── startupLog ────────────────────────────────────────────────────────

func TestStartupLog_InfoNoTUI(t *testing.T) {
	// When tui is nil, Info writes to stdout. We can't capture stdout
	// easily in a unit test, but we verify it doesn't panic.
	log := &startupLog{tui: nil}
	log.Info("test message")
}

func TestStartupLog_WarnNoTUI(t *testing.T) {
	log := &startupLog{tui: nil}
	log.Warn("test warning")
}

// ── loadContextFile ───────────────────────────────────────────────────

func TestLoadContextFile_Success(t *testing.T) {
	tmp := t.TempDir()
	path := filepath.Join(tmp, "test.txt")
	os.WriteFile(path, []byte("file content here"), 0644)

	estimator := chatctx.NewTokenEstimator()
	mgr := chatctx.NewManager(chatctx.Config{CtxSize: 4096, ResponseBudget: 512}, estimator)

	msg, err := loadContextFile(mgr, path)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !strings.Contains(msg, "17 bytes") {
		t.Errorf("expected byte count in msg, got %q", msg)
	}
	files := mgr.ContextFiles()
	if len(files) != 1 {
		t.Errorf("expected 1 context file, got %d", len(files))
	}
}

func TestLoadContextFile_NotFound(t *testing.T) {
	estimator := chatctx.NewTokenEstimator()
	mgr := chatctx.NewManager(chatctx.Config{CtxSize: 4096, ResponseBudget: 512}, estimator)

	_, err := loadContextFile(mgr, "/nonexistent/file.txt")
	if err == nil {
		t.Error("expected error for nonexistent file")
	}
}

// ── handleREPLCommand ─────────────────────────────────────────────────

func TestHandleREPLCommand_Clear(t *testing.T) {
	var buf bytes.Buffer
	estimator := chatctx.NewTokenEstimator()
	mgr := chatctx.NewManager(chatctx.Config{CtxSize: 4096, ResponseBudget: 512}, estimator)

	handled := handleREPLCommand(&buf, "/clear", mgr, nil, false)
	if !handled {
		t.Error("expected /clear to be handled")
	}
	if !strings.Contains(buf.String(), "History cleared") {
		t.Errorf("unexpected output: %q", buf.String())
	}
}

func TestHandleREPLCommand_Tokens(t *testing.T) {
	var buf bytes.Buffer
	estimator := chatctx.NewTokenEstimator()
	mgr := chatctx.NewManager(chatctx.Config{CtxSize: 4096, ResponseBudget: 512}, estimator)

	handled := handleREPLCommand(&buf, "/tokens", mgr, nil, false)
	if !handled {
		t.Error("expected /tokens to be handled")
	}
	if !strings.Contains(buf.String(), "Token budget") {
		t.Errorf("unexpected output: %q", buf.String())
	}
}

func TestHandleREPLCommand_Help(t *testing.T) {
	var buf bytes.Buffer
	handled := handleREPLCommand(&buf, "/help", nil, nil, false)
	if !handled {
		t.Error("expected /help to be handled")
	}
	if !strings.Contains(buf.String(), "/clear") {
		t.Errorf("help should list commands, got %q", buf.String())
	}
}

func TestHandleREPLCommand_ContextAdd(t *testing.T) {
	tmp := t.TempDir()
	path := filepath.Join(tmp, "ctx.txt")
	os.WriteFile(path, []byte("context data"), 0644)

	var buf bytes.Buffer
	estimator := chatctx.NewTokenEstimator()
	mgr := chatctx.NewManager(chatctx.Config{CtxSize: 4096, ResponseBudget: 512}, estimator)

	handled := handleREPLCommand(&buf, "/context add "+path, mgr, nil, false)
	if !handled {
		t.Error("expected /context add to be handled")
	}
	if !strings.Contains(buf.String(), "Loaded context file") {
		t.Errorf("unexpected output: %q", buf.String())
	}
}

func TestHandleREPLCommand_ContextList(t *testing.T) {
	var buf bytes.Buffer
	estimator := chatctx.NewTokenEstimator()
	mgr := chatctx.NewManager(chatctx.Config{CtxSize: 4096, ResponseBudget: 512}, estimator)

	handled := handleREPLCommand(&buf, "/context list", mgr, nil, false)
	if !handled {
		t.Error("expected /context list to be handled")
	}
	if !strings.Contains(buf.String(), "No context files") {
		t.Errorf("unexpected output: %q", buf.String())
	}
}

func TestHandleREPLCommand_ContextClear(t *testing.T) {
	var buf bytes.Buffer
	estimator := chatctx.NewTokenEstimator()
	mgr := chatctx.NewManager(chatctx.Config{CtxSize: 4096, ResponseBudget: 512}, estimator)

	handled := handleREPLCommand(&buf, "/context clear", mgr, nil, false)
	if !handled {
		t.Error("expected /context clear to be handled")
	}
	if !strings.Contains(buf.String(), "Context files cleared") {
		t.Errorf("unexpected output: %q", buf.String())
	}
}

func TestHandleREPLCommand_MemoryWithoutEnabled(t *testing.T) {
	var buf bytes.Buffer
	handled := handleREPLCommand(&buf, "/memory", nil, nil, false)
	if !handled {
		t.Error("expected /memory to be handled")
	}
	if !strings.Contains(buf.String(), "not enabled") {
		t.Errorf("unexpected output: %q", buf.String())
	}
}

func TestHandleREPLCommand_MemorySearchWithoutEnabled(t *testing.T) {
	var buf bytes.Buffer
	handled := handleREPLCommand(&buf, "/memory search test", nil, nil, false)
	if !handled {
		t.Error("expected /memory search to be handled")
	}
	if !strings.Contains(buf.String(), "not enabled") {
		t.Errorf("unexpected output: %q", buf.String())
	}
}

func TestHandleREPLCommand_MemoryForgetWithoutEnabled(t *testing.T) {
	var buf bytes.Buffer
	handled := handleREPLCommand(&buf, "/memory forget abc", nil, nil, false)
	if !handled {
		t.Error("expected /memory forget to be handled")
	}
	if !strings.Contains(buf.String(), "not enabled") {
		t.Errorf("unexpected output: %q", buf.String())
	}
}

func TestHandleREPLCommand_MemoryClearWithoutEnabled(t *testing.T) {
	var buf bytes.Buffer
	handled := handleREPLCommand(&buf, "/memory clear", nil, nil, false)
	if !handled {
		t.Error("expected /memory clear to be handled")
	}
	if !strings.Contains(buf.String(), "not enabled") {
		t.Errorf("unexpected output: %q", buf.String())
	}
}

func TestHandleREPLCommand_Unknown(t *testing.T) {
	var buf bytes.Buffer
	handled := handleREPLCommand(&buf, "/unknown", nil, nil, false)
	if handled {
		t.Error("expected unknown command to not be handled")
	}
}

func TestHandleREPLCommand_ContextAddEmpty(t *testing.T) {
	var buf bytes.Buffer
	estimator := chatctx.NewTokenEstimator()
	mgr := chatctx.NewManager(chatctx.Config{CtxSize: 4096, ResponseBudget: 512}, estimator)

	handled := handleREPLCommand(&buf, "/context add ", mgr, nil, false)
	if !handled {
		t.Error("expected /context add (empty) to be handled")
	}
	if !strings.Contains(buf.String(), "Usage") {
		t.Errorf("expected usage message, got %q", buf.String())
	}
}

func TestHandleREPLCommand_MemorySearchEmpty(t *testing.T) {
	var buf bytes.Buffer
	handled := handleREPLCommand(&buf, "/memory search ", nil, nil, true)
	if !handled {
		t.Error("expected /memory search (empty) to be handled")
	}
	if !strings.Contains(buf.String(), "Usage") {
		t.Errorf("expected usage message, got %q", buf.String())
	}
}

func TestHandleREPLCommand_MemoryForgetEmpty(t *testing.T) {
	var buf bytes.Buffer
	handled := handleREPLCommand(&buf, "/memory forget ", nil, nil, true)
	if !handled {
		t.Error("expected /memory forget (empty) to be handled")
	}
	if !strings.Contains(buf.String(), "Usage") {
		t.Errorf("expected usage message, got %q", buf.String())
	}
}

// ── persistPipeMemory ─────────────────────────────────────────────────

func TestPersistPipeMemory_DisabledMemory(t *testing.T) {
	// Should return immediately without panic when memory disabled.
	deps := &sessionDeps{memoryEnabled: false}
	persistPipeMemory(nil, deps, nil) // no panic = pass
}

func TestPersistPipeMemory_NoAssistantContent(t *testing.T) {
	deps := &sessionDeps{memoryEnabled: true}
	msgs := []api.Message{{Role: "user", Content: "hello"}}
	persistPipeMemory(nil, deps, msgs) // no panic, skips store
}

func TestPersistPipeMemory_NoUserContent(t *testing.T) {
	deps := &sessionDeps{memoryEnabled: true}
	msgs := []api.Message{{Role: "assistant", Content: "world"}}
	persistPipeMemory(nil, deps, msgs) // no panic, skips store
}

// ── logDir ────────────────────────────────────────────────────────────

func TestLogDir_XDGOverride(t *testing.T) {
	t.Setenv("XDG_STATE_HOME", "/tmp/xdg-state")
	got := logDir()
	if got != "/tmp/xdg-state/tanrenai" {
		t.Errorf("got %q, want /tmp/xdg-state/tanrenai", got)
	}
}

func TestLogDir_Default(t *testing.T) {
	t.Setenv("XDG_STATE_HOME", "")
	got := logDir()
	if !strings.Contains(got, "tanrenai") {
		t.Errorf("expected path containing 'tanrenai', got %q", got)
	}
}

// ── cleanupOldLogs ────────────────────────────────────────────────────

func TestCleanupOldLogs_RemovesExcess(t *testing.T) {
	dir := t.TempDir()
	// Create 5 rotated logs (maxLogFiles-1 = 2, so 3 should be removed)
	for _, name := range []string{
		"tanrenai.1000.log",
		"tanrenai.2000.log",
		"tanrenai.3000.log",
		"tanrenai.4000.log",
		"tanrenai.5000.log",
	} {
		os.WriteFile(filepath.Join(dir, name), []byte("log"), 0644)
	}
	// Also create the current log (should not be touched)
	os.WriteFile(filepath.Join(dir, "tanrenai.log"), []byte("current"), 0644)

	cleanupOldLogs(dir)

	entries, _ := os.ReadDir(dir)
	var rotated int
	for _, e := range entries {
		if e.Name() != "tanrenai.log" {
			rotated++
		}
	}
	// maxLogFiles=3, so maxLogFiles-1=2 rotated should remain
	if rotated != 2 {
		t.Errorf("expected 2 rotated logs remaining, got %d", rotated)
	}
}

func TestCleanupOldLogs_NothingToClean(t *testing.T) {
	dir := t.TempDir()
	os.WriteFile(filepath.Join(dir, "tanrenai.1000.log"), []byte("log"), 0644)

	cleanupOldLogs(dir)

	entries, _ := os.ReadDir(dir)
	if len(entries) != 1 {
		t.Errorf("expected 1 file, got %d", len(entries))
	}
}
