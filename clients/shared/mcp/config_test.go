package mcp

import (
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestServerConfig_Transport(t *testing.T) {
	tests := []struct {
		name string
		cfg  ServerConfig
		want Transport
	}{
		{"stdio", ServerConfig{Command: "npx", Args: []string{"-y", "foo"}}, TransportStdio},
		{"http", ServerConfig{URL: "https://example.com/mcp"}, TransportHTTP},
		{"both — invalid", ServerConfig{Command: "x", URL: "y"}, TransportUnknown},
		{"neither — invalid", ServerConfig{}, TransportUnknown},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := tc.cfg.Transport(); got != tc.want {
				t.Errorf("Transport() = %v, want %v", got, tc.want)
			}
		})
	}
}

func TestConfig_Validate(t *testing.T) {
	cfg := Config{Servers: map[string]ServerConfig{
		"ok-stdio":     {Command: "npx"},
		"ok-http":      {URL: "https://x"},
		"bad-both":     {Command: "x", URL: "y"},
		"bad-neither":  {},
	}}
	err := cfg.Validate()
	if err == nil {
		t.Fatal("expected validation error")
	}
	// Both invalid entries surface — Validate joins errors so users
	// see every problem at once instead of fix-one-discover-next.
	msg := err.Error()
	if !strings.Contains(msg, "bad-both") {
		t.Errorf("missing bad-both in error: %s", msg)
	}
	if !strings.Contains(msg, "bad-neither") {
		t.Errorf("missing bad-neither in error: %s", msg)
	}
}

func TestLoad_LayersProjectOverUser(t *testing.T) {
	// Set up a fake user config dir + a fake project dir, write two
	// mcp.json files that share a server name. Project should win.
	tmpUser := t.TempDir()
	tmpProject := t.TempDir()
	userConfigDir = func() (string, error) { return tmpUser, nil }
	t.Cleanup(func() { userConfigDir = os.UserConfigDir })

	if err := os.MkdirAll(filepath.Join(tmpUser, "tanrenai"), 0o755); err != nil {
		t.Fatal(err)
	}
	userJSON := `{"mcpServers":{"shared":{"command":"user-cmd"},"only-user":{"command":"u"}}}`
	if err := os.WriteFile(filepath.Join(tmpUser, "tanrenai", "mcp.json"), []byte(userJSON), 0o644); err != nil {
		t.Fatal(err)
	}

	if err := os.MkdirAll(filepath.Join(tmpProject, ".tanrenai"), 0o755); err != nil {
		t.Fatal(err)
	}
	projectJSON := `{"mcpServers":{"shared":{"command":"project-cmd"},"only-project":{"command":"p"}}}`
	if err := os.WriteFile(filepath.Join(tmpProject, ".tanrenai", "mcp.json"), []byte(projectJSON), 0o644); err != nil {
		t.Fatal(err)
	}

	cfg, err := Load(tmpProject)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if len(cfg.Servers) != 3 {
		t.Errorf("got %d servers, want 3", len(cfg.Servers))
	}
	if cfg.Servers["shared"].Command != "project-cmd" {
		t.Errorf("shared server: project entry should override user; got Command=%q", cfg.Servers["shared"].Command)
	}
	if _, ok := cfg.Servers["only-user"]; !ok {
		t.Error("missing only-user — user entries without project overrides should pass through")
	}
	if _, ok := cfg.Servers["only-project"]; !ok {
		t.Error("missing only-project")
	}
}

func TestLoad_MissingFilesAreNotErrors(t *testing.T) {
	// Empty user dir + empty workspace — neither file exists. Load
	// should return an empty config, not an error, so fresh checkouts
	// work without setup.
	tmpUser := t.TempDir()
	userConfigDir = func() (string, error) { return tmpUser, nil }
	t.Cleanup(func() { userConfigDir = os.UserConfigDir })

	cfg, err := Load(t.TempDir())
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if len(cfg.Servers) != 0 {
		t.Errorf("expected empty config, got %d servers", len(cfg.Servers))
	}
}

func TestLoad_MalformedJSONErrors(t *testing.T) {
	// A typo in mcp.json should surface as a parse error, not silently
	// become "no servers configured". The error must name the file
	// path so users know which one to fix.
	tmpProject := t.TempDir()
	if err := os.MkdirAll(filepath.Join(tmpProject, ".tanrenai"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(tmpProject, ".tanrenai", "mcp.json"), []byte("{not-json"), 0o644); err != nil {
		t.Fatal(err)
	}
	tmpUser := t.TempDir()
	userConfigDir = func() (string, error) { return tmpUser, nil }
	t.Cleanup(func() { userConfigDir = os.UserConfigDir })

	_, err := Load(tmpProject)
	if err == nil {
		t.Fatal("expected parse error")
	}
	if !strings.Contains(err.Error(), "mcp.json") {
		t.Errorf("error should reference the file path: %v", err)
	}
}

func TestLoad_InvalidServerFailsValidation(t *testing.T) {
	// Even a successfully-parsed config gets rejected at Load time if
	// any server is malformed. Users discover both at once rather
	// than load → call → mid-turn failure.
	tmpProject := t.TempDir()
	tmpUser := t.TempDir()
	userConfigDir = func() (string, error) { return tmpUser, nil }
	t.Cleanup(func() { userConfigDir = os.UserConfigDir })

	if err := os.MkdirAll(filepath.Join(tmpProject, ".tanrenai"), 0o755); err != nil {
		t.Fatal(err)
	}
	body := `{"mcpServers":{"bad":{"command":"c","url":"u"}}}`
	if err := os.WriteFile(filepath.Join(tmpProject, ".tanrenai", "mcp.json"), []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}

	_, err := Load(tmpProject)
	if err == nil || !strings.Contains(err.Error(), "cannot set both") {
		t.Fatalf("expected validation error, got %v", err)
	}
	// Sanity: errors.Is doesn't have to match a sentinel, but the
	// returned error should not be a not-found.
	if errors.Is(err, os.ErrNotExist) {
		t.Errorf("unexpected not-exist error: %v", err)
	}
}
