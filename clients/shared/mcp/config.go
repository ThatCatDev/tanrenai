// Package mcp wires Model Context Protocol clients into the tanrenai
// agent's tool registry. Tanrenai connects to one or more external MCP
// servers (stdio subprocess or HTTP/SSE), discovers their tools at
// session start, and registers each as a tools.Tool so the agent can
// call them like any built-in.
//
// Config layering mirrors Claude Code:
//   - ~/.config/tanrenai/mcp.json     — user-scoped, applies everywhere
//   - <workspace>/.tanrenai/mcp.json  — project-scoped, committed
//
// Project entries override user entries by server name on collision.
package mcp

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
)

// Config is the parsed shape of a tanrenai mcp.json. Mirrors the
// Claude Code / Claude Desktop schema so users can copy-paste an
// existing `mcpServers` block.
type Config struct {
	// Servers keyed by user-visible name (e.g. "playwright", "github").
	// Each server is either a stdio subprocess (Command set) or an HTTP
	// endpoint (URL set) — never both. Validate() rejects the ambiguous
	// shape so we surface bad config at load time rather than mid-turn.
	Servers map[string]ServerConfig `json:"mcpServers"`
}

// ServerConfig describes a single MCP server to connect to.
type ServerConfig struct {
	// stdio transport — present when Command is non-empty.
	Command string            `json:"command,omitempty"`
	Args    []string          `json:"args,omitempty"`
	Env     map[string]string `json:"env,omitempty"`

	// HTTP transport — present when URL is non-empty. Headers are sent
	// on every request (typical use: Authorization).
	URL     string            `json:"url,omitempty"`
	Headers map[string]string `json:"headers,omitempty"`

	// Disabled servers are parsed but skipped at connect time. Useful
	// when committing a project mcp.json that includes optional servers
	// teammates can flip on locally without editing the file.
	Disabled bool `json:"disabled,omitempty"`
}

// Transport reports which transport this server config uses.
// Defined here (not derived per call site) so the loader's validation
// and the connect path agree on the same classification.
type Transport string

const (
	TransportStdio   Transport = "stdio"
	TransportHTTP    Transport = "http"
	TransportUnknown Transport = "unknown"
)

// Transport classifies a ServerConfig based on which fields are set.
func (s ServerConfig) Transport() Transport {
	switch {
	case s.URL != "" && s.Command == "":
		return TransportHTTP
	case s.Command != "" && s.URL == "":
		return TransportStdio
	default:
		return TransportUnknown
	}
}

// Validate checks the config for structural problems. Returns a joined
// error listing every malformed server so users see all problems at
// once rather than one-per-edit.
func (c Config) Validate() error {
	var errs []error
	for name, s := range c.Servers {
		switch s.Transport() {
		case TransportStdio, TransportHTTP:
			// fine
		case TransportUnknown:
			if s.Command == "" && s.URL == "" {
				errs = append(errs, fmt.Errorf("server %q: must set either `command` (stdio) or `url` (http)", name))
			} else {
				errs = append(errs, fmt.Errorf("server %q: cannot set both `command` and `url` — pick one transport", name))
			}
		}
	}
	return errors.Join(errs...)
}

// Load reads project + user config and returns the merged result.
// Project entries override user entries on name collision; missing
// files are treated as empty configs (not errors) so a fresh project
// works without setup.
//
// `workspace` is the project root (.tanrenai/mcp.json is read relative
// to it). Empty string skips the project layer — useful for the CLI
// case where no workspace is set.
func Load(workspace string) (Config, error) {
	user, err := loadUserConfig()
	if err != nil {
		return Config{}, fmt.Errorf("user mcp.json: %w", err)
	}
	project, err := loadProjectConfig(workspace)
	if err != nil {
		return Config{}, fmt.Errorf("project mcp.json: %w", err)
	}

	// Start with user entries; project overrides by name.
	merged := Config{Servers: map[string]ServerConfig{}}
	for k, v := range user.Servers {
		merged.Servers[k] = v
	}
	for k, v := range project.Servers {
		merged.Servers[k] = v
	}

	if err := merged.Validate(); err != nil {
		return Config{}, err
	}
	return merged, nil
}

func loadProjectConfig(workspace string) (Config, error) {
	if workspace == "" {
		return Config{}, nil
	}
	return loadFile(ProjectConfigPath(workspace))
}

func loadUserConfig() (Config, error) {
	path, err := UserConfigPath()
	if err != nil {
		// No HOME → no user config. Not an error worth blocking start;
		// the agent simply won't have user-scoped MCP servers.
		return Config{}, nil
	}
	return loadFile(path)
}

// ProjectConfigPath returns the canonical project-scoped mcp.json path
// for a given workspace root. Surface for the `tanrenai mcp` CLI
// subcommands which need to read AND write the same file the agent
// reads at session startup.
func ProjectConfigPath(workspace string) string {
	return filepath.Join(workspace, ".tanrenai", "mcp.json")
}

// UserConfigPath returns the per-user mcp.json path. Errors only when
// the platform doesn't expose a user-config dir (rare; e.g. no HOME
// set).
func UserConfigPath() (string, error) {
	dir, err := userConfigDir()
	if err != nil {
		return "", err
	}
	return filepath.Join(dir, "tanrenai", "mcp.json"), nil
}

// LoadFile reads + parses one config file at an explicit path. Exposed
// for CLI subcommands that operate on a single scope (project OR user)
// rather than the merged view Load returns. A missing file returns an
// empty config without error so `tanrenai mcp add` works on a fresh
// project where no file exists yet.
func LoadFile(path string) (Config, error) {
	return loadFile(path)
}

// SaveFile writes a config to disk, creating the parent directory if
// needed. Uses 0o644 so committed project configs are readable to the
// team; user config inherits the same mode (still under HOME, fine).
func SaveFile(path string, cfg Config) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return fmt.Errorf("create config dir: %w", err)
	}
	// Marshal with two-space indent — these files are meant to be
	// hand-editable and reviewable in git, not minified.
	data, err := json.MarshalIndent(cfg, "", "  ")
	if err != nil {
		return fmt.Errorf("marshal config: %w", err)
	}
	// Trailing newline so the file plays nicely with editors that
	// auto-strip end-of-file whitespace.
	data = append(data, '\n')
	if err := os.WriteFile(path, data, 0o644); err != nil {
		return fmt.Errorf("write %s: %w", path, err)
	}
	return nil
}

// userConfigDir returns the platform's per-user config directory.
// Wrapped so tests can stub it via package-level override.
var userConfigDir = func() (string, error) {
	return os.UserConfigDir()
}

// loadFile reads + parses one mcp.json. A missing file returns an
// empty config without error — the file is optional.
func loadFile(path string) (Config, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return Config{}, nil
		}
		return Config{}, err
	}
	var cfg Config
	if err := json.Unmarshal(raw, &cfg); err != nil {
		return Config{}, fmt.Errorf("parse %s: %w", path, err)
	}
	return cfg, nil
}
