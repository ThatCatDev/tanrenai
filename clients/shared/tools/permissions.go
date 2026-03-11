package tools

import (
	"encoding/json"
	"os"
	"path/filepath"
	"sync"
)

// Permissions manages tool approval state. It merges a global config
// (~/.tanrenai/permissions.json) with a project-local config
// (.tanrenai/permissions.json in the current directory). "Always allow"
// writes to the local file so permissions are project-scoped.
type Permissions struct {
	AllowedTools map[string]bool `json:"allowed_tools"`
	mu           sync.RWMutex
	path         string // local project path (written to on Allow)
}

func globalConfigDir() string {
	if d := os.Getenv("TANRENAI_CONFIG_DIR"); d != "" {
		return d
	}
	home, _ := os.UserHomeDir()
	return filepath.Join(home, ".tanrenai")
}

// LoadPermissions loads permissions from both global (~/.tanrenai/) and
// local (.tanrenai/ in cwd) config files, merging them. Writes go to the
// local project config.
func LoadPermissions() *Permissions {
	p := &Permissions{
		AllowedTools: make(map[string]bool),
		path:         filepath.Join(".tanrenai", "permissions.json"),
	}

	// Load global config first.
	globalPath := filepath.Join(globalConfigDir(), "permissions.json")
	if data, err := os.ReadFile(globalPath); err == nil {
		var global Permissions
		if json.Unmarshal(data, &global) == nil && global.AllowedTools != nil {
			for k, v := range global.AllowedTools {
				p.AllowedTools[k] = v
			}
		}
	}

	// Load local (project) config, overriding global.
	if data, err := os.ReadFile(p.path); err == nil {
		var local Permissions
		if json.Unmarshal(data, &local) == nil && local.AllowedTools != nil {
			for k, v := range local.AllowedTools {
				p.AllowedTools[k] = v
			}
		}
	}

	return p
}

// IsAllowed returns true if the tool has been permanently approved.
func (p *Permissions) IsAllowed(toolName string) bool {
	p.mu.RLock()
	defer p.mu.RUnlock()
	return p.AllowedTools[toolName]
}

// Allow permanently approves a tool and saves to disk.
func (p *Permissions) Allow(toolName string) error {
	p.mu.Lock()
	p.AllowedTools[toolName] = true
	p.mu.Unlock()
	return p.save()
}

func (p *Permissions) save() error {
	p.mu.RLock()
	defer p.mu.RUnlock()
	if err := os.MkdirAll(filepath.Dir(p.path), 0755); err != nil {
		return err
	}
	data, err := json.MarshalIndent(p, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(p.path, data, 0644)
}
