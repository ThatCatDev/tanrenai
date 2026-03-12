package tools

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"sync"
)

// PermissionRule defines what is allowed for a tool.
type PermissionRule struct {
	Tool string `json:"tool"`
	// If empty, blanket-allow the tool. Otherwise, only allow when
	// the argument value matches one of these exact strings.
	// Key is the JSON argument field name (e.g. "command", "path").
	AllowedArgs map[string][]string `json:"allowed_args,omitempty"`
}

// PermissionsConfig is the on-disk format.
type PermissionsConfig struct {
	Rules []PermissionRule `json:"rules"`
}

// Permissions manages tool approval state. It merges a global config
// (~/.tanrenai/permissions.json) with a project-local config
// (.tanrenai/permissions.json in the current directory). "Always allow"
// writes to the local file so permissions are project-scoped.
type Permissions struct {
	config PermissionsConfig
	mu     sync.RWMutex
	path   string // local project path (written to on Allow)
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
		path: filepath.Join(".tanrenai", "permissions.json"),
	}

	// Load global config first.
	globalPath := filepath.Join(globalConfigDir(), "permissions.json")
	if data, err := os.ReadFile(globalPath); err == nil {
		var cfg PermissionsConfig
		if json.Unmarshal(data, &cfg) == nil {
			p.config.Rules = append(p.config.Rules, cfg.Rules...)
		}
	}

	// Load local (project) config, appending after global.
	if data, err := os.ReadFile(p.path); err == nil {
		var cfg PermissionsConfig
		if json.Unmarshal(data, &cfg) == nil {
			p.config.Rules = append(p.config.Rules, cfg.Rules...)
		}
	}

	return p
}

// IsAllowed checks if a tool call (with its arguments) is permitted by
// any existing rule.
func (p *Permissions) IsAllowed(toolName string, argsJSON string) bool {
	p.mu.RLock()
	defer p.mu.RUnlock()

	for _, rule := range p.config.Rules {
		if rule.Tool != toolName {
			continue
		}
		// Blanket allow — no argument constraints.
		if len(rule.AllowedArgs) == 0 {
			return true
		}
		// Check each constraint: all must match.
		if p.argsMatch(rule.AllowedArgs, argsJSON) {
			return true
		}
	}
	return false
}

// argsMatch returns true if the parsed arguments satisfy all constraints.
func (p *Permissions) argsMatch(constraints map[string][]string, argsJSON string) bool {
	if argsJSON == "" {
		return false
	}
	var parsed map[string]interface{}
	if json.Unmarshal([]byte(argsJSON), &parsed) != nil {
		return false
	}

	for key, allowed := range constraints {
		val, ok := parsed[key]
		if !ok {
			return false
		}
		strVal, ok := val.(string)
		if !ok {
			return false
		}
		matched := false
		for _, pattern := range allowed {
			if matchPattern(pattern, strVal) {
				matched = true
				break
			}
		}
		if !matched {
			return false
		}
	}
	return true
}

// matchPattern supports exact match and simple prefix glob (e.g. "ls *").
func matchPattern(pattern, value string) bool {
	if pattern == value {
		return true
	}
	// Support trailing wildcard: "ls *" matches "ls -la", "ls /tmp"
	if strings.HasSuffix(pattern, " *") {
		prefix := strings.TrimSuffix(pattern, " *")
		return strings.HasPrefix(value, prefix+" ") || value == prefix
	}
	// Support path prefix: "/home/user/project/*" matches files under that dir
	if strings.HasSuffix(pattern, "/*") {
		prefix := strings.TrimSuffix(pattern, "/*")
		return strings.HasPrefix(value, prefix+"/") || value == prefix
	}
	return false
}

// AllowTool adds a blanket allow rule for a tool.
func (p *Permissions) AllowTool(toolName string) error {
	p.mu.Lock()
	p.config.Rules = append(p.config.Rules, PermissionRule{Tool: toolName})
	p.mu.Unlock()
	return p.save()
}

// AllowToolWithArgs adds a rule that allows a tool only when specific
// argument values match. For shell_exec this saves the exact command;
// for file tools this saves the exact path.
func (p *Permissions) AllowToolWithArgs(toolName string, args map[string][]string) error {
	p.mu.Lock()
	p.config.Rules = append(p.config.Rules, PermissionRule{
		Tool:        toolName,
		AllowedArgs: args,
	})
	p.mu.Unlock()
	return p.save()
}

func (p *Permissions) save() error {
	p.mu.RLock()
	// Only save rules that were added to local (not global).
	// For simplicity, save all rules to local — the user can edit the file.
	data, err := json.MarshalIndent(p.config, "", "  ")
	p.mu.RUnlock()
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(p.path), 0755); err != nil {
		return err
	}
	return os.WriteFile(p.path, data, 0644)
}

// ToolRiskLevel classifies tools by how dangerous they are.
type ToolRiskLevel int

const (
	RiskReadOnly  ToolRiskLevel = iota // safe to blanket-allow
	RiskWrite                          // modifies files — show path
	RiskExecute                        // runs arbitrary code — show full command
	RiskNetwork                        // network access — show details
)

// ToolRisk returns the risk level for a tool, which determines how
// granular the approval should be.
func ToolRisk(toolName string) ToolRiskLevel {
	switch toolName {
	case "file_read", "list_dir", "find_files", "grep_search", "git_info":
		return RiskReadOnly
	case "file_write", "patch_file":
		return RiskWrite
	case "shell_exec":
		return RiskExecute
	case "web_search":
		return RiskNetwork
	default:
		return RiskExecute // unknown tools default to highest scrutiny
	}
}

// ApprovalKey returns the argument field name that should be used for
// granular permissions (e.g. "command" for shell_exec, "path" for file tools).
func ApprovalKey(toolName string) string {
	switch toolName {
	case "shell_exec":
		return "command"
	case "file_write", "patch_file", "file_read":
		return "path"
	default:
		return ""
	}
}

// ExtractArg parses the JSON arguments and returns the value of the given key.
func ExtractArg(argsJSON, key string) string {
	if key == "" || argsJSON == "" {
		return ""
	}
	var parsed map[string]interface{}
	if json.Unmarshal([]byte(argsJSON), &parsed) != nil {
		return ""
	}
	if val, ok := parsed[key]; ok {
		if s, ok := val.(string); ok {
			return s
		}
	}
	return ""
}

// CommandPrefix returns the base command from a shell command string.
// e.g. "ls -la /tmp" → "ls", "git status" → "git"
func CommandPrefix(cmd string) string {
	parts := strings.Fields(cmd)
	if len(parts) == 0 {
		return ""
	}
	return parts[0]
}
