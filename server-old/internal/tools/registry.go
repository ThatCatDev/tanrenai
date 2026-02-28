package tools

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/ThatCatDev/tanrenai/server/pkg/api"
)

// Tool is the interface that all built-in tools implement.
type Tool interface {
	Name() string
	Description() string
	Parameters() json.RawMessage
	Execute(ctx context.Context, arguments string) (*ToolResult, error)
}

// Registry holds a set of tools keyed by name.
type Registry struct {
	tools map[string]Tool
	order []string
}

// NewRegistry creates an empty Registry.
func NewRegistry() *Registry {
	return &Registry{
		tools: make(map[string]Tool),
	}
}

// Register adds a tool to the registry. Panics on duplicate names.
func (r *Registry) Register(t Tool) {
	name := t.Name()
	if _, exists := r.tools[name]; exists {
		panic(fmt.Sprintf("tools: duplicate tool name %q", name))
	}
	r.tools[name] = t
	r.order = append(r.order, name)
}

// Get looks up a tool by name. Returns nil if not found.
func (r *Registry) Get(name string) Tool {
	return r.tools[name]
}

// APITools returns the tools in OpenAI API format for inclusion in requests.
func (r *Registry) APITools() []api.Tool {
	out := make([]api.Tool, 0, len(r.order))
	for _, name := range r.order {
		t := r.tools[name]
		out = append(out, api.Tool{
			Type: "function",
			Function: api.ToolFunction{
				Name:        t.Name(),
				Description: t.Description(),
				Parameters:  t.Parameters(),
			},
		})
	}
	return out
}

// DefaultRegistry returns a registry pre-loaded with all built-in tools.
func DefaultRegistry() *Registry {
	r := NewRegistry()
	r.Register(&FileReadTool{})
	r.Register(&FileWriteTool{})
	r.Register(&PatchFileTool{})
	r.Register(&ListDirTool{})
	r.Register(&FindFilesTool{})
	r.Register(&GrepSearchTool{})
	r.Register(&GitInfoTool{})
	r.Register(&ShellExecTool{})
	r.Register(&WebSearchTool{})
	return r
}
