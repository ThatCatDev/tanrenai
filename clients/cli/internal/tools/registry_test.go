package tools

import (
	"context"
	"encoding/json"
	"testing"
)

// mockTool is a minimal Tool implementation for testing.
type mockTool struct {
	name string
}

func (m *mockTool) Name() string        { return m.name }
func (m *mockTool) Description() string  { return "mock tool" }
func (m *mockTool) Parameters() json.RawMessage { return json.RawMessage(`{"type":"object"}`) }
func (m *mockTool) Execute(_ context.Context, _ string) (*ToolResult, error) {
	return &ToolResult{Output: "ok"}, nil
}

func TestRegistryRegisterAndGet(t *testing.T) {
	r := NewRegistry()
	r.Register(&mockTool{name: "test_tool"})

	got := r.Get("test_tool")
	if got == nil {
		t.Fatal("expected to find registered tool")
	}
	if got.Name() != "test_tool" {
		t.Errorf("got name %q, want %q", got.Name(), "test_tool")
	}
}

func TestRegistryGetUnknown(t *testing.T) {
	r := NewRegistry()
	if r.Get("nonexistent") != nil {
		t.Error("expected nil for unregistered tool")
	}
}

func TestRegistryDuplicatePanics(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic on duplicate registration")
		}
	}()

	r := NewRegistry()
	r.Register(&mockTool{name: "dup"})
	r.Register(&mockTool{name: "dup"})
}

func TestRegistryAPITools(t *testing.T) {
	r := NewRegistry()
	r.Register(&mockTool{name: "alpha"})
	r.Register(&mockTool{name: "beta"})

	apiTools := r.APITools()
	if len(apiTools) != 2 {
		t.Fatalf("got %d tools, want 2", len(apiTools))
	}
	if apiTools[0].Function.Name != "alpha" {
		t.Errorf("first tool name = %q, want %q", apiTools[0].Function.Name, "alpha")
	}
	if apiTools[1].Function.Name != "beta" {
		t.Errorf("second tool name = %q, want %q", apiTools[1].Function.Name, "beta")
	}
	for _, tool := range apiTools {
		if tool.Type != "function" {
			t.Errorf("tool type = %q, want %q", tool.Type, "function")
		}
	}
}

func TestDefaultRegistryHasAllTools(t *testing.T) {
	r := DefaultRegistry()
	expected := []string{"file_read", "file_write", "list_dir", "shell_exec"}
	for _, name := range expected {
		if r.Get(name) == nil {
			t.Errorf("DefaultRegistry missing tool %q", name)
		}
	}
}
