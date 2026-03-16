package tools

import (
	"context"
	"encoding/json"
	"testing"
)

// stubTool is a minimal Tool implementation for testing the registry.
type stubTool struct {
	name string
}

func (s *stubTool) Name() string                { return s.name }
func (s *stubTool) Description() string         { return "stub: " + s.name }
func (s *stubTool) Parameters() json.RawMessage { return json.RawMessage(`{"type":"object"}`) }
func (s *stubTool) Execute(_ context.Context, _ string) (*ToolResult, error) {
	return &ToolResult{Output: "ok"}, nil
}

func TestRegister(t *testing.T) {
	r := NewRegistry()
	r.Register(&stubTool{name: "alpha"})

	got := r.Get("alpha")
	if got == nil {
		t.Fatal("expected to retrieve registered tool 'alpha', got nil")
	}
	if got.Name() != "alpha" {
		t.Fatalf("expected tool name 'alpha', got %q", got.Name())
	}
}

func TestRegisterDuplicatePanics(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Fatal("expected panic on duplicate registration, but did not panic")
		}
	}()

	r := NewRegistry()
	r.Register(&stubTool{name: "dup"})
	r.Register(&stubTool{name: "dup"})
}

func TestGetNonExistent(t *testing.T) {
	r := NewRegistry()
	got := r.Get("does_not_exist")
	if got != nil {
		t.Fatalf("expected nil for non-existent tool, got %v", got)
	}
}

func TestAPITools(t *testing.T) {
	r := NewRegistry()
	r.Register(&stubTool{name: "first"})
	r.Register(&stubTool{name: "second"})

	apiTools := r.APITools()

	if len(apiTools) != 2 {
		t.Fatalf("expected 2 API tools, got %d", len(apiTools))
	}

	// Verify ordering matches registration order.
	if apiTools[0].Function.Name != "first" {
		t.Errorf("expected first tool name 'first', got %q", apiTools[0].Function.Name)
	}
	if apiTools[1].Function.Name != "second" {
		t.Errorf("expected second tool name 'second', got %q", apiTools[1].Function.Name)
	}

	// Verify the Type field is set correctly.
	for i, at := range apiTools {
		if at.Type != "function" {
			t.Errorf("apiTools[%d].Type = %q, want 'function'", i, at.Type)
		}
		if at.Function.Description == "" {
			t.Errorf("apiTools[%d].Function.Description is empty", i)
		}
		if at.Function.Parameters == nil {
			t.Errorf("apiTools[%d].Function.Parameters is nil", i)
		}
	}
}
