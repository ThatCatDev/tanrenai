package tools

import (
	"context"
	"encoding/json"
	"errors"
	"testing"
)

type fakeRequester struct {
	onRequest func(name, arguments string) (*ToolResult, error)
	gotName   string
	gotArgs   string
	calls     int
}

func (f *fakeRequester) RequestTool(_ context.Context, name, arguments string) (*ToolResult, error) {
	f.gotName = name
	f.gotArgs = arguments
	f.calls++
	if f.onRequest != nil {
		return f.onRequest(name, arguments)
	}

	return &ToolResult{Output: "ok"}, nil
}

func TestRPCTool_PreservesIdentity(t *testing.T) {
	schema := json.RawMessage(`{"type":"object"}`)
	tool := NewRPCTool("my_tool", "does a thing", schema, &fakeRequester{})

	if tool.Name() != "my_tool" {
		t.Errorf("Name() = %q, want %q", tool.Name(), "my_tool")
	}
	if tool.Description() != "does a thing" {
		t.Errorf("Description() = %q", tool.Description())
	}
	if string(tool.Parameters()) != string(schema) {
		t.Errorf("Parameters() = %s, want %s", tool.Parameters(), schema)
	}
}

func TestRPCTool_ExecuteForwardsToRequester(t *testing.T) {
	req := &fakeRequester{
		onRequest: func(_, _ string) (*ToolResult, error) {
			return &ToolResult{Output: "remote result"}, nil
		},
	}
	tool := NewRPCTool("file_read", "", nil, req)

	res, err := tool.Execute(context.Background(), `{"path":"foo"}`)
	if err != nil {
		t.Fatalf("Execute returned error: %v", err)
	}
	if res.Output != "remote result" {
		t.Errorf("Output = %q, want %q", res.Output, "remote result")
	}
	if req.gotName != "file_read" {
		t.Errorf("requester saw name=%q", req.gotName)
	}
	if req.gotArgs != `{"path":"foo"}` {
		t.Errorf("requester saw args=%q", req.gotArgs)
	}
}

func TestRPCTool_ExecutePropagatesError(t *testing.T) {
	req := &fakeRequester{
		onRequest: func(_, _ string) (*ToolResult, error) {
			return nil, errors.New("boom")
		},
	}
	tool := NewRPCTool("x", "", nil, req)

	if _, err := tool.Execute(context.Background(), "{}"); err == nil {
		t.Error("expected error to propagate, got nil")
	}
}

// ── Registry.Replace ────────────────────────────────────────────────────

func TestRegistry_ReplaceSwapsTool(t *testing.T) {
	r := NewRegistry()
	original := &stubTool{name: "x"}
	replacement := &stubTool{name: "x"}
	r.Register(original)

	if !r.Replace("x", replacement) {
		t.Fatal("Replace returned false for a registered name")
	}
	if r.Get("x") != replacement {
		t.Error("Get(\"x\") did not return the replacement")
	}
}

func TestRegistry_ReplaceUnknownReturnsFalse(t *testing.T) {
	r := NewRegistry()
	if r.Replace("missing", &stubTool{name: "missing"}) {
		t.Error("Replace returned true for an unregistered name")
	}
}

func TestRegistry_ReplacePreservesOrder(t *testing.T) {
	r := NewRegistry()
	r.Register(&stubTool{name: "a"})
	r.Register(&stubTool{name: "b"})
	r.Register(&stubTool{name: "c"})

	r.Replace("b", &stubTool{name: "b"})

	want := []string{"a", "b", "c"}
	got := r.APITools()
	if len(got) != len(want) {
		t.Fatalf("APITools count = %d, want %d", len(got), len(want))
	}
	for i, name := range want {
		if got[i].Function.Name != name {
			t.Errorf("position %d: got %q, want %q", i, got[i].Function.Name, name)
		}
	}
}
