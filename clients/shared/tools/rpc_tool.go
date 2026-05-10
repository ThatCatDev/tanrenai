package tools

import (
	"context"
	"encoding/json"
)

// RPCRequester is the contract an out-of-process tool host satisfies: given a
// tool name and JSON arguments, return a ToolResult or an error. Used by
// RPCTool to delegate Execute() to a peer (e.g. an editor extension).
type RPCRequester interface {
	RequestTool(ctx context.Context, name, arguments string) (*ToolResult, error)
}

// RPCTool is a Tool whose Execute is forwarded to an RPCRequester. The local
// process retains the tool's identity (name, description, schema) so the
// agent loop and the LLM see the tool exactly as if it ran locally — only
// the actual execution lives elsewhere.
type RPCTool struct {
	name        string
	description string
	schema      json.RawMessage
	requester   RPCRequester
}

// NewRPCTool wraps an existing local Tool's identity but routes execution
// through the requester.
func NewRPCTool(name, description string, schema json.RawMessage, r RPCRequester) *RPCTool {
	return &RPCTool{
		name:        name,
		description: description,
		schema:      schema,
		requester:   r,
	}
}

func (t *RPCTool) Name() string                { return t.name }
func (t *RPCTool) Description() string         { return t.description }
func (t *RPCTool) Parameters() json.RawMessage { return t.schema }

func (t *RPCTool) Execute(ctx context.Context, arguments string) (*ToolResult, error) {
	return t.requester.RequestTool(ctx, t.name, arguments)
}
