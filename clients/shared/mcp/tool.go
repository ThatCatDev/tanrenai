package mcp

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	sdkmcp "github.com/modelcontextprotocol/go-sdk/mcp"

	"github.com/ThatCatDev/tanrenai/shared/tools"
)

// Tool adapts an MCP-exposed tool so the agent's tools.Registry can
// hold and dispatch it like any built-in. Each MCP server's tools land
// in the registry under names of the form `<server>__<tool>`. Prefix
// keeps multiple MCP servers (or an MCP server + a built-in) from
// colliding; the separator is `__` because Anthropic API tool-name
// rules accept letters, digits, underscore, and hyphen, but disallow
// special characters that would help disambiguate visually.
type Tool struct {
	// PrefixedName is the namespaced name the registry uses
	// ("playwright__browser_navigate"). The MCP server only knows
	// the bare `tool.Name` ("browser_navigate") — Execute strips the
	// prefix before the wire call.
	PrefixedName string

	// Bare name as known by the MCP server. Used by the Execute path
	// and by tests that want to assert wire-level behavior.
	BareName string

	tool   *sdkmcp.Tool
	client *Client
}

// NameSeparator is the joiner between server prefix and the MCP tool's
// own name in the registry. Exposed for test assertions.
const NameSeparator = "__"

func (t *Tool) Name() string        { return t.PrefixedName }
func (t *Tool) Description() string { return t.tool.Description }

// Parameters returns the JSON-Schema the model receives. MCP servers
// publish their input schemas as `inputSchema` (typed `any` in the
// SDK, since the spec allows any JSON-Schema object); we marshal it
// straight through. Falls back to an empty object schema so the agent
// never sees a tool with a nil schema (the OpenAI-compatible JSON would
// be malformed otherwise).
func (t *Tool) Parameters() json.RawMessage {
	if t.tool.InputSchema == nil {
		return json.RawMessage(`{"type":"object"}`)
	}
	raw, err := json.Marshal(t.tool.InputSchema)
	if err != nil {
		// Shouldn't happen — the SDK already validated the schema on
		// the server side. Fall back to the same safe default so a
		// malformed schema doesn't crash the agent loop.
		return json.RawMessage(`{"type":"object"}`)
	}
	return raw
}

// Execute calls the underlying MCP tool, then projects the
// `CallToolResult` content into the tools.ToolResult shape the agent
// loop expects.
//
// Content projection rules:
//   - Multiple TextContent blocks are concatenated with newlines —
//     mirrors how built-in tools that produce both stdout/stderr
//     present a single Output string.
//   - Non-text content (image, audio, embedded resource) is summarised
//     as "[<n> <kind> attachment(s)]" so the model sees that something
//     came back without us inventing fake text. Multimodal MCP support
//     is a v2 — would need the agent loop to forward those blocks to
//     the model as actual image/audio inputs.
//   - IsError on the MCP response flips IsError on our ToolResult, so
//     the agent's error-handling path (stuck-detection, retries) kicks
//     in without needing to inspect strings.
func (t *Tool) Execute(ctx context.Context, arguments string) (*tools.ToolResult, error) {
	res, err := t.client.Call(ctx, t.BareName, arguments)
	if err != nil {
		// Transport-level failure (server crashed, network drop, etc.)
		// — surface as an erroring ToolResult so the agent can decide
		// to retry / abandon / report. Returning an error here would
		// make the agent treat the whole turn as failed.
		return &tools.ToolResult{
			IsError: true,
			Output:  fmt.Sprintf("mcp transport error: %v", err),
		}, nil
	}

	return &tools.ToolResult{
		Output:  projectContent(res.Content),
		IsError: res.IsError,
	}, nil
}

// projectContent flattens MCP Content blocks into a single string.
// See the Execute doc for the rules.
func projectContent(blocks []sdkmcp.Content) string {
	if len(blocks) == 0 {
		return ""
	}
	var sb strings.Builder
	type counted struct {
		text       int
		image      int
		audio      int
		resource   int
		other      int
	}
	var counts counted
	for _, b := range blocks {
		switch c := b.(type) {
		case *sdkmcp.TextContent:
			if sb.Len() > 0 {
				sb.WriteByte('\n')
			}
			sb.WriteString(c.Text)
			counts.text++
		case *sdkmcp.ImageContent:
			counts.image++
		case *sdkmcp.AudioContent:
			counts.audio++
		case *sdkmcp.ResourceLink, *sdkmcp.EmbeddedResource:
			counts.resource++
		default:
			counts.other++
		}
	}
	// Append a one-line summary for non-text attachments so the model
	// at least sees that they exist. Kept brief — the model can ask a
	// follow-up if needed.
	if counts.image+counts.audio+counts.resource+counts.other > 0 {
		if sb.Len() > 0 {
			sb.WriteByte('\n')
		}
		sb.WriteString("[")
		first := true
		joinSummary := func(n int, kind string) {
			if n == 0 {
				return
			}
			if !first {
				sb.WriteString(", ")
			}
			first = false
			fmt.Fprintf(&sb, "%d %s", n, kind)
			if n > 1 {
				sb.WriteByte('s')
			}
		}
		joinSummary(counts.image, "image attachment")
		joinSummary(counts.audio, "audio attachment")
		joinSummary(counts.resource, "resource attachment")
		joinSummary(counts.other, "other attachment")
		sb.WriteString("]")
	}
	return sb.String()
}

// adaptTools wraps every MCP tool a client exposes as a tools.Tool
// keyed by the namespaced name. Used by Registry.AttachTo.
func adaptTools(c *Client) []*Tool {
	mcpTools := c.Tools()
	out := make([]*Tool, len(mcpTools))
	for i, mt := range mcpTools {
		out[i] = &Tool{
			PrefixedName: c.Name + NameSeparator + mt.Name,
			BareName:     mt.Name,
			tool:         mt,
			client:       c,
		}
	}
	return out
}
