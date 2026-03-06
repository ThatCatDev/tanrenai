package tools

// ToolResult is the output of a tool execution.
// Tool-level errors (file not found, command failed) use IsError=true
// so the LLM can see and react to them. Go errors are reserved for
// infrastructure failures (context cancelled, etc.).
type ToolResult struct {
	Output  string
	IsError bool
}

// ErrorResult creates a ToolResult representing a tool-level error.
func ErrorResult(msg string) *ToolResult {
	return &ToolResult{Output: msg, IsError: true}
}
