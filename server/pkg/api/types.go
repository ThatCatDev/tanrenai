package api

import (
	"encoding/json"
	"time"
)

// Message represents a chat message.
type Message struct {
	Role       string     `json:"role"`
	Content    string     `json:"content"`
	ToolCalls  []ToolCall `json:"tool_calls,omitempty"`
	ToolCallID string     `json:"tool_call_id,omitempty"`
	Name       string     `json:"name,omitempty"`
}

// Tool represents a tool available for the model to call.
type Tool struct {
	Type     string       `json:"type"`
	Function ToolFunction `json:"function"`
}

// ToolFunction describes a function tool.
type ToolFunction struct {
	Name        string          `json:"name"`
	Description string          `json:"description"`
	Parameters  json.RawMessage `json:"parameters"`
}

// ToolCall represents a tool call made by the model.
type ToolCall struct {
	ID       string           `json:"id"`
	Type     string           `json:"type"`
	Function ToolCallFunction `json:"function"`
}

// ToolCallFunction is the function invocation within a tool call.
type ToolCallFunction struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

// ToolCallDelta is a streaming delta for a tool call.
type ToolCallDelta struct {
	Index    int               `json:"index"`
	ID       string            `json:"id,omitempty"`
	Type     string            `json:"type,omitempty"`
	Function *ToolCallFunction `json:"function,omitempty"`
}

// ChatCompletionRequest matches the OpenAI chat completions request schema.
type ChatCompletionRequest struct {
	Model       string    `json:"model"`
	Messages    []Message `json:"messages"`
	Temperature *float64  `json:"temperature,omitempty"`
	TopP        *float64  `json:"top_p,omitempty"`
	MaxTokens   *int      `json:"max_tokens,omitempty"`
	Stream      bool      `json:"stream,omitempty"`
	Stop        []string  `json:"stop,omitempty"`
	Tools       []Tool    `json:"tools,omitempty"`
	ToolChoice  any       `json:"tool_choice,omitempty"`
}

// ChatCompletionResponse matches the OpenAI chat completions response schema.
type ChatCompletionResponse struct {
	ID      string   `json:"id"`
	Object  string   `json:"object"`
	Created int64    `json:"created"`
	Model   string   `json:"model"`
	Choices []Choice `json:"choices"`
	Usage   *Usage   `json:"usage,omitempty"`
}

// Choice is a single completion choice.
type Choice struct {
	Index        int     `json:"index"`
	Message      Message `json:"message"`
	FinishReason string  `json:"finish_reason"`
}

// ChatCompletionChunk is a streaming SSE chunk.
type ChatCompletionChunk struct {
	ID      string        `json:"id"`
	Object  string        `json:"object"`
	Created int64         `json:"created"`
	Model   string        `json:"model"`
	Choices []ChunkChoice `json:"choices"`
}

// ChunkChoice is a single choice within a streaming chunk.
type ChunkChoice struct {
	Index        int          `json:"index"`
	Delta        MessageDelta `json:"delta"`
	FinishReason *string      `json:"finish_reason"`
}

// MessageDelta is the incremental content in a streaming chunk.
type MessageDelta struct {
	Role      string          `json:"role,omitempty"`
	Content   string          `json:"content,omitempty"`
	ToolCalls []ToolCallDelta `json:"tool_calls,omitempty"`
}

// Usage contains token usage information.
type Usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

// ModelInfo represents a model in the /v1/models response.
type ModelInfo struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Created int64  `json:"created"`
	OwnedBy string `json:"owned_by"`
}

// ModelListResponse is the response for GET /v1/models.
type ModelListResponse struct {
	Object string      `json:"object"`
	Data   []ModelInfo `json:"data"`
}

// ErrorResponse is the standard error response.
type ErrorResponse struct {
	Error ErrorDetail `json:"error"`
}

// ErrorDetail contains error information.
type ErrorDetail struct {
	Message string `json:"message"`
	Type    string `json:"type"`
	Code    string `json:"code,omitempty"`
}

// Embedding API types

// EmbeddingRequest is the request for POST /v1/embeddings.
type EmbeddingRequest struct {
	Input string `json:"input"`
	Model string `json:"model"`
}

// EmbeddingResponse is the response for POST /v1/embeddings.
type EmbeddingResponse struct {
	Data []EmbeddingData `json:"data"`
}

// EmbeddingData contains a single embedding vector.
type EmbeddingData struct {
	Embedding []float32 `json:"embedding"`
	Index     int       `json:"index"`
}

// Memory API types

// MemoryEntry represents a single memory entry.
type MemoryEntry struct {
	ID        string    `json:"id"`
	UserMsg   string    `json:"user_msg"`
	AssistMsg string    `json:"assist_msg"`
	Timestamp time.Time `json:"timestamp"`
	SessionID string    `json:"session_id,omitempty"`
}

// MemorySearchResult is a memory entry with associated scores.
type MemorySearchResult struct {
	Entry         MemoryEntry `json:"entry"`
	SemanticScore float32     `json:"semantic_score"`
	KeywordScore  float32     `json:"keyword_score"`
	CombinedScore float32     `json:"combined_score"`
}

// MemorySearchRequest is the request for POST /v1/memory/search.
type MemorySearchRequest struct {
	Query string `json:"query"`
	Limit int    `json:"limit,omitempty"`
}

// MemorySearchResponse is the response for POST /v1/memory/search.
type MemorySearchResponse struct {
	Results []MemorySearchResult `json:"results"`
}

// MemoryStoreRequest is the request for POST /v1/memory/store.
type MemoryStoreRequest struct {
	UserMsg   string `json:"user_msg"`
	AssistMsg string `json:"assist_msg"`
}

// MemoryStoreResponse is the response for POST /v1/memory/store.
type MemoryStoreResponse struct {
	ID string `json:"id"`
}

// MemoryListResponse is the response for GET /v1/memory/list.
type MemoryListResponse struct {
	Entries []MemoryEntry `json:"entries"`
	Total   int           `json:"total"`
}

// MemoryCountResponse is the response for GET /v1/memory/count.
type MemoryCountResponse struct {
	Count int `json:"count"`
}

// Instance management types

// InstanceStatus represents the status of a GPU instance.
type InstanceStatus struct {
	Status    string     `json:"status"` // running, stopped, starting
	GPUURL    string     `json:"gpu_url,omitempty"`
	IdleSince *time.Time `json:"idle_since,omitempty"`
}
