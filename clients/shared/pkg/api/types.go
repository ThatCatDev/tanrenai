// SYNC: This file is kept in sync with server/pkg/api/types.go.
// When modifying, update both copies.
package api

import (
	"encoding/json"
	"fmt"
	"time"
)

// Message represents a chat message.
type Message struct {
	Role       string     `json:"-"`
	Content    string     `json:"-"`
	// ContentParts, when non-empty, is sent as the message's `content`
	// field as an array (OpenAI multimodal format). Otherwise Content is
	// sent as a plain string. Use NewTextMessage / NewMultimodalMessage to
	// build messages rather than touching these directly.
	ContentParts []ContentPart `json:"-"`
	ToolCalls    []ToolCall    `json:"tool_calls,omitempty"`
	ToolCallID   string        `json:"tool_call_id,omitempty"`
	Name         string        `json:"name,omitempty"`
}

// ContentPart is one piece of a multimodal message — currently either
// text or an image URL (which may be a data: URL with base64 payload).
type ContentPart struct {
	Type     string    `json:"type"` // "text" | "image_url"
	Text     string    `json:"text,omitempty"`
	ImageURL *ImageURL `json:"image_url,omitempty"`
}

// ImageURL carries the actual image — usually a `data:image/<type>;base64,...`
// URL but http(s) URLs work too for vision models that fetch them.
type ImageURL struct {
	URL    string `json:"url"`
	Detail string `json:"detail,omitempty"` // "low" | "high" | "auto"
}

// MarshalJSON emits the OpenAI-compatible shape: top-level `content`
// is either a string (legacy / text-only) or an array (multimodal).
func (m Message) MarshalJSON() ([]byte, error) {
	type wire struct {
		Role       string     `json:"role"`
		Content    any        `json:"content,omitempty"`
		ToolCalls  []ToolCall `json:"tool_calls,omitempty"`
		ToolCallID string     `json:"tool_call_id,omitempty"`
		Name       string     `json:"name,omitempty"`
	}
	w := wire{
		Role:       m.Role,
		ToolCalls:  m.ToolCalls,
		ToolCallID: m.ToolCallID,
		Name:       m.Name,
	}
	if len(m.ContentParts) > 0 {
		w.Content = m.ContentParts
	} else {
		w.Content = m.Content
	}

	return json.Marshal(w)
}

// UnmarshalJSON accepts both string and array `content`.
func (m *Message) UnmarshalJSON(data []byte) error {
	type wire struct {
		Role       string          `json:"role"`
		Content    json.RawMessage `json:"content"`
		ToolCalls  []ToolCall      `json:"tool_calls,omitempty"`
		ToolCallID string          `json:"tool_call_id,omitempty"`
		Name       string          `json:"name,omitempty"`
	}
	var w wire
	if err := json.Unmarshal(data, &w); err != nil {
		return err
	}
	m.Role = w.Role
	m.ToolCalls = w.ToolCalls
	m.ToolCallID = w.ToolCallID
	m.Name = w.Name

	if len(w.Content) == 0 || string(w.Content) == "null" {
		return nil
	}
	// Try string first — most common shape.
	var s string
	if err := json.Unmarshal(w.Content, &s); err == nil {
		m.Content = s

		return nil
	}
	// Fall back to array.
	var parts []ContentPart
	if err := json.Unmarshal(w.Content, &parts); err != nil {
		return fmt.Errorf("message content was neither string nor array: %w", err)
	}
	m.ContentParts = parts
	// Surface text portions through .Content so existing string-based
	// consumers (token estimation, summary, scrolls) still see something
	// meaningful even when they can't render the image bits.
	var b []byte
	for _, p := range parts {
		if p.Type == "text" {
			if len(b) > 0 {
				b = append(b, '\n')
			}
			b = append(b, p.Text...)
		}
	}
	m.Content = string(b)

	return nil
}

// NewTextMessage builds a plain-text message — equivalent to the
// pre-multimodal struct literal pattern.
func NewTextMessage(role, content string) Message {
	return Message{Role: role, Content: content}
}

// NewMultimodalMessage builds a message with both a text prompt and one
// or more image URLs. URLs can be `data:image/<type>;base64,...` or http(s).
func NewMultimodalMessage(role, text string, imageURLs []string) Message {
	parts := make([]ContentPart, 0, 1+len(imageURLs))
	if text != "" {
		parts = append(parts, ContentPart{Type: "text", Text: text})
	}
	for _, u := range imageURLs {
		parts = append(parts, ContentPart{
			Type:     "image_url",
			ImageURL: &ImageURL{URL: u},
		})
	}

	return Message{Role: role, ContentParts: parts, Content: text}
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
	Model          string    `json:"model"`
	Messages       []Message `json:"messages"`
	Temperature    *float64  `json:"temperature,omitempty"`
	TopP           *float64  `json:"top_p,omitempty"`
	MaxTokens      *int      `json:"max_tokens,omitempty"`
	Stream         bool      `json:"stream,omitempty"`
	Stop           []string  `json:"stop,omitempty"`
	Tools          []Tool    `json:"tools,omitempty"`
	ToolChoice     any       `json:"tool_choice,omitempty"`
	EnableThinking bool      `json:"enable_thinking,omitempty"`
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
	Role             string          `json:"role,omitempty"`
	Content          string          `json:"content,omitempty"`
	ReasoningContent string          `json:"reasoning_content,omitempty"`
	ToolCalls        []ToolCallDelta `json:"tool_calls,omitempty"`
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

// LoadResponse is the response for POST /api/load.
type LoadResponse struct {
	Status  string `json:"status"`
	Model   string `json:"model"`
	CtxSize int    `json:"ctx_size"`
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
	Status         string          `json:"status"`                    // none, pending, provisioning, running, destroying, destroyed
	ProvisionState string          `json:"provision_state,omitempty"` // searching, creating, booting, ready, failed
	GPUURL         string          `json:"gpu_url,omitempty"`
	GPUName        string          `json:"gpu_name,omitempty"`
	ModelLoaded    string          `json:"model_loaded,omitempty"`
	IdleSince      *time.Time      `json:"idle_since,omitempty"`
	Download       *DownloadStatus `json:"download,omitempty"`
}

// DownloadStatus mirrors the platform's in-flight /api/pull tracker. Set
// on InstanceStatus while the platform is auto-pulling a model that
// wasn't on disk yet.
type DownloadStatus struct {
	Model       string    `json:"model"`
	URI         string    `json:"uri"`
	StartedAt   time.Time `json:"started_at"`
	Done        bool      `json:"done"`
	Error       string    `json:"error,omitempty"`
	CurrentFile int       `json:"current_file,omitempty"` // 1-indexed, across a multi-shard download
	TotalFiles  int       `json:"total_files,omitempty"`  // >1 indicates a sharded model
	Percent     int       `json:"percent,omitempty"`      // percent of the current file
}

// Model pull types

// PullRequest is the request body for POST /api/pull.
//
// `Name`, when set, is the destination basename the GPU should save the
// downloaded GGUF under (without `.gguf`). For sharded models the
// `-NNNNN-of-MMMMM.gguf` suffix from the source URL is preserved so each
// shard still lands in its own file. When empty, the GPU saves under the
// source URL's filename. Letting callers pin the on-disk name means a
// user-typed model identifier flows through pull → load → /v1/models
// unchanged, with no normalization layer.
type PullRequest struct {
	URL  string `json:"url"`
	Name string `json:"name,omitempty"`
}

// PullEvent is a streaming SSE event during a model download.
type PullEvent struct {
	Status     string `json:"status"` // "resolving", "downloading", "downloaded", "error"
	Downloaded int64  `json:"downloaded,omitempty"`
	Total      int64  `json:"total,omitempty"`
	Percent    int    `json:"percent,omitempty"`
	File       int    `json:"file,omitempty"`        // current file number (for split GGUFs)
	TotalFiles int    `json:"total_files,omitempty"` // total file count
	Path       string `json:"path,omitempty"`
	Error      string `json:"error,omitempty"`
}
