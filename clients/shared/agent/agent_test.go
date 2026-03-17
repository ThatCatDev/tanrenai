package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"testing"

	"github.com/ThatCatDev/tanrenai/shared/apiclient"
	"github.com/ThatCatDev/tanrenai/shared/pkg/api"
	"github.com/ThatCatDev/tanrenai/shared/tools"
)

// mockTool returns a fixed result when executed.
type mockTool struct {
	name   string
	result string
}

func (m *mockTool) Name() string        { return m.name }
func (m *mockTool) Description() string { return "mock tool" }
func (m *mockTool) Parameters() json.RawMessage {
	return json.RawMessage(`{"type":"object","properties":{"path":{"type":"string"}}}`)
}
func (m *mockTool) Execute(_ context.Context, _ string) (*tools.ToolResult, error) {
	return &tools.ToolResult{Output: m.result}, nil
}

// TestAgentLoopToolMessages verifies that tool messages are properly formed
// on iteration 2 after a tool call.
func TestAgentLoopToolMessages(t *testing.T) {
	registry := tools.NewRegistry()
	registry.Register(&mockTool{name: "list_dir", result: "file1.go\nfile2.go"})

	// Track requests sent to the mock LLM
	var requests []*api.ChatCompletionRequest

	completeFn := func(_ context.Context, req *api.ChatCompletionRequest) (*api.ChatCompletionResponse, error) {
		// Deep copy messages to capture state
		reqCopy := *req
		msgCopy := make([]api.Message, len(req.Messages))
		copy(msgCopy, req.Messages)
		reqCopy.Messages = msgCopy
		requests = append(requests, &reqCopy)

		iteration := len(requests)

		switch iteration {
		case 1:
			// First iteration: model calls a tool
			return &api.ChatCompletionResponse{
				Choices: []api.Choice{{
					FinishReason: "tool_calls",
					Message: api.Message{
						Role: "assistant",
						ToolCalls: []api.ToolCall{{
							ID:   "call_123",
							Type: "function",
							Function: api.ToolCallFunction{
								Name:      "list_dir",
								Arguments: `{"path":"."}`,
							},
						}},
					},
				}},
			}, nil
		case 2:
			// Second iteration: model responds with content
			return &api.ChatCompletionResponse{
				Choices: []api.Choice{{
					FinishReason: "stop",
					Message: api.Message{
						Role:    "assistant",
						Content: "The directory contains file1.go and file2.go.",
					},
				}},
			}, nil
		default:
			return nil, fmt.Errorf("unexpected iteration %d", iteration)
		}
	}

	messages := []api.Message{
		{Role: "system", Content: "You are a helpful assistant."},
		{Role: "user", Content: "List the files"},
	}

	cfg := Config{
		MaxIterations: 5,
		Tools:         registry,
		Hooks:         Hooks{},
	}

	result, err := Run(context.Background(), completeFn, messages, cfg)
	if err != nil {
		t.Fatalf("Run failed: %v", err)
	}

	// Verify we got 2 requests
	if len(requests) != 2 {
		t.Fatalf("expected 2 requests, got %d", len(requests))
	}

	// Verify iteration 2 request has proper tool result message
	req2 := requests[1]
	t.Logf("Iteration 2 request has %d messages:", len(req2.Messages))
	for i, m := range req2.Messages {
		t.Logf("  [%d] role=%s content_len=%d tool_calls=%d tool_call_id=%q name=%q",
			i, m.Role, len(m.Content), len(m.ToolCalls), m.ToolCallID, m.Name)
	}

	// Find the tool result message
	var toolMsg *api.Message
	for i := range req2.Messages {
		if req2.Messages[i].Role == "tool" {
			toolMsg = &req2.Messages[i]

			break
		}
	}
	if toolMsg == nil {
		t.Fatal("no tool message found in iteration 2 request")
	}

	if toolMsg.ToolCallID != "call_123" {
		t.Errorf("tool message ToolCallID = %q, want %q", toolMsg.ToolCallID, "call_123")
	}
	if toolMsg.Name != "list_dir" {
		t.Errorf("tool message Name = %q, want %q", toolMsg.Name, "list_dir")
	}
	if toolMsg.Content != "file1.go\nfile2.go" {
		t.Errorf("tool message Content = %q, want %q", toolMsg.Content, "file1.go\nfile2.go")
	}

	// Find the assistant message with tool calls (should have content stripped)
	var assistToolMsg *api.Message
	for i := range req2.Messages {
		if req2.Messages[i].Role == "assistant" && len(req2.Messages[i].ToolCalls) > 0 {
			assistToolMsg = &req2.Messages[i]

			break
		}
	}
	if assistToolMsg == nil {
		t.Fatal("no assistant tool_calls message found in iteration 2 request")
	}
	if assistToolMsg.Content != "" {
		t.Errorf("assistant tool_calls message should have empty content after stripNarration, got %q", assistToolMsg.Content)
	}

	// Verify final result includes the response
	var lastAssist string
	for _, m := range result {
		if m.Role == "assistant" && m.Content != "" {
			lastAssist = m.Content
		}
	}
	if !strings.Contains(lastAssist, "file1.go") {
		t.Errorf("expected final response to contain 'file1.go', got %q", lastAssist)
	}
}

// TestAgentLoopToolCallIDPropagation verifies that when llama-server returns
// empty tool_call IDs, they're still properly propagated.
func TestAgentLoopToolCallIDPropagation(t *testing.T) {
	registry := tools.NewRegistry()
	registry.Register(&mockTool{name: "list_dir", result: "output"})

	var req2Messages []api.Message

	completeFn := func(_ context.Context, req *api.ChatCompletionRequest) (*api.ChatCompletionResponse, error) {
		if len(req2Messages) == 0 && hasToolMessage(req.Messages) {
			req2Messages = make([]api.Message, len(req.Messages))
			copy(req2Messages, req.Messages)
		}

		if !hasToolMessage(req.Messages) {
			// Iteration 1: return tool call with EMPTY ID (simulating llama-server)
			return &api.ChatCompletionResponse{
				Choices: []api.Choice{{
					FinishReason: "tool_calls",
					Message: api.Message{
						Role: "assistant",
						ToolCalls: []api.ToolCall{{
							ID:   "", // Empty ID from llama-server
							Type: "function",
							Function: api.ToolCallFunction{
								Name:      "list_dir",
								Arguments: `{"path":"."}`,
							},
						}},
					},
				}},
			}, nil
		}

		return &api.ChatCompletionResponse{
			Choices: []api.Choice{{
				FinishReason: "stop",
				Message:      api.Message{Role: "assistant", Content: "Done."},
			}},
		}, nil
	}

	messages := []api.Message{
		{Role: "user", Content: "test"},
	}

	_, err := Run(context.Background(), completeFn, messages, Config{
		MaxIterations: 5,
		Tools:         registry,
	})
	if err != nil {
		t.Fatalf("Run failed: %v", err)
	}

	// Check that tool result has a generated ID (since model sent empty)
	for _, m := range req2Messages {
		if m.Role == "tool" {
			t.Logf("Tool message: tool_call_id=%q name=%q content_len=%d", m.ToolCallID, m.Name, len(m.Content))
			if m.ToolCallID != "call_0" {
				t.Errorf("expected generated tool_call_id 'call_0', got %q", m.ToolCallID)
			}
		}
	}

	// Check assistant message also got the generated ID
	for _, m := range req2Messages {
		if m.Role == "assistant" && len(m.ToolCalls) > 0 {
			if m.ToolCalls[0].ID != "call_0" {
				t.Errorf("expected assistant tool_call ID to be updated to 'call_0', got %q", m.ToolCalls[0].ID)
			}
		}
	}

	// Dump the full request for inspection
	t.Logf("Iteration 2 messages:")
	for i, m := range req2Messages {
		j, _ := json.Marshal(m)
		t.Logf("  [%d] %s", i, string(j))
	}
}

// TestStreamingAgentToolMessages tests the streaming variant.
func TestStreamingAgentToolMessages(t *testing.T) {
	registry := tools.NewRegistry()
	registry.Register(&mockTool{name: "list_dir", result: "file1.go"})

	var requests []*api.ChatCompletionRequest

	streamFn := func(_ context.Context, req *api.ChatCompletionRequest) (<-chan apiclient.StreamEvent, error) {
		reqCopy := *req
		msgCopy := make([]api.Message, len(req.Messages))
		copy(msgCopy, req.Messages)
		reqCopy.Messages = msgCopy
		requests = append(requests, &reqCopy)

		ch := make(chan apiclient.StreamEvent, 10)
		iteration := len(requests)

		go func() {
			defer close(ch)
			switch iteration {
			case 1:
				// Tool call via streaming
				fr := "tool_calls"
				ch <- apiclient.StreamEvent{Chunk: &api.ChatCompletionChunk{
					Choices: []api.ChunkChoice{{
						Delta: api.MessageDelta{
							Role: "assistant",
							ToolCalls: []api.ToolCallDelta{{
								Index: 0,
								ID:    "call_456",
								Type:  "function",
								Function: &api.ToolCallFunction{
									Name:      "list_dir",
									Arguments: `{"path":"."}`,
								},
							}},
						},
						FinishReason: &fr,
					}},
				}}
				ch <- apiclient.StreamEvent{Done: true}
			case 2:
				// Final response
				fr := "stop"
				ch <- apiclient.StreamEvent{Chunk: &api.ChatCompletionChunk{
					Choices: []api.ChunkChoice{{
						Delta:        api.MessageDelta{Role: "assistant", Content: "Found file1.go."},
						FinishReason: &fr,
					}},
				}}
				ch <- apiclient.StreamEvent{Done: true}
			}
		}()

		return ch, nil
	}

	messages := []api.Message{
		{Role: "user", Content: "list files"},
	}

	cfg := StreamingConfig{
		Config: Config{
			MaxIterations: 5,
			Tools:         registry,
		},
	}

	result, err := RunStreaming(context.Background(), streamFn, messages, cfg)
	if err != nil {
		t.Fatalf("RunStreaming failed: %v", err)
	}

	if len(requests) != 2 {
		t.Fatalf("expected 2 requests, got %d", len(requests))
	}

	// Dump iteration 2 request
	req2 := requests[1]
	t.Logf("Streaming iteration 2 messages:")
	for i, m := range req2.Messages {
		j, _ := json.Marshal(m)
		t.Logf("  [%d] %s", i, string(j))
	}

	// Verify tool message exists and has correct ID
	found := false
	for _, m := range req2.Messages {
		if m.Role == "tool" {
			found = true
			if m.ToolCallID != "call_456" {
				t.Errorf("tool_call_id = %q, want %q", m.ToolCallID, "call_456")
			}
		}
	}
	if !found {
		t.Error("no tool message found in streaming iteration 2 request")
	}

	// Verify result contains final response
	var lastContent string
	for _, m := range result {
		if m.Role == "assistant" && m.Content != "" {
			lastContent = m.Content
		}
	}
	if !strings.Contains(lastContent, "file1.go") {
		t.Errorf("expected final content to mention file1.go, got %q", lastContent)
	}
}

func hasToolMessage(messages []api.Message) bool {
	for _, m := range messages {
		if m.Role == "tool" {
			return true
		}
	}

	return false
}
