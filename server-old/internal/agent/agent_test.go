package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"testing"

	"github.com/ThatCatDev/tanrenai/server/internal/chatctx"
	"github.com/ThatCatDev/tanrenai/server/internal/runner"
	"github.com/ThatCatDev/tanrenai/server/internal/tools"
	"github.com/ThatCatDev/tanrenai/server/pkg/api"
)

// echoTool returns whatever arguments it receives.
type echoTool struct{}

func (t *echoTool) Name() string                { return "echo" }
func (t *echoTool) Description() string          { return "echoes input" }
func (t *echoTool) Parameters() json.RawMessage  { return json.RawMessage(`{"type":"object"}`) }
func (t *echoTool) Execute(_ context.Context, arguments string) (*tools.ToolResult, error) {
	return &tools.ToolResult{Output: "echo: " + arguments}, nil
}

// failTool always returns an error result.
type failTool struct{}

func (t *failTool) Name() string                { return "fail" }
func (t *failTool) Description() string          { return "always fails" }
func (t *failTool) Parameters() json.RawMessage  { return json.RawMessage(`{"type":"object"}`) }
func (t *failTool) Execute(_ context.Context, _ string) (*tools.ToolResult, error) {
	return tools.ErrorResult("something went wrong"), nil
}

func newTestRegistry(tt ...tools.Tool) *tools.Registry {
	r := tools.NewRegistry()
	for _, t := range tt {
		r.Register(t)
	}
	return r
}

func TestRunSimpleResponse(t *testing.T) {
	// Model returns a plain text response, no tool calls.
	completeFn := func(_ context.Context, req *api.ChatCompletionRequest) (*api.ChatCompletionResponse, error) {
		return &api.ChatCompletionResponse{
			Choices: []api.Choice{
				{
					Message:      api.Message{Role: "assistant", Content: "Hello!"},
					FinishReason: "stop",
				},
			},
		}, nil
	}

	messages := []api.Message{{Role: "user", Content: "Hi"}}
	cfg := Config{Tools: newTestRegistry(&echoTool{})}

	result, err := Run(context.Background(), completeFn, messages, cfg)
	if err != nil {
		t.Fatal(err)
	}

	// Should have: user + assistant = 2 messages
	if len(result) != 2 {
		t.Fatalf("got %d messages, want 2", len(result))
	}
	if result[1].Role != "assistant" || result[1].Content != "Hello!" {
		t.Errorf("unexpected assistant message: %+v", result[1])
	}
}

func TestRunWithToolCall(t *testing.T) {
	callCount := 0
	completeFn := func(_ context.Context, req *api.ChatCompletionRequest) (*api.ChatCompletionResponse, error) {
		callCount++
		if callCount == 1 {
			// First call: model wants to use echo tool
			return &api.ChatCompletionResponse{
				Choices: []api.Choice{
					{
						Message: api.Message{
							Role: "assistant",
							ToolCalls: []api.ToolCall{
								{
									ID:   "call_1",
									Type: "function",
									Function: api.ToolCallFunction{
										Name:      "echo",
										Arguments: `{"text":"hello"}`,
									},
								},
							},
						},
						FinishReason: "tool_calls",
					},
				},
			}, nil
		}
		// Second call: model responds with final answer
		return &api.ChatCompletionResponse{
			Choices: []api.Choice{
				{
					Message:      api.Message{Role: "assistant", Content: "Done!"},
					FinishReason: "stop",
				},
			},
		}, nil
	}

	messages := []api.Message{{Role: "user", Content: "test"}}
	cfg := Config{Tools: newTestRegistry(&echoTool{})}

	result, err := Run(context.Background(), completeFn, messages, cfg)
	if err != nil {
		t.Fatal(err)
	}

	// user + assistant(tool_call) + tool(result) + assistant(final) = 4
	if len(result) != 4 {
		t.Fatalf("got %d messages, want 4", len(result))
	}
	if result[1].Role != "assistant" {
		t.Errorf("message[1] role = %q, want assistant", result[1].Role)
	}
	if result[2].Role != "tool" {
		t.Errorf("message[2] role = %q, want tool", result[2].Role)
	}
	if result[2].ToolCallID != "call_1" {
		t.Errorf("tool result tool_call_id = %q, want call_1", result[2].ToolCallID)
	}
	if result[3].Content != "Done!" {
		t.Errorf("final message content = %q, want Done!", result[3].Content)
	}
}

func TestRunUnknownTool(t *testing.T) {
	callCount := 0
	completeFn := func(_ context.Context, req *api.ChatCompletionRequest) (*api.ChatCompletionResponse, error) {
		callCount++
		if callCount == 1 {
			return &api.ChatCompletionResponse{
				Choices: []api.Choice{
					{
						Message: api.Message{
							Role: "assistant",
							ToolCalls: []api.ToolCall{
								{
									ID:   "call_1",
									Type: "function",
									Function: api.ToolCallFunction{
										Name:      "nonexistent",
										Arguments: `{}`,
									},
								},
							},
						},
						FinishReason: "tool_calls",
					},
				},
			}, nil
		}
		return &api.ChatCompletionResponse{
			Choices: []api.Choice{
				{
					Message:      api.Message{Role: "assistant", Content: "ok"},
					FinishReason: "stop",
				},
			},
		}, nil
	}

	messages := []api.Message{{Role: "user", Content: "test"}}
	cfg := Config{Tools: newTestRegistry(&echoTool{})}

	result, err := Run(context.Background(), completeFn, messages, cfg)
	if err != nil {
		t.Fatal(err)
	}

	// The tool result should contain the unknown tool error
	if result[2].Role != "tool" {
		t.Fatalf("expected tool message at index 2")
	}
	if result[2].Content != "unknown tool: nonexistent" {
		t.Errorf("got %q, want unknown tool error", result[2].Content)
	}
}

func TestRunMaxIterations(t *testing.T) {
	// Model always makes tool calls, never stops
	completeFn := func(_ context.Context, req *api.ChatCompletionRequest) (*api.ChatCompletionResponse, error) {
		return &api.ChatCompletionResponse{
			Choices: []api.Choice{
				{
					Message: api.Message{
						Role: "assistant",
						ToolCalls: []api.ToolCall{
							{
								ID:   "call_1",
								Type: "function",
								Function: api.ToolCallFunction{
									Name:      "echo",
									Arguments: `{"different":"` + fmt.Sprintf("%d", len(req.Messages)) + `"}`,
								},
							},
						},
					},
					FinishReason: "tool_calls",
				},
			},
		}, nil
	}

	messages := []api.Message{{Role: "user", Content: "test"}}
	cfg := Config{
		MaxIterations: 3,
		Tools:         newTestRegistry(&echoTool{}),
	}

	_, err := Run(context.Background(), completeFn, messages, cfg)
	if err == nil {
		t.Fatal("expected error for max iterations")
	}
	if err.Error() != "agent loop reached maximum iterations (3)" {
		t.Errorf("unexpected error: %v", err)
	}
}

func TestRunStuckDetection(t *testing.T) {
	// Model keeps making the exact same failing tool call
	completeFn := func(_ context.Context, req *api.ChatCompletionRequest) (*api.ChatCompletionResponse, error) {
		return &api.ChatCompletionResponse{
			Choices: []api.Choice{
				{
					Message: api.Message{
						Role: "assistant",
						ToolCalls: []api.ToolCall{
							{
								ID:   "call_1",
								Type: "function",
								Function: api.ToolCallFunction{
									Name:      "fail",
									Arguments: `{"same":"args"}`,
								},
							},
						},
					},
					FinishReason: "tool_calls",
				},
			},
		}, nil
	}

	messages := []api.Message{{Role: "user", Content: "test"}}
	cfg := Config{
		MaxIterations: 20,
		Tools:         newTestRegistry(&failTool{}),
	}

	_, err := Run(context.Background(), completeFn, messages, cfg)
	if err == nil {
		t.Fatal("expected error for stuck loop")
	}
	if err.Error() != "agent stuck: repeated identical failing tool calls" {
		t.Errorf("unexpected error: %v", err)
	}
}

func TestRunHooksAreCalled(t *testing.T) {
	var (
		gotAssistantMsg string
		gotToolCall     api.ToolCall
		gotToolResult   string
	)

	callCount := 0
	completeFn := func(_ context.Context, req *api.ChatCompletionRequest) (*api.ChatCompletionResponse, error) {
		callCount++
		if callCount == 1 {
			return &api.ChatCompletionResponse{
				Choices: []api.Choice{
					{
						Message: api.Message{
							Role:    "assistant",
							Content: "Let me check",
							ToolCalls: []api.ToolCall{
								{
									ID:   "call_1",
									Type: "function",
									Function: api.ToolCallFunction{
										Name:      "echo",
										Arguments: `{"x":"y"}`,
									},
								},
							},
						},
						FinishReason: "tool_calls",
					},
				},
			}, nil
		}
		return &api.ChatCompletionResponse{
			Choices: []api.Choice{
				{
					Message:      api.Message{Role: "assistant", Content: "Final answer"},
					FinishReason: "stop",
				},
			},
		}, nil
	}

	messages := []api.Message{{Role: "user", Content: "test"}}
	cfg := Config{
		Tools: newTestRegistry(&echoTool{}),
		Hooks: Hooks{
			OnAssistantMessage: func(content string) { gotAssistantMsg = content },
			OnToolCall:         func(call api.ToolCall) { gotToolCall = call },
			OnToolResult:       func(call api.ToolCall, result string) { gotToolResult = result },
		},
	}

	_, err := Run(context.Background(), completeFn, messages, cfg)
	if err != nil {
		t.Fatal(err)
	}

	// OnAssistantMessage should have been called with the final message
	if gotAssistantMsg != "Final answer" {
		t.Errorf("OnAssistantMessage got %q, want %q", gotAssistantMsg, "Final answer")
	}
	if gotToolCall.Function.Name != "echo" {
		t.Errorf("OnToolCall got name %q, want %q", gotToolCall.Function.Name, "echo")
	}
	if gotToolResult != `echo: {"x":"y"}` {
		t.Errorf("OnToolResult got %q, want %q", gotToolResult, `echo: {"x":"y"}`)
	}
}

func TestRunCompletionError(t *testing.T) {
	completeFn := func(_ context.Context, req *api.ChatCompletionRequest) (*api.ChatCompletionResponse, error) {
		return nil, fmt.Errorf("connection refused")
	}

	messages := []api.Message{{Role: "user", Content: "test"}}
	cfg := Config{Tools: newTestRegistry(&echoTool{})}

	_, err := Run(context.Background(), completeFn, messages, cfg)
	if err == nil {
		t.Fatal("expected error")
	}
	if err.Error() != "completion request failed: connection refused" {
		t.Errorf("unexpected error: %v", err)
	}
}

func TestRunEmptyResponse(t *testing.T) {
	completeFn := func(_ context.Context, req *api.ChatCompletionRequest) (*api.ChatCompletionResponse, error) {
		return &api.ChatCompletionResponse{Choices: []api.Choice{}}, nil
	}

	messages := []api.Message{{Role: "user", Content: "test"}}
	cfg := Config{Tools: newTestRegistry(&echoTool{})}

	_, err := Run(context.Background(), completeFn, messages, cfg)
	if err == nil {
		t.Fatal("expected error for empty response")
	}
}

// --- Streaming tests ---

// makeStreamEvents creates a channel of StreamEvents from content chunks.
// Simulates a streaming response with text content.
func makeStreamEvents(content string) <-chan runner.StreamEvent {
	ch := make(chan runner.StreamEvent)
	go func() {
		defer close(ch)
		// Send role chunk
		ch <- runner.StreamEvent{
			Chunk: &api.ChatCompletionChunk{
				ID:    "chatcmpl-1",
				Model: "test-model",
				Choices: []api.ChunkChoice{
					{Delta: api.MessageDelta{Role: "assistant"}},
				},
			},
		}
		// Send content in small pieces
		for i := 0; i < len(content); i += 5 {
			end := i + 5
			if end > len(content) {
				end = len(content)
			}
			ch <- runner.StreamEvent{
				Chunk: &api.ChatCompletionChunk{
					ID:    "chatcmpl-1",
					Model: "test-model",
					Choices: []api.ChunkChoice{
						{Delta: api.MessageDelta{Content: content[i:end]}},
					},
				},
			}
		}
		ch <- runner.StreamEvent{Done: true}
	}()
	return ch
}

// makeStreamToolCallEvents creates a channel simulating a streaming tool call response.
func makeStreamToolCallEvents(toolID, toolName, toolArgs string) <-chan runner.StreamEvent {
	ch := make(chan runner.StreamEvent)
	go func() {
		defer close(ch)
		ch <- runner.StreamEvent{
			Chunk: &api.ChatCompletionChunk{
				ID:    "chatcmpl-1",
				Model: "test-model",
				Choices: []api.ChunkChoice{
					{Delta: api.MessageDelta{
						Role: "assistant",
						ToolCalls: []api.ToolCallDelta{
							{
								Index: 0,
								ID:    toolID,
								Type:  "function",
								Function: &api.ToolCallFunction{
									Name: toolName,
								},
							},
						},
					}},
				},
			},
		}
		// Stream args in chunks
		for i := 0; i < len(toolArgs); i += 5 {
			end := i + 5
			if end > len(toolArgs) {
				end = len(toolArgs)
			}
			ch <- runner.StreamEvent{
				Chunk: &api.ChatCompletionChunk{
					ID:    "chatcmpl-1",
					Model: "test-model",
					Choices: []api.ChunkChoice{
						{Delta: api.MessageDelta{
							ToolCalls: []api.ToolCallDelta{
								{
									Index:    0,
									Function: &api.ToolCallFunction{Arguments: toolArgs[i:end]},
								},
							},
						}},
					},
				},
			}
		}
		ch <- runner.StreamEvent{Done: true}
	}()
	return ch
}

func TestRunStreamingSimpleResponse(t *testing.T) {
	completeFn := func(_ context.Context, req *api.ChatCompletionRequest) (<-chan runner.StreamEvent, error) {
		return makeStreamEvents("Hello from stream!"), nil
	}

	messages := []api.Message{{Role: "user", Content: "Hi"}}
	var deltas []string
	cfg := StreamingConfig{
		Config: Config{Tools: newTestRegistry(&echoTool{})},
		OnContentDelta: func(delta string) {
			deltas = append(deltas, delta)
		},
	}

	result, err := RunStreaming(context.Background(), completeFn, messages, cfg)
	if err != nil {
		t.Fatal(err)
	}

	if len(result) != 2 {
		t.Fatalf("got %d messages, want 2", len(result))
	}
	if result[1].Content != "Hello from stream!" {
		t.Errorf("content = %q, want %q", result[1].Content, "Hello from stream!")
	}
	if len(deltas) == 0 {
		t.Error("expected OnContentDelta to be called")
	}
	// All deltas concatenated should equal the full content
	full := strings.Join(deltas, "")
	if full != "Hello from stream!" {
		t.Errorf("concatenated deltas = %q, want %q", full, "Hello from stream!")
	}
}

func TestRunStreamingWithToolCall(t *testing.T) {
	callCount := 0
	completeFn := func(_ context.Context, req *api.ChatCompletionRequest) (<-chan runner.StreamEvent, error) {
		callCount++
		if callCount == 1 {
			return makeStreamToolCallEvents("call_1", "echo", `{"text":"hello"}`), nil
		}
		return makeStreamEvents("Done!"), nil
	}

	messages := []api.Message{{Role: "user", Content: "test"}}
	cfg := StreamingConfig{
		Config: Config{Tools: newTestRegistry(&echoTool{})},
	}

	result, err := RunStreaming(context.Background(), completeFn, messages, cfg)
	if err != nil {
		t.Fatal(err)
	}

	// user + assistant(tool_call) + tool(result) + assistant(final) = 4
	if len(result) != 4 {
		t.Fatalf("got %d messages, want 4", len(result))
	}
	if result[1].Role != "assistant" {
		t.Errorf("message[1] role = %q, want assistant", result[1].Role)
	}
	if len(result[1].ToolCalls) != 1 {
		t.Fatalf("expected 1 tool call, got %d", len(result[1].ToolCalls))
	}
	if result[1].ToolCalls[0].Function.Name != "echo" {
		t.Errorf("tool call name = %q, want echo", result[1].ToolCalls[0].Function.Name)
	}
	if result[2].Role != "tool" {
		t.Errorf("message[2] role = %q, want tool", result[2].Role)
	}
	if result[3].Content != "Done!" {
		t.Errorf("final content = %q, want Done!", result[3].Content)
	}
}

func TestRunStreamingCompletionError(t *testing.T) {
	completeFn := func(_ context.Context, req *api.ChatCompletionRequest) (<-chan runner.StreamEvent, error) {
		return nil, fmt.Errorf("connection refused")
	}

	messages := []api.Message{{Role: "user", Content: "test"}}
	cfg := StreamingConfig{
		Config: Config{Tools: newTestRegistry(&echoTool{})},
	}

	_, err := RunStreaming(context.Background(), completeFn, messages, cfg)
	if err == nil {
		t.Fatal("expected error")
	}
	if !strings.Contains(err.Error(), "connection refused") {
		t.Errorf("unexpected error: %v", err)
	}
}

func TestRunStreamingEmptyResponse(t *testing.T) {
	completeFn := func(_ context.Context, req *api.ChatCompletionRequest) (<-chan runner.StreamEvent, error) {
		ch := make(chan runner.StreamEvent)
		go func() {
			defer close(ch)
			// Send done immediately with no content — produces empty choices
			ch <- runner.StreamEvent{Done: true}
		}()
		return ch, nil
	}

	messages := []api.Message{{Role: "user", Content: "test"}}
	cfg := StreamingConfig{
		Config: Config{Tools: newTestRegistry(&echoTool{})},
	}

	// Should succeed with a single choice containing empty content
	result, err := RunStreaming(context.Background(), completeFn, messages, cfg)
	if err != nil {
		t.Fatal(err)
	}
	// user + assistant(empty) = 2
	if len(result) != 2 {
		t.Fatalf("got %d messages, want 2", len(result))
	}
}

func TestRunStreamingMaxIterations(t *testing.T) {
	callCount := 0
	completeFn := func(_ context.Context, req *api.ChatCompletionRequest) (<-chan runner.StreamEvent, error) {
		callCount++
		// Always return a tool call with different args to avoid stuck detection
		return makeStreamToolCallEvents("call_1", "echo", fmt.Sprintf(`{"n":"%d"}`, callCount)), nil
	}

	messages := []api.Message{{Role: "user", Content: "test"}}
	cfg := StreamingConfig{
		Config: Config{
			MaxIterations: 3,
			Tools:         newTestRegistry(&echoTool{}),
		},
	}

	_, err := RunStreaming(context.Background(), completeFn, messages, cfg)
	if err == nil {
		t.Fatal("expected error for max iterations")
	}
	if !strings.Contains(err.Error(), "maximum iterations") {
		t.Errorf("unexpected error: %v", err)
	}
}

func TestRunStreamingStuckDetection(t *testing.T) {
	completeFn := func(_ context.Context, req *api.ChatCompletionRequest) (<-chan runner.StreamEvent, error) {
		return makeStreamToolCallEvents("call_1", "fail", `{"same":"args"}`), nil
	}

	messages := []api.Message{{Role: "user", Content: "test"}}
	cfg := StreamingConfig{
		Config: Config{
			MaxIterations: 20,
			Tools:         newTestRegistry(&failTool{}),
		},
	}

	_, err := RunStreaming(context.Background(), completeFn, messages, cfg)
	if err == nil {
		t.Fatal("expected error for stuck loop")
	}
	if !strings.Contains(err.Error(), "agent stuck") {
		t.Errorf("unexpected error: %v", err)
	}
}

func TestRunStreamingHooks(t *testing.T) {
	var (
		iterationStarts []int
		thinkingCalled  bool
		thinkingDone    bool
		toolCalls       []string
		toolResults     []string
	)

	callCount := 0
	completeFn := func(_ context.Context, req *api.ChatCompletionRequest) (<-chan runner.StreamEvent, error) {
		callCount++
		if callCount == 1 {
			return makeStreamToolCallEvents("call_1", "echo", `{"x":"y"}`), nil
		}
		return makeStreamEvents("Final answer"), nil
	}

	messages := []api.Message{{Role: "user", Content: "test"}}
	cfg := StreamingConfig{
		Config: Config{
			Tools: newTestRegistry(&echoTool{}),
			Hooks: Hooks{
				OnToolCall:   func(call api.ToolCall) { toolCalls = append(toolCalls, call.Function.Name) },
				OnToolResult: func(call api.ToolCall, result string) { toolResults = append(toolResults, result) },
			},
		},
		OnIterationStart: func(iteration, max int) { iterationStarts = append(iterationStarts, iteration) },
		OnThinking:       func() { thinkingCalled = true },
		OnThinkingDone:   func() { thinkingDone = true },
		OnContentDelta:   func(delta string) {},
	}

	_, err := RunStreaming(context.Background(), completeFn, messages, cfg)
	if err != nil {
		t.Fatal(err)
	}

	if !thinkingCalled {
		t.Error("OnThinking was not called")
	}
	if !thinkingDone {
		t.Error("OnThinkingDone was not called")
	}
	if len(iterationStarts) != 2 {
		t.Errorf("expected 2 iteration starts, got %d", len(iterationStarts))
	}
	if len(toolCalls) != 1 || toolCalls[0] != "echo" {
		t.Errorf("unexpected tool calls: %v", toolCalls)
	}
	if len(toolResults) != 1 {
		t.Errorf("expected 1 tool result, got %d", len(toolResults))
	}
}

func TestRunStreamingStreamError(t *testing.T) {
	completeFn := func(_ context.Context, req *api.ChatCompletionRequest) (<-chan runner.StreamEvent, error) {
		ch := make(chan runner.StreamEvent)
		go func() {
			defer close(ch)
			ch <- runner.StreamEvent{Err: fmt.Errorf("stream broken")}
		}()
		return ch, nil
	}

	messages := []api.Message{{Role: "user", Content: "test"}}
	cfg := StreamingConfig{
		Config:     Config{Tools: newTestRegistry(&echoTool{})},
		OnThinking: func() {},
	}

	_, err := RunStreaming(context.Background(), completeFn, messages, cfg)
	if err == nil {
		t.Fatal("expected error for broken stream")
	}
	if !strings.Contains(err.Error(), "stream broken") {
		t.Errorf("unexpected error: %v", err)
	}
}

func TestRunStreamingUnknownTool(t *testing.T) {
	callCount := 0
	completeFn := func(_ context.Context, req *api.ChatCompletionRequest) (<-chan runner.StreamEvent, error) {
		callCount++
		if callCount == 1 {
			return makeStreamToolCallEvents("call_1", "nonexistent", `{}`), nil
		}
		return makeStreamEvents("ok"), nil
	}

	messages := []api.Message{{Role: "user", Content: "test"}}
	cfg := StreamingConfig{
		Config: Config{Tools: newTestRegistry(&echoTool{})},
	}

	result, err := RunStreaming(context.Background(), completeFn, messages, cfg)
	if err != nil {
		t.Fatal(err)
	}
	// The tool result should indicate unknown tool
	if result[2].Role != "tool" {
		t.Fatalf("expected tool message at index 2")
	}
	if !strings.Contains(result[2].Content, "unknown tool") {
		t.Errorf("expected unknown tool error, got %q", result[2].Content)
	}
}

// --- accumulateWithCallbacks tests ---

func TestAccumulateWithCallbacksContentDelta(t *testing.T) {
	var deltas []string
	events := makeStreamEvents("Hello world")
	cfg := &StreamingConfig{
		OnContentDelta: func(delta string) { deltas = append(deltas, delta) },
		OnThinkingDone: func() {},
	}

	resp, err := accumulateWithCallbacks(events, cfg)
	if err != nil {
		t.Fatal(err)
	}

	if resp.Choices[0].Message.Content != "Hello world" {
		t.Errorf("content = %q, want %q", resp.Choices[0].Message.Content, "Hello world")
	}
	if len(deltas) == 0 {
		t.Error("expected deltas")
	}
	if resp.Choices[0].FinishReason != "stop" {
		t.Errorf("finish_reason = %q, want stop", resp.Choices[0].FinishReason)
	}
}

func TestAccumulateWithCallbacksToolCall(t *testing.T) {
	events := makeStreamToolCallEvents("call_1", "my_tool", `{"arg":"val"}`)
	cfg := &StreamingConfig{}

	resp, err := accumulateWithCallbacks(events, cfg)
	if err != nil {
		t.Fatal(err)
	}

	msg := resp.Choices[0].Message
	if len(msg.ToolCalls) != 1 {
		t.Fatalf("expected 1 tool call, got %d", len(msg.ToolCalls))
	}
	if msg.ToolCalls[0].Function.Name != "my_tool" {
		t.Errorf("tool name = %q, want my_tool", msg.ToolCalls[0].Function.Name)
	}
	if msg.ToolCalls[0].Function.Arguments != `{"arg":"val"}` {
		t.Errorf("tool args = %q, want %q", msg.ToolCalls[0].Function.Arguments, `{"arg":"val"}`)
	}
	if resp.Choices[0].FinishReason != "tool_calls" {
		t.Errorf("finish_reason = %q, want tool_calls", resp.Choices[0].FinishReason)
	}
}

func TestAccumulateWithCallbacksError(t *testing.T) {
	ch := make(chan runner.StreamEvent)
	go func() {
		defer close(ch)
		ch <- runner.StreamEvent{Err: fmt.Errorf("broken")}
	}()

	thinkingDoneCalled := false
	cfg := &StreamingConfig{
		OnThinkingDone: func() { thinkingDoneCalled = true },
	}

	_, err := accumulateWithCallbacks(ch, cfg)
	if err == nil {
		t.Fatal("expected error")
	}
	if !thinkingDoneCalled {
		t.Error("OnThinkingDone should be called on error")
	}
}

func TestAccumulateWithCallbacksNilChunk(t *testing.T) {
	ch := make(chan runner.StreamEvent)
	go func() {
		defer close(ch)
		ch <- runner.StreamEvent{Chunk: nil} // nil chunk should be skipped
		ch <- runner.StreamEvent{
			Chunk: &api.ChatCompletionChunk{
				Choices: []api.ChunkChoice{
					{Delta: api.MessageDelta{Content: "ok"}},
				},
			},
		}
		ch <- runner.StreamEvent{Done: true}
	}()

	cfg := &StreamingConfig{}
	resp, err := accumulateWithCallbacks(ch, cfg)
	if err != nil {
		t.Fatal(err)
	}
	if resp.Choices[0].Message.Content != "ok" {
		t.Errorf("content = %q, want ok", resp.Choices[0].Message.Content)
	}
}

func TestAccumulateThinkingDoneOnNoContent(t *testing.T) {
	// If no content is ever received, OnThinkingDone should still fire at the end
	ch := make(chan runner.StreamEvent)
	go func() {
		defer close(ch)
		ch <- runner.StreamEvent{Done: true}
	}()

	thinkingDoneCalled := false
	cfg := &StreamingConfig{
		OnThinkingDone: func() { thinkingDoneCalled = true },
	}

	_, err := accumulateWithCallbacks(ch, cfg)
	if err != nil {
		t.Fatal(err)
	}
	if !thinkingDoneCalled {
		t.Error("OnThinkingDone should be called even with no content")
	}
}

// --- truncateToolResults tests ---

func TestTruncateToolResultsUnderBudget(t *testing.T) {
	estimator := chatctx.NewTokenEstimator()
	messages := []api.Message{
		{Role: "user", Content: "hello"},
		{Role: "assistant", Content: "hi"},
	}

	result := truncateToolResults(messages, 10000, estimator)
	// Should be unchanged
	if len(result) != 2 {
		t.Fatalf("expected 2 messages, got %d", len(result))
	}
	if result[0].Content != "hello" {
		t.Errorf("content modified when under budget")
	}
}

func TestTruncateToolResultsOverBudget(t *testing.T) {
	estimator := chatctx.NewTokenEstimator()

	// Create a very large tool result
	largeContent := strings.Repeat("x", 5000)
	messages := []api.Message{
		{Role: "user", Content: "hello"},
		{Role: "assistant", Content: "let me check"},
		{Role: "tool", Content: largeContent, ToolCallID: "call_1", Name: "file_read"},
		{Role: "assistant", Content: "done"},
	}

	// Set max to something small to force truncation
	result := truncateToolResults(messages, 100, estimator)

	// The tool result should have been truncated
	if len(result[2].Content) >= len(largeContent) {
		t.Errorf("tool result was not truncated: len=%d", len(result[2].Content))
	}
	if !strings.Contains(result[2].Content, "[truncated") {
		t.Error("truncated message should contain truncation notice")
	}

	// Original should not be modified
	if messages[2].Content != largeContent {
		t.Error("original messages were mutated")
	}
}

func TestTruncateToolResultsSkipsSmallResults(t *testing.T) {
	estimator := chatctx.NewTokenEstimator()

	messages := []api.Message{
		{Role: "user", Content: "hello"},
		{Role: "tool", Content: "small", ToolCallID: "call_1", Name: "echo"},
		{Role: "tool", Content: strings.Repeat("big content ", 500), ToolCallID: "call_2", Name: "file_read"},
		{Role: "assistant", Content: "done"},
	}

	result := truncateToolResults(messages, 100, estimator)

	// Small tool result should be unchanged
	if result[1].Content != "small" {
		t.Errorf("small tool result was modified: %q", result[1].Content)
	}
	// Large tool result should be truncated
	if len(result[2].Content) >= len(messages[2].Content) {
		t.Error("large tool result was not truncated")
	}
}

func TestTruncateToolResultsSkipsNonToolMessages(t *testing.T) {
	estimator := chatctx.NewTokenEstimator()

	largeContent := strings.Repeat("x", 5000)
	messages := []api.Message{
		{Role: "user", Content: largeContent},
		{Role: "assistant", Content: "ok"},
	}

	result := truncateToolResults(messages, 100, estimator)

	// User message should not be truncated (only tool messages are)
	if result[0].Content != largeContent {
		t.Error("non-tool message was truncated")
	}
}
