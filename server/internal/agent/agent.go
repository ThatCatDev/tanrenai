package agent

import (
	"context"
	"fmt"
	"strings"

	"github.com/thatcatdev/tanrenai/server/internal/chatctx"
	"github.com/thatcatdev/tanrenai/server/internal/runner"
	"github.com/thatcatdev/tanrenai/server/internal/tools"
	"github.com/thatcatdev/tanrenai/server/pkg/api"
)

// CompletionFunc sends a chat completion request and returns the response.
// This abstraction decouples the agent from both the runner and HTTP client.
// CLI passes an HTTP-based func, server handler passes a runner-based func.
type CompletionFunc func(ctx context.Context, req *api.ChatCompletionRequest) (*api.ChatCompletionResponse, error)

// Hooks are optional callbacks invoked during the agent loop for observability.
type Hooks struct {
	OnAssistantMessage func(content string)
	OnToolCall         func(call api.ToolCall)
	OnToolResult       func(call api.ToolCall, result string)
}

// Config configures the agent loop.
type Config struct {
	MaxIterations  int
	Tools          *tools.Registry
	Hooks          Hooks
	MaxTokens      int                    // 0 = no limit (backward compatible)
	TokenEstimator *chatctx.TokenEstimator // nil = no estimation
}

// StreamingCompletionFunc returns a channel of stream events instead of blocking.
type StreamingCompletionFunc func(ctx context.Context, req *api.ChatCompletionRequest) (<-chan runner.StreamEvent, error)

// StreamingConfig extends Config with streaming hooks.
type StreamingConfig struct {
	Config
	OnIterationStart func(iteration, maxIterations int)
	OnThinking       func()
	OnThinkingDone   func()
	OnContentDelta   func(delta string)
}

const maxConsecutiveErrors = 3

// toolCallKey returns a string that identifies a tool call by name + arguments,
// used for detecting repeated identical calls.
func toolCallKey(tc api.ToolCall) string {
	return tc.Function.Name + ":" + tc.Function.Arguments
}

// Run executes the agentic loop: send messages to the LLM, execute any tool
// calls it makes, feed results back, and repeat until the model stops calling
// tools or the iteration limit is reached.
//
// Returns the final message history (including all tool calls and results).
func Run(ctx context.Context, complete CompletionFunc, messages []api.Message, cfg Config) ([]api.Message, error) {
	if cfg.MaxIterations <= 0 {
		cfg.MaxIterations = 20
	}

	apiTools := cfg.Tools.APITools()
	errorCounts := make(map[string]int) // track repeated failing calls

	for i := 0; i < cfg.MaxIterations; i++ {
		// Safety net: truncate large tool results if over token budget
		if cfg.MaxTokens > 0 && cfg.TokenEstimator != nil {
			messages = truncateToolResults(messages, cfg.MaxTokens, cfg.TokenEstimator)
		}

		req := &api.ChatCompletionRequest{
			Messages: messages,
			Stream:   false,
			Tools:    apiTools,
		}

		resp, err := complete(ctx, req)
		if err != nil {
			return messages, fmt.Errorf("completion request failed: %w", err)
		}

		if len(resp.Choices) == 0 {
			return messages, fmt.Errorf("empty response from model")
		}

		choice := resp.Choices[0]
		messages = append(messages, choice.Message)

		// Notify about assistant text content
		if choice.Message.Content != "" && cfg.Hooks.OnAssistantMessage != nil {
			cfg.Hooks.OnAssistantMessage(choice.Message.Content)
		}

		// If the model didn't make tool calls, we're done
		if choice.FinishReason != "tool_calls" || len(choice.Message.ToolCalls) == 0 {
			return messages, nil
		}

		// Execute each tool call and append results
		stuck := true
		for _, tc := range choice.Message.ToolCalls {
			if cfg.Hooks.OnToolCall != nil {
				cfg.Hooks.OnToolCall(tc)
			}

			tool := cfg.Tools.Get(tc.Function.Name)
			var result *tools.ToolResult
			if tool == nil {
				result = tools.ErrorResult(fmt.Sprintf("unknown tool: %s", tc.Function.Name))
			} else {
				var execErr error
				result, execErr = tool.Execute(ctx, tc.Function.Arguments)
				if execErr != nil {
					return messages, fmt.Errorf("tool %q execution error: %w", tc.Function.Name, execErr)
				}
			}

			// Track repeated failing calls to detect stuck loops
			key := toolCallKey(tc)
			if result.IsError {
				errorCounts[key]++
				if errorCounts[key] >= maxConsecutiveErrors {
					// Force a hint into the result so the model knows to stop
					result.Output += "\n\nYou have repeated this exact failing call multiple times. Do NOT retry it. Either try different arguments or respond to the user explaining what went wrong."
				}
			} else {
				delete(errorCounts, key)
				stuck = false
			}

			if cfg.Hooks.OnToolResult != nil {
				cfg.Hooks.OnToolResult(tc, result.Output)
			}

			messages = append(messages, api.Message{
				Role:       "tool",
				Content:    result.Output,
				ToolCallID: tc.ID,
				Name:       tc.Function.Name,
			})
		}

		// If every tool call in this iteration was a repeat failure, bail out
		allRepeats := true
		for _, tc := range choice.Message.ToolCalls {
			if errorCounts[toolCallKey(tc)] < maxConsecutiveErrors {
				allRepeats = false
				break
			}
		}
		if stuck && allRepeats {
			// Give the model one more chance with the hint appended above,
			// but if we've already done that, stop
			anyOverLimit := false
			for _, tc := range choice.Message.ToolCalls {
				if errorCounts[toolCallKey(tc)] > maxConsecutiveErrors {
					anyOverLimit = true
					break
				}
			}
			if anyOverLimit {
				return messages, fmt.Errorf("agent stuck: repeated identical failing tool calls")
			}
		}
	}

	return messages, fmt.Errorf("agent loop reached maximum iterations (%d)", cfg.MaxIterations)
}

// RunStreaming executes the agentic loop with streaming: text is delivered via
// callbacks as it generates, giving real-time feedback to the user. Tool call
// processing, stuck detection, and error tracking are identical to Run().
func RunStreaming(ctx context.Context, complete StreamingCompletionFunc, messages []api.Message, cfg StreamingConfig) ([]api.Message, error) {
	if cfg.MaxIterations <= 0 {
		cfg.MaxIterations = 20
	}

	apiTools := cfg.Tools.APITools()
	errorCounts := make(map[string]int)

	for i := 0; i < cfg.MaxIterations; i++ {
		if cfg.OnIterationStart != nil {
			cfg.OnIterationStart(i+1, cfg.MaxIterations)
		}

		if cfg.MaxTokens > 0 && cfg.TokenEstimator != nil {
			messages = truncateToolResults(messages, cfg.MaxTokens, cfg.TokenEstimator)
		}

		req := &api.ChatCompletionRequest{
			Messages: messages,
			Stream:   true,
			Tools:    apiTools,
		}

		if cfg.OnThinking != nil {
			cfg.OnThinking()
		}

		events, err := complete(ctx, req)
		if err != nil {
			if cfg.OnThinkingDone != nil {
				cfg.OnThinkingDone()
			}
			return messages, fmt.Errorf("completion request failed: %w", err)
		}

		resp, err := accumulateWithCallbacks(events, &cfg)
		if err != nil {
			return messages, fmt.Errorf("stream accumulation failed: %w", err)
		}

		if len(resp.Choices) == 0 {
			return messages, fmt.Errorf("empty response from model")
		}

		choice := resp.Choices[0]
		messages = append(messages, choice.Message)

		// If the model didn't make tool calls, we're done
		if choice.FinishReason != "tool_calls" || len(choice.Message.ToolCalls) == 0 {
			return messages, nil
		}

		// Execute each tool call and append results
		stuck := true
		for _, tc := range choice.Message.ToolCalls {
			if cfg.Hooks.OnToolCall != nil {
				cfg.Hooks.OnToolCall(tc)
			}

			tool := cfg.Tools.Get(tc.Function.Name)
			var result *tools.ToolResult
			if tool == nil {
				result = tools.ErrorResult(fmt.Sprintf("unknown tool: %s", tc.Function.Name))
			} else {
				var execErr error
				result, execErr = tool.Execute(ctx, tc.Function.Arguments)
				if execErr != nil {
					return messages, fmt.Errorf("tool %q execution error: %w", tc.Function.Name, execErr)
				}
			}

			key := toolCallKey(tc)
			if result.IsError {
				errorCounts[key]++
				if errorCounts[key] >= maxConsecutiveErrors {
					result.Output += "\n\nYou have repeated this exact failing call multiple times. Do NOT retry it. Either try different arguments or respond to the user explaining what went wrong."
				}
			} else {
				delete(errorCounts, key)
				stuck = false
			}

			if cfg.Hooks.OnToolResult != nil {
				cfg.Hooks.OnToolResult(tc, result.Output)
			}

			messages = append(messages, api.Message{
				Role:       "tool",
				Content:    result.Output,
				ToolCallID: tc.ID,
				Name:       tc.Function.Name,
			})
		}

		// If every tool call in this iteration was a repeat failure, bail out
		allRepeats := true
		for _, tc := range choice.Message.ToolCalls {
			if errorCounts[toolCallKey(tc)] < maxConsecutiveErrors {
				allRepeats = false
				break
			}
		}
		if stuck && allRepeats {
			anyOverLimit := false
			for _, tc := range choice.Message.ToolCalls {
				if errorCounts[toolCallKey(tc)] > maxConsecutiveErrors {
					anyOverLimit = true
					break
				}
			}
			if anyOverLimit {
				return messages, fmt.Errorf("agent stuck: repeated identical failing tool calls")
			}
		}
	}

	return messages, fmt.Errorf("agent loop reached maximum iterations (%d)", cfg.MaxIterations)
}

// accumulateWithCallbacks collects streaming chunks into a complete response,
// firing callbacks as content arrives. Based on runner.AccumulateResponse but
// with OnContentDelta and OnThinkingDone hooks.
func accumulateWithCallbacks(events <-chan runner.StreamEvent, cfg *StreamingConfig) (*api.ChatCompletionResponse, error) {
	var (
		content       strings.Builder
		role          string
		model         string
		id            string
		toolCalls     []api.ToolCall
		toolArgBuf    = make(map[int]*strings.Builder)
		gotContent    bool
		thinkingDone  bool
	)

	for ev := range events {
		if ev.Err != nil {
			if !thinkingDone && cfg.OnThinkingDone != nil {
				cfg.OnThinkingDone()
			}
			return nil, ev.Err
		}
		if ev.Done {
			break
		}
		if ev.Chunk == nil {
			continue
		}

		if id == "" {
			id = ev.Chunk.ID
		}
		if model == "" {
			model = ev.Chunk.Model
		}

		for _, choice := range ev.Chunk.Choices {
			if choice.Delta.Role != "" {
				role = choice.Delta.Role
			}
			if choice.Delta.Content != "" {
				if !thinkingDone && cfg.OnThinkingDone != nil {
					cfg.OnThinkingDone()
					thinkingDone = true
				}
				gotContent = true
				content.WriteString(choice.Delta.Content)
				if cfg.OnContentDelta != nil {
					cfg.OnContentDelta(choice.Delta.Content)
				}
			}

			for _, tcd := range choice.Delta.ToolCalls {
				for len(toolCalls) <= tcd.Index {
					toolCalls = append(toolCalls, api.ToolCall{})
				}
				if tcd.ID != "" {
					toolCalls[tcd.Index].ID = tcd.ID
				}
				if tcd.Type != "" {
					toolCalls[tcd.Index].Type = tcd.Type
				}
				if tcd.Function != nil {
					if tcd.Function.Name != "" {
						toolCalls[tcd.Index].Function.Name = tcd.Function.Name
					}
					if tcd.Function.Arguments != "" {
						if toolArgBuf[tcd.Index] == nil {
							toolArgBuf[tcd.Index] = &strings.Builder{}
						}
						toolArgBuf[tcd.Index].WriteString(tcd.Function.Arguments)
					}
				}
			}
		}
	}

	// Clear thinking spinner if no content was ever received
	if !thinkingDone && cfg.OnThinkingDone != nil {
		cfg.OnThinkingDone()
	}
	_ = gotContent

	// Finalize accumulated tool call arguments
	for idx, buf := range toolArgBuf {
		if idx < len(toolCalls) {
			toolCalls[idx].Function.Arguments = buf.String()
		}
	}

	if role == "" {
		role = "assistant"
	}

	msg := api.Message{
		Role:    role,
		Content: content.String(),
	}
	if len(toolCalls) > 0 {
		msg.ToolCalls = toolCalls
	}

	finishReason := "stop"
	if len(toolCalls) > 0 {
		finishReason = "tool_calls"
	}

	return &api.ChatCompletionResponse{
		ID:     id,
		Object: "chat.completion",
		Model:  model,
		Choices: []api.Choice{
			{
				Index:        0,
				Message:      msg,
				FinishReason: finishReason,
			},
		},
	}, nil
}

// truncateToolResults reduces the content of tool result messages (the largest
// messages) when the total estimated tokens exceed the budget. It works backwards
// from the oldest tool results, truncating their content.
func truncateToolResults(messages []api.Message, maxTokens int, estimator *chatctx.TokenEstimator) []api.Message {
	total := estimator.EstimateMessages(messages)
	if total <= maxTokens {
		return messages
	}

	// Work on a copy to avoid mutating the caller's slice
	msgs := make([]api.Message, len(messages))
	copy(msgs, messages)

	// Truncate tool results from oldest to newest until under budget
	for i := range msgs {
		if msgs[i].Role != "tool" || msgs[i].Content == "" {
			continue
		}

		contentTokens := estimator.Estimate(msgs[i].Content)
		if contentTokens <= 50 {
			continue // already small, skip
		}

		// Truncate to ~50 tokens worth of characters
		maxChars := 50 * 4 // conservative estimate
		if len(msgs[i].Content) > maxChars {
			msgs[i].Content = msgs[i].Content[:maxChars] + "\n[truncated to fit context window]"
		}

		total = estimator.EstimateMessages(msgs)
		if total <= maxTokens {
			break
		}
	}

	return msgs
}
