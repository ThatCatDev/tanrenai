package agent

import (
	"context"
	"fmt"
	"strings"

	"github.com/ThatCatDev/tanrenai/server/internal/chatctx"
	"github.com/ThatCatDev/tanrenai/server/internal/runner"
	"github.com/ThatCatDev/tanrenai/server/internal/tools"
	"github.com/ThatCatDev/tanrenai/server/pkg/api"
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
	MaxIterations     int
	Tools             *tools.Registry
	Hooks             Hooks
	MaxTokens         int                    // 0 = no limit (backward compatible)
	MaxResponseTokens int                    // max tokens per generation (0 = default 4096)
	TokenEstimator    *chatctx.TokenEstimator // nil = no estimation
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

const (
	maxConsecutiveErrors     = 3
	defaultMaxResponseTokens = 4096 // generous default so tool calls aren't truncated
	maxNudges                = 3    // max times we'll nudge the model to continue using tools
)

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
	if cfg.MaxResponseTokens <= 0 {
		cfg.MaxResponseTokens = defaultMaxResponseTokens
	}

	apiTools := cfg.Tools.APITools()
	errorCounts := make(map[string]int) // track repeated failing calls
	nudgeCount := 0

	for i := 0; i < cfg.MaxIterations; i++ {
		// Safety net: truncate large tool results if over token budget
		if cfg.MaxTokens > 0 && cfg.TokenEstimator != nil {
			messages = truncateToolResults(messages, cfg.MaxTokens, cfg.TokenEstimator)
		}

		maxTokens := cfg.MaxResponseTokens
		req := &api.ChatCompletionRequest{
			Messages:  messages,
			Stream:    false,
			Tools:     apiTools,
			MaxTokens: &maxTokens,
		}

		resp, err := complete(ctx, req)
		if err != nil {
			return messages, fmt.Errorf("completion request failed: %w", err)
		}

		if len(resp.Choices) == 0 {
			return messages, fmt.Errorf("empty response from model")
		}

		choice := resp.Choices[0]

		// Strip narration text from messages that also have tool calls
		stripNarration(&choice.Message)

		messages = append(messages, choice.Message)

		// Notify about assistant text content
		if choice.Message.Content != "" && cfg.Hooks.OnAssistantMessage != nil {
			cfg.Hooks.OnAssistantMessage(choice.Message.Content)
		}

		// If truncated by max_tokens ("length"), continue so the model can finish
		if choice.FinishReason == "length" && len(choice.Message.ToolCalls) == 0 {
			continue
		}

		// If the model didn't make tool calls, check if it was about to
		if choice.FinishReason != "tool_calls" || len(choice.Message.ToolCalls) == 0 {
			// Nudge: if the text looks like the model intended to continue using
			// tools but stopped, inject a continuation message and keep going.
			if nudgeCount < maxNudges && looksLikeContinuation(choice.Message.Content) {
				nudgeCount++
				messages = append(messages, api.Message{
					Role:    "user",
					Content: "Do not guess or speculate. Use your tools to gather the actual information, then answer.",
				})
				continue
			}
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
	if cfg.MaxResponseTokens <= 0 {
		cfg.MaxResponseTokens = defaultMaxResponseTokens
	}

	apiTools := cfg.Tools.APITools()
	errorCounts := make(map[string]int)
	nudgeCount := 0

	for i := 0; i < cfg.MaxIterations; i++ {
		if cfg.OnIterationStart != nil {
			cfg.OnIterationStart(i+1, cfg.MaxIterations)
		}

		if cfg.MaxTokens > 0 && cfg.TokenEstimator != nil {
			messages = truncateToolResults(messages, cfg.MaxTokens, cfg.TokenEstimator)
		}

		maxTokens := cfg.MaxResponseTokens
		req := &api.ChatCompletionRequest{
			Messages:  messages,
			Stream:    true,
			Tools:     apiTools,
			MaxTokens: &maxTokens,
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

		// Strip narration text from messages that also have tool calls.
		// This saves tokens and sets a cleaner example in the context.
		stripNarration(&choice.Message)

		messages = append(messages, choice.Message)

		// If truncated by max_tokens ("length"), continue so the model can finish
		if choice.FinishReason == "length" && len(choice.Message.ToolCalls) == 0 {
			if cfg.OnContentDelta != nil {
				cfg.OnContentDelta("\n[continuing...]\n")
			}
			continue
		}

		// If the model didn't make tool calls, check if it was about to
		if choice.FinishReason != "tool_calls" || len(choice.Message.ToolCalls) == 0 {
			// Nudge: if the text looks like the model intended to continue using
			// tools but stopped, inject a continuation message and keep going.
			if nudgeCount < maxNudges && looksLikeContinuation(choice.Message.Content) {
				nudgeCount++
				if cfg.OnContentDelta != nil {
					cfg.OnContentDelta("\n[continuing...]\n")
				}
				messages = append(messages, api.Message{
					Role:    "user",
					Content: "Do not guess or speculate. Use your tools to gather the actual information, then answer.",
				})
				continue
			}
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
		finishReason  string
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
			// Capture finish_reason from the stream (set on last chunk)
			if choice.FinishReason != nil {
				finishReason = *choice.FinishReason
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

	// Use the actual finish_reason from the stream; fall back to inference
	if finishReason == "" {
		finishReason = "stop"
		if len(toolCalls) > 0 {
			finishReason = "tool_calls"
		}
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

// looksLikeContinuation checks if the model's text response suggests it intended
// to take more actions but stopped prematurely. This is general-purpose — it
// detects both "I'll do X next" narration and speculative answers where the
// model could have used tools to get real data.
func looksLikeContinuation(text string) bool {
	lower := strings.ToLower(text)

	// The model announced it will do something next but didn't call a tool
	intentPrefixes := []string{
		"let's ", "let me ", "i'll ", "i will ", "i'm going to ",
		"next,", "next ", "now,", "now ", "please wait",
		"here are the function calls", "here are the tool calls",
	}
	for _, sig := range intentPrefixes {
		if strings.Contains(lower, sig) {
			return true
		}
	}

	// The model is speculating/guessing when it could look things up.
	// Count speculation phrases — 2+ means it's clearly not using tools enough.
	specSignals := []string{
		"typically", "likely", "might be", "might contain",
		"may be", "may contain", "could be", "could contain",
		"probably", "presumably", "unknown", "unclear",
		"further investigation", "further exploration",
		"would need to", "need to check", "need to verify",
	}
	specCount := 0
	for _, sig := range specSignals {
		if strings.Contains(lower, sig) {
			specCount++
		}
	}
	return specCount >= 2
}

// stripNarration removes text content from assistant messages that also have
// tool calls. This reduces token waste from narration like "Let me read that file"
// that precedes actual tool calls, and sets a cleaner example in the context.
func stripNarration(msg *api.Message) {
	if len(msg.ToolCalls) > 0 && msg.Content != "" {
		msg.Content = ""
	}
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
