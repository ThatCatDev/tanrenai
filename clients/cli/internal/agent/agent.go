package agent

import (
	"context"
	"fmt"
	"strings"

	"github.com/ThatCatDev/tanrenai/client/internal/apiclient"
	"github.com/ThatCatDev/tanrenai/client/internal/chatctx"
	"github.com/ThatCatDev/tanrenai/client/internal/tools"
	"github.com/ThatCatDev/tanrenai/client/pkg/api"
)

// CompletionFunc sends a chat completion request and returns the response.
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
type StreamingCompletionFunc func(ctx context.Context, req *api.ChatCompletionRequest) (<-chan apiclient.StreamEvent, error)

// StreamingConfig extends Config with streaming hooks.
type StreamingConfig struct {
	Config
	OnIterationStart func(iteration, maxIterations int, messages []api.Message)
	OnThinking       func()
	OnThinkingDone   func()
	OnContentDelta   func(delta string)
}

const (
	maxConsecutiveErrors     = 3
	defaultMaxResponseTokens = 4096
	maxNudges                = 3
)

func toolCallKey(tc api.ToolCall) string {
	return tc.Function.Name + ":" + tc.Function.Arguments
}

// Run executes the agentic loop: send messages to the LLM, execute any tool
// calls it makes, feed results back, and repeat until the model stops calling
// tools or the iteration limit is reached.
func Run(ctx context.Context, complete CompletionFunc, messages []api.Message, cfg Config) ([]api.Message, error) {
	if cfg.MaxIterations <= 0 {
		cfg.MaxIterations = 1<<31 - 1
	}
	if cfg.MaxResponseTokens <= 0 {
		cfg.MaxResponseTokens = defaultMaxResponseTokens
	}

	apiTools := cfg.Tools.APITools()
	errorCounts := make(map[string]int)
	nudgeCount := 0

	for i := 0; i < cfg.MaxIterations; i++ {
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
		stripNarration(&choice.Message)
		messages = append(messages, choice.Message)

		if choice.Message.Content != "" && cfg.Hooks.OnAssistantMessage != nil {
			cfg.Hooks.OnAssistantMessage(choice.Message.Content)
		}

		if choice.FinishReason == "length" && len(choice.Message.ToolCalls) == 0 {
			continue
		}

		if choice.FinishReason != "tool_calls" || len(choice.Message.ToolCalls) == 0 {
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

// RunStreaming executes the agentic loop with streaming.
func RunStreaming(ctx context.Context, complete StreamingCompletionFunc, messages []api.Message, cfg StreamingConfig) ([]api.Message, error) {
	if cfg.MaxIterations <= 0 {
		cfg.MaxIterations = 1<<31 - 1
	}
	if cfg.MaxResponseTokens <= 0 {
		cfg.MaxResponseTokens = defaultMaxResponseTokens
	}

	apiTools := cfg.Tools.APITools()
	errorCounts := make(map[string]int)
	nudgeCount := 0

	for i := 0; i < cfg.MaxIterations; i++ {
		if cfg.OnIterationStart != nil {
			cfg.OnIterationStart(i+1, cfg.MaxIterations, messages)
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
		stripNarration(&choice.Message)
		messages = append(messages, choice.Message)

		if choice.FinishReason == "length" && len(choice.Message.ToolCalls) == 0 {
			if cfg.OnContentDelta != nil {
				cfg.OnContentDelta("\n[continuing...]\n")
			}
			continue
		}

		if choice.FinishReason != "tool_calls" || len(choice.Message.ToolCalls) == 0 {
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

func accumulateWithCallbacks(events <-chan apiclient.StreamEvent, cfg *StreamingConfig) (*api.ChatCompletionResponse, error) {
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

	if !thinkingDone && cfg.OnThinkingDone != nil {
		cfg.OnThinkingDone()
	}
	_ = gotContent

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

func looksLikeContinuation(text string) bool {
	lower := strings.ToLower(text)

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

func stripNarration(msg *api.Message) {
	if len(msg.ToolCalls) > 0 && msg.Content != "" {
		msg.Content = ""
	}
}

func truncateToolResults(messages []api.Message, maxTokens int, estimator *chatctx.TokenEstimator) []api.Message {
	total := estimator.EstimateMessages(messages)
	if total <= maxTokens {
		return messages
	}

	msgs := make([]api.Message, len(messages))
	copy(msgs, messages)

	for i := range msgs {
		if msgs[i].Role != "tool" || msgs[i].Content == "" {
			continue
		}

		contentTokens := estimator.Estimate(msgs[i].Content)
		if contentTokens <= 50 {
			continue
		}

		maxChars := 50 * 4
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
