package agent

import (
	"context"
	"fmt"
	"strings"

	"github.com/ThatCatDev/tanrenai/shared/apiclient"
	"github.com/ThatCatDev/tanrenai/shared/pkg/api"
	"github.com/ThatCatDev/tanrenai/shared/tools"
)

// CompletionFunc sends a chat completion request and returns the response.
type CompletionFunc func(ctx context.Context, req *api.ChatCompletionRequest) (*api.ChatCompletionResponse, error)

// ApprovalAction is the user's response to a tool approval prompt.
type ApprovalAction int

const (
	ApprovalAllow       ApprovalAction = iota // allow this one call
	ApprovalBlock                             // block this one call
	ApprovalAlwaysAllow                       // allow and remember
)

// Hooks are optional callbacks invoked during the agent loop for observability.
type Hooks struct {
	OnAssistantMessage func(content string)
	OnToolCall         func(call api.ToolCall)
	OnToolResult       func(call api.ToolCall, result *tools.ToolResult)
	// OnToolApproval is called before executing a tool. It blocks until the
	// user responds. If nil, all tools are auto-approved.
	OnToolApproval func(call api.ToolCall) ApprovalAction
}

// Config configures the agent loop.
type Config struct {
	MaxIterations     int
	Tools             *tools.Registry
	Hooks             Hooks
	MaxResponseTokens int // max tokens per generation (0 = default 4096)
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

// processToolCalls executes each tool call, tracks error counts for stuck
// detection, fires hooks, and returns the resulting tool messages plus whether
// any call succeeded (i.e. the agent is not stuck). A fatal execution error
// is returned as err.
func processToolCalls(ctx context.Context, toolCalls []api.ToolCall, registry *tools.Registry, hooks Hooks, errorCounts map[string]int) (msgs []api.Message, anySuccess bool, err error) {
	for idx, tc := range toolCalls {
		// Ensure tool calls have IDs — some backends (e.g. llama-server) may
		// not generate them, but they're required for matching tool results.
		if tc.ID == "" {
			tc.ID = fmt.Sprintf("call_%d", idx)
			toolCalls[idx] = tc
		}
		if hooks.OnToolCall != nil {
			hooks.OnToolCall(tc)
		}

		// Check tool approval before executing.
		if hooks.OnToolApproval != nil {
			action := hooks.OnToolApproval(tc)
			if action == ApprovalBlock {
				result := tools.ErrorResult("Tool call blocked by user.")
				msgs = append(msgs, api.Message{
					Role:       "tool",
					Content:    result.Output,
					ToolCallID: tc.ID,
					Name:       tc.Function.Name,
				})
				if hooks.OnToolResult != nil {
					hooks.OnToolResult(tc, result)
				}
				continue
			}
		}

		tool := registry.Get(tc.Function.Name)
		var result *tools.ToolResult
		if tool == nil {
			result = tools.ErrorResult(fmt.Sprintf("unknown tool: %s", tc.Function.Name))
		} else {
			var execErr error
			result, execErr = tool.Execute(ctx, tc.Function.Arguments)
			if execErr != nil {
				// Convert Go-level errors to tool-level errors so the LLM
				// always sees feedback and can respond or try a different approach.
				result = tools.ErrorResult(fmt.Sprintf("tool execution error: %v", execErr))
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
			anySuccess = true
		}

		if hooks.OnToolResult != nil {
			hooks.OnToolResult(tc, result)
		}

		msgs = append(msgs, api.Message{
			Role:       "tool",
			Content:    result.Output,
			ToolCallID: tc.ID,
			Name:       tc.Function.Name,
		})
	}
	return msgs, anySuccess, nil
}

// checkStuck returns true when every tool call has hit the error-repeat
// threshold and at least one has exceeded it, meaning the agent should stop.
func checkStuck(toolCalls []api.ToolCall, errorCounts map[string]int) bool {
	for _, tc := range toolCalls {
		if errorCounts[toolCallKey(tc)] < maxConsecutiveErrors {
			return false
		}
	}
	// All at or above threshold — check if any exceeded it.
	for _, tc := range toolCalls {
		if errorCounts[toolCallKey(tc)] > maxConsecutiveErrors {
			return true
		}
	}
	return false
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

		toolMsgs, anySuccess, err := processToolCalls(ctx, choice.Message.ToolCalls, cfg.Tools, cfg.Hooks, errorCounts)
		if err != nil {
			return messages, err
		}
		messages = append(messages, toolMsgs...)

		if !anySuccess && checkStuck(choice.Message.ToolCalls, errorCounts) {
			return messages, fmt.Errorf("agent stuck: repeated identical failing tool calls")
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
		resp, hadReasoning, err := accumulateWithCallbacks(events, &cfg)
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
			// Thinking models may produce only reasoning_content and stop
			// without visible content. Nudge them to produce a response.
			if nudgeCount < maxNudges && (looksLikeContinuation(choice.Message.Content) || (choice.Message.Content == "" && hadReasoning)) {
				nudgeCount++
				if cfg.OnContentDelta != nil {
					cfg.OnContentDelta("\n[continuing...]\n")
				}
				messages = append(messages, api.Message{
					Role:    "user",
					Content: "Please provide your response to the user.",
				})
				continue
			}
			return messages, nil
		}

		toolMsgs, anySuccess, err := processToolCalls(ctx, choice.Message.ToolCalls, cfg.Tools, cfg.Hooks, errorCounts)
		if err != nil {
			return messages, err
		}
		messages = append(messages, toolMsgs...)

		if !anySuccess && checkStuck(choice.Message.ToolCalls, errorCounts) {
			return messages, fmt.Errorf("agent stuck: repeated identical failing tool calls")
		}
	}

	return messages, fmt.Errorf("agent loop reached maximum iterations (%d)", cfg.MaxIterations)
}

func accumulateWithCallbacks(events <-chan apiclient.StreamEvent, cfg *StreamingConfig) (*api.ChatCompletionResponse, bool, error) {
	var (
		content      strings.Builder
		role         string
		model        string
		id           string
		finishReason string
		toolCalls    []api.ToolCall
		toolArgBuf   = make(map[int]*strings.Builder)
		gotContent   bool
		gotReasoning bool
		thinkingDone bool
	)

	for ev := range events {
		if ev.Err != nil {
			if !thinkingDone && cfg.OnThinkingDone != nil {
				cfg.OnThinkingDone()
			}
			return nil, false, ev.Err
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
			if choice.Delta.ReasoningContent != "" {
				gotReasoning = true
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
				Index: 0,
				Message:      msg,
				FinishReason: finishReason,
			},
		},
	}, gotReasoning, nil
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
