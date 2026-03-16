package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"strings"

	"github.com/ThatCatDev/tanrenai/shared/apiclient"
	"github.com/ThatCatDev/tanrenai/shared/pkg/api"
	"github.com/ThatCatDev/tanrenai/shared/tools"
)

// debugEnabled returns true when TANRENAI_DEBUG is set.
func debugEnabled() bool { return os.Getenv("TANRENAI_DEBUG") != "" }

func debugf(format string, args ...any) {
	if !debugEnabled() {
		return
	}
	f, err := os.OpenFile("/tmp/tanrenai-debug.log", os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0644)
	if err != nil {
		return
	}
	defer func() { _ = f.Close() }()
	_, _ = fmt.Fprintf(f, "[agent] "+format+"\n", args...)
}

// TokenEstimator is an optional interface for estimating token counts.
// If provided via Config.TokenEstimator, it enables tool result truncation.
type TokenEstimator interface {
	Estimate(text string) int
	EstimateMessages(msgs []api.Message) int
}

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
	MaxTokens         int            // 0 = no limit (backward compatible)
	MaxResponseTokens int            // max tokens per generation (0 = default 4096)
	TokenEstimator    TokenEstimator // nil = no estimation
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
	OnReasoningDelta func(delta string)
}

const (
	maxConsecutiveErrors     = 3
	defaultMaxResponseTokens = 16384
	maxRetries               = 3
)

func toolCallKey(tc api.ToolCall) string {
	return tc.Function.Name + ":" + tc.Function.Arguments
}

// processToolCalls executes each tool call, tracks error counts for stuck
// detection, fires hooks, and returns the resulting tool messages plus whether
// any call succeeded.
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
// threshold and at least one has exceeded it.
func checkStuck(toolCalls []api.ToolCall, errorCounts map[string]int) bool {
	for _, tc := range toolCalls {
		if errorCounts[toolCallKey(tc)] < maxConsecutiveErrors {
			return false
		}
	}
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
	retryCount := 0

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

		if choice.Message.Content != "" && cfg.Hooks.OnAssistantMessage != nil {
			cfg.Hooks.OnAssistantMessage(choice.Message.Content)
		}

		if choice.FinishReason == "length" && len(choice.Message.ToolCalls) == 0 {
			messages = append(messages, choice.Message)

			continue
		}

		if choice.FinishReason != "tool_calls" || len(choice.Message.ToolCalls) == 0 {
			isEmpty := choice.Message.Content == ""
			isContinuation := looksLikeContinuation(choice.Message.Content)

			if retryCount < maxRetries && (isEmpty || isContinuation) {
				retryCount++
				if retryCount == 1 {
					continue
				}
				messages = append(messages, api.Message{
					Role:    "user",
					Content: retryPrompt,
				})

				continue
			}
			messages = append(messages, choice.Message)

			return messages, nil
		}

		// Got tool calls — append and execute
		retryCount = 0
		messages = removeRetryPrompts(messages)
		messages = append(messages, choice.Message)

		toolMsgs, anySuccess, toolErr := processToolCalls(ctx, choice.Message.ToolCalls, cfg.Tools, cfg.Hooks, errorCounts)
		messages = append(messages, toolMsgs...)

		if toolErr != nil {
			return messages, toolErr
		}
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
	retryCount := 0

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

		if debugEnabled() {
			debugf("iter %d: sending %d messages", i+1, len(req.Messages))
			for j, m := range req.Messages {
				debugf("  msg[%d] role=%q content_len=%d tool_calls=%d tool_call_id=%q", j, m.Role, len(m.Content), len(m.ToolCalls), m.ToolCallID)
			}
			if i > 0 {
				reqJSON, _ := json.MarshalIndent(req, "", "  ")
				debugf("iter %d full request:\n%s", i+1, string(reqJSON))
			}
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
		debugf("iter %d: finish_reason=%q content_len=%d tool_calls=%d reasoning=%v", i+1, choice.FinishReason, len(choice.Message.Content), len(choice.Message.ToolCalls), hadReasoning)

		stripNarration(&choice.Message)

		if choice.FinishReason == "length" && len(choice.Message.ToolCalls) == 0 {
			messages = append(messages, choice.Message)
			if cfg.OnContentDelta != nil {
				cfg.OnContentDelta("\n[continuing...]\n")
			}

			continue
		}

		if choice.FinishReason != "tool_calls" || len(choice.Message.ToolCalls) == 0 {
			isEmpty := choice.Message.Content == ""
			isContinuation := looksLikeContinuation(choice.Message.Content)

			if retryCount < maxRetries && (isEmpty || isContinuation) {
				retryCount++
				debugf("iter %d: retry %d/%d (empty=%v continuation=%v reasoning=%v)", i+1, retryCount, maxRetries, isEmpty, isContinuation, hadReasoning)

				if retryCount == 1 {
					// First retry: silent (same messages, no injection)
					continue
				}
				// Subsequent retries: inject a short prompt to break the
				// model out of a reasoning-only loop, then remove it after.
				messages = append(messages, api.Message{
					Role:    "user",
					Content: retryPrompt,
				})

				continue
			}
			messages = append(messages, choice.Message)

			return messages, nil
		}

		// Got tool calls — append and execute
		retryCount = 0
		// Clean up any injected retry prompts
		messages = removeRetryPrompts(messages)
		messages = append(messages, choice.Message)

		toolMsgs, anySuccess, toolErr := processToolCalls(ctx, choice.Message.ToolCalls, cfg.Tools, cfg.Hooks, errorCounts)
		messages = append(messages, toolMsgs...)

		if toolErr != nil {
			return messages, toolErr
		}
		if !anySuccess && checkStuck(choice.Message.ToolCalls, errorCounts) {
			return messages, fmt.Errorf("agent stuck: repeated identical failing tool calls")
		}
	}

	return messages, fmt.Errorf("agent loop reached maximum iterations (%d)", cfg.MaxIterations)
}

// streamAccumulator holds mutable state while consuming a stream of events.
type streamAccumulator struct {
	content      strings.Builder
	role         string
	model        string
	id           string
	finishReason string
	toolCalls    []api.ToolCall
	toolArgBuf   map[int]*strings.Builder
	gotReasoning bool
	thinkingDone bool
}

func newStreamAccumulator() *streamAccumulator {
	return &streamAccumulator{toolArgBuf: make(map[int]*strings.Builder)}
}

// applyChunkMeta captures the stream ID and model name on first sight.
func (a *streamAccumulator) applyChunkMeta(chunk *api.ChatCompletionChunk) {
	if a.id == "" {
		a.id = chunk.ID
	}
	if a.model == "" {
		a.model = chunk.Model
	}
}

// applyChoiceDelta processes a single choice delta: role, finish reason,
// reasoning content, text content, and tool-call fragments.
func (a *streamAccumulator) applyChoiceDelta(choice api.ChunkChoice, cfg *StreamingConfig) {
	if choice.Delta.Role != "" {
		a.role = choice.Delta.Role
	}
	if choice.FinishReason != nil {
		a.finishReason = *choice.FinishReason
	}

	a.applyReasoningDelta(choice.Delta.ReasoningContent, cfg)
	a.applyContentDelta(choice.Delta.Content, cfg)
	a.applyToolCallDeltas(choice.Delta.ToolCalls)
}

// applyReasoningDelta handles the reasoning_content field from thinking models.
func (a *streamAccumulator) applyReasoningDelta(reasoning string, cfg *StreamingConfig) {
	if reasoning == "" {
		return
	}
	a.gotReasoning = true
	if cfg.OnReasoningDelta != nil {
		cfg.OnReasoningDelta(reasoning)
	}
}

// applyContentDelta handles a text content delta, firing the thinking-done
// callback on the first content token after a reasoning phase.
func (a *streamAccumulator) applyContentDelta(delta string, cfg *StreamingConfig) {
	if delta == "" {
		return
	}
	if !a.thinkingDone && cfg.OnThinkingDone != nil {
		cfg.OnThinkingDone()
		a.thinkingDone = true
	}
	a.content.WriteString(delta)
	if cfg.OnContentDelta != nil {
		cfg.OnContentDelta(delta)
	}
}

// applyToolCallDeltas accumulates streamed tool-call fragments.
func (a *streamAccumulator) applyToolCallDeltas(deltas []api.ToolCallDelta) {
	for _, tcd := range deltas {
		for len(a.toolCalls) <= tcd.Index {
			a.toolCalls = append(a.toolCalls, api.ToolCall{})
		}
		if tcd.ID != "" {
			a.toolCalls[tcd.Index].ID = tcd.ID
		}
		if tcd.Type != "" {
			a.toolCalls[tcd.Index].Type = tcd.Type
		}
		if tcd.Function != nil {
			if tcd.Function.Name != "" {
				a.toolCalls[tcd.Index].Function.Name = tcd.Function.Name
			}
			if tcd.Function.Arguments != "" {
				if a.toolArgBuf[tcd.Index] == nil {
					a.toolArgBuf[tcd.Index] = &strings.Builder{}
				}
				a.toolArgBuf[tcd.Index].WriteString(tcd.Function.Arguments)
			}
		}
	}
}

// finalizeToolCalls merges the per-index argument buffers into the tool calls.
func (a *streamAccumulator) finalizeToolCalls() {
	for idx, buf := range a.toolArgBuf {
		if idx < len(a.toolCalls) {
			a.toolCalls[idx].Function.Arguments = buf.String()
		}
	}
}

// buildResponse assembles the final ChatCompletionResponse from accumulated state.
func (a *streamAccumulator) buildResponse() *api.ChatCompletionResponse {
	role := a.role
	if role == "" {
		role = "assistant"
	}

	msg := api.Message{
		Role:    role,
		Content: a.content.String(),
	}
	if len(a.toolCalls) > 0 {
		msg.ToolCalls = a.toolCalls
	}

	finishReason := a.finishReason
	if finishReason == "" {
		finishReason = "stop"
		if len(a.toolCalls) > 0 {
			finishReason = "tool_calls"
		}
	}

	return &api.ChatCompletionResponse{
		ID:     a.id,
		Object: "chat.completion",
		Model:  a.model,
		Choices: []api.Choice{
			{
				Index:        0,
				Message:      msg,
				FinishReason: finishReason,
			},
		},
	}
}

func accumulateWithCallbacks(events <-chan apiclient.StreamEvent, cfg *StreamingConfig) (*api.ChatCompletionResponse, bool, error) {
	acc := newStreamAccumulator()

	for ev := range events {
		if ev.Err != nil {
			if !acc.thinkingDone && cfg.OnThinkingDone != nil {
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

		acc.applyChunkMeta(ev.Chunk)
		for _, choice := range ev.Chunk.Choices {
			acc.applyChoiceDelta(choice, cfg)
		}
	}

	if !acc.thinkingDone && cfg.OnThinkingDone != nil {
		cfg.OnThinkingDone()
	}

	acc.finalizeToolCalls()

	debugf("accumulation: reasoning=%v content_len=%d finish=%q", acc.gotReasoning, acc.content.Len(), acc.finishReason)

	return acc.buildResponse(), acc.gotReasoning, nil
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

const retryPrompt = "[system: empty response detected — please use your tools or provide a visible answer]"

// removeRetryPrompts strips any injected retry prompts from the message history.
func removeRetryPrompts(messages []api.Message) []api.Message {
	var cleaned []api.Message
	for _, m := range messages {
		if m.Role == "user" && m.Content == retryPrompt {
			continue
		}
		cleaned = append(cleaned, m)
	}

	return cleaned
}

func stripNarration(msg *api.Message) {
	if len(msg.ToolCalls) > 0 && msg.Content != "" {
		msg.Content = ""
	}
}

func truncateToolResults(messages []api.Message, maxTokens int, estimator TokenEstimator) []api.Message {
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
