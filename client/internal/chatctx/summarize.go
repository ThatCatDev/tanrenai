package chatctx

import (
	"context"
	"fmt"

	"github.com/ThatCatDev/tanrenai/client/pkg/api"
)

// CompletionFunc sends a chat completion request and returns the response.
type CompletionFunc func(ctx context.Context, req *api.ChatCompletionRequest) (*api.ChatCompletionResponse, error)

const summarizationPrompt = `Summarize the following conversation concisely. Preserve:
- Key facts and decisions made
- File paths and code references mentioned
- Tool results and their outcomes
- User preferences and requirements stated
- Any errors encountered and how they were resolved

Be concise but thorough. This summary will replace the original messages to save context space.`

// Summarize condenses older messages that won't fit in the context window.
// It calls the LLM to generate a summary, then stores it in the Manager.
// The summary replaces evicted messages when Messages() builds the window.
func (m *Manager) Summarize(ctx context.Context, complete CompletionFunc) error {
	if !m.NeedsSummary() {
		return nil
	}

	// Figure out which messages are being evicted (won't fit in window)
	systemMsgs := m.buildSystemMessages()
	systemTokens := m.estimator.EstimateMessages(systemMsgs)
	available := m.cfg.CtxSize - systemTokens - m.cfg.ResponseBudget

	if m.summary != "" {
		sm := api.Message{
			Role:    "system",
			Content: fmt.Sprintf("[Conversation summary] %s", m.summary),
		}
		available -= m.estimator.EstimateMessages([]api.Message{sm})
	}

	// Find cutoff: walk backwards
	cutoff := len(m.history)
	used := 0
	for i := len(m.history) - 1; i >= 0; i-- {
		msgTokens := m.estimator.EstimateMessages([]api.Message{m.history[i]})
		if used+msgTokens > available {
			break
		}
		used += msgTokens
		cutoff = i
	}

	if cutoff == 0 {
		return nil // nothing to summarize
	}

	// Collect messages to summarize (the ones that would be evicted)
	toSummarize := m.history[:cutoff]

	// Cap summarization input to 50% of context window to avoid overflow
	maxSummarizeTokens := m.cfg.CtxSize / 2
	summarizeTokens := 0
	startIdx := 0
	for i := len(toSummarize) - 1; i >= 0; i-- {
		t := m.estimator.EstimateMessages([]api.Message{toSummarize[i]})
		if summarizeTokens+t > maxSummarizeTokens {
			startIdx = i + 1
			break
		}
		summarizeTokens += t
	}
	toSummarize = toSummarize[startIdx:]

	if len(toSummarize) == 0 {
		return nil
	}

	// Build the summarization request
	summaryMsgs := []api.Message{
		{Role: "system", Content: summarizationPrompt},
	}

	// Include existing summary if present
	if m.summary != "" {
		summaryMsgs = append(summaryMsgs, api.Message{
			Role:    "user",
			Content: fmt.Sprintf("Previous summary:\n%s", m.summary),
		})
	}

	// Add the messages to summarize as a user message
	var msgText string
	for _, msg := range toSummarize {
		msgText += fmt.Sprintf("[%s", msg.Role)
		if msg.Name != "" {
			msgText += fmt.Sprintf("/%s", msg.Name)
		}
		msgText += fmt.Sprintf("] %s\n", msg.Content)
	}

	summaryMsgs = append(summaryMsgs, api.Message{
		Role:    "user",
		Content: fmt.Sprintf("Conversation to summarize:\n%s", msgText),
	})

	req := &api.ChatCompletionRequest{
		Messages: summaryMsgs,
		Stream:   false,
	}

	resp, err := complete(ctx, req)
	if err != nil {
		return fmt.Errorf("summarization failed: %w", err)
	}

	if len(resp.Choices) == 0 {
		return fmt.Errorf("empty summarization response")
	}

	m.summary = resp.Choices[0].Message.Content

	// Remove the summarized messages from history
	m.history = m.history[cutoff:]

	return nil
}
