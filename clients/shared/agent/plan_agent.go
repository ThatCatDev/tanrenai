package agent

import (
	"context"
	"fmt"
	"strings"

	"github.com/ThatCatDev/tanrenai/shared/pkg/api"
)

// System prompts for the plan-execute architecture.
const (
	planningSystemPrompt = `You are a planning assistant. Output ONLY a numbered list of implementation steps. No explanation, no code, no preamble.

Format each step as:
1. <verb> <what>
2. <verb> <what>

Rules:
- Each step = one concrete action (read a file, create a component, run a command)
- Use 3-8 steps
- Start each step with an action verb (Create, Read, Write, Add, Configure, Build, Test, etc.)
- Output NOTHING except the numbered list — no thinking, no explanation, no summary`

	stepPreambleTemplate = `You are executing step %d of %d: "%s"
Completed so far:
%s
Focus only on this step. Use your tools to complete it — do not just describe what you would do. When done, summarize what you accomplished in 1-2 sentences.`

	synthesisSystemPrompt = `You completed a multi-step task. Below are the results of each step.
Summarize what was accomplished overall. Note any failures or skipped steps.`
)

const (
	maxResultLen = 800 // chars, ~200 tokens
)

// PlanAgentConfig configures the plan-execute orchestrator.
type PlanAgentConfig struct {
	StreamingConfig               // embeds existing config (tools, hooks, etc.)
	UserInput       <-chan string // non-blocking read for mid-turn injection
	OnPlanningStart func()
	OnPlanGenerated func(plan *Plan)
	OnStepStart     func(stepIdx int, step *PlanStep)
	OnStepDone      func(stepIdx int, step *PlanStep)
	OnReplan        func(reason string, newPlan *Plan)
	OnSynthesize    func()
}

// RunPlannedStreaming runs a single continuous agent loop, keeping the full
// conversation history in context. This delegates directly to RunStreaming —
// the model decides on its own how to decompose and execute the task.
func RunPlannedStreaming(ctx context.Context, complete StreamingCompletionFunc,
	messages []api.Message, cfg PlanAgentConfig) ([]api.Message, error) {
	return RunStreaming(ctx, complete, messages, cfg.StreamingConfig)
}

// generatePlan calls the LLM with no tools to produce a numbered plan.
func generatePlan(ctx context.Context, complete StreamingCompletionFunc,
	messages []api.Message, userRequest string, cfg *PlanAgentConfig) (*Plan, error) {
	// Merge planning prompt + original system messages into one
	systemParts := []string{planningSystemPrompt}
	for _, m := range messages {
		if m.Role == "system" && m.Content != "" {
			systemParts = append(systemParts, m.Content)
		}
	}
	planMsgs := []api.Message{
		{Role: "system", Content: strings.Join(systemParts, "\n\n")},
		{Role: "user", Content: "Break this request into numbered steps:\n\n" + userRequest},
	}

	req := &api.ChatCompletionRequest{
		Messages: planMsgs,
		Stream:   true,
		// Disable thinking for plan generation — we need the numbered list
		// in content, not buried in reasoning_content.
		EnableThinking: false,
	}

	events, err := complete(ctx, req)
	if err != nil {
		return nil, fmt.Errorf("plan completion failed: %w", err)
	}

	var content, reasoning strings.Builder
	for ev := range events {
		if ev.Err != nil {
			return nil, ev.Err
		}
		if ev.Done {
			break
		}
		if ev.Chunk == nil {
			continue
		}
		for _, choice := range ev.Chunk.Choices {
			if choice.Delta.Content != "" {
				content.WriteString(choice.Delta.Content)
			}
			if choice.Delta.ReasoningContent != "" {
				reasoning.WriteString(choice.Delta.ReasoningContent)
			}
		}
	}

	// Some models put the plan in reasoning_content instead of content
	text := content.String()
	if text == "" && reasoning.Len() > 0 {
		debugf("plan: no content, using reasoning_content (%d chars)", reasoning.Len())
		text = reasoning.String()
	}

	if strings.TrimSpace(text) == "" {
		return nil, fmt.Errorf("model returned empty plan")
	}

	debugf("plan raw text (%d chars): %s", len(text), text)

	plan := parsePlan(text)
	if plan == nil {
		return nil, fmt.Errorf("model did not produce numbered steps")
	}

	return plan, nil
}

// executeStep runs a single plan step as an isolated sub-agent.
func executeStep(ctx context.Context, complete StreamingCompletionFunc,
	systemMsgs []api.Message, plan *Plan, stepIdx int, cfg *PlanAgentConfig) ([]api.Message, error) {
	step := &plan.Steps[stepIdx]
	total := len(plan.Steps)
	summaries := formatStepSummaries(plan.Steps[:stepIdx])

	preamble := fmt.Sprintf(stepPreambleTemplate, step.Index, total, step.Description, summaries)

	// Build focused context: merge all system content into one message
	var systemParts []string
	for _, m := range systemMsgs {
		if m.Content != "" {
			systemParts = append(systemParts, m.Content)
		}
	}
	systemParts = append(systemParts, preamble)

	var stepMsgs []api.Message
	stepMsgs = append(stepMsgs, api.Message{Role: "system", Content: strings.Join(systemParts, "\n\n")})
	stepMsgs = append(stepMsgs, api.Message{Role: "user", Content: step.Description})

	return RunStreaming(ctx, complete, stepMsgs, cfg.StreamingConfig)
}

// synthesize produces a final summary from all step results.
func synthesize(ctx context.Context, complete StreamingCompletionFunc,
	systemMsgs []api.Message, plan *Plan, userRequest string, cfg *PlanAgentConfig) (string, error) {
	summaryBlock := formatStepSummaries(plan.Steps)

	// Merge all system content into one message
	var synthParts []string
	for _, m := range systemMsgs {
		if m.Content != "" {
			synthParts = append(synthParts, m.Content)
		}
	}
	synthParts = append(synthParts, synthesisSystemPrompt)

	synthMsgs := []api.Message{
		{Role: "system", Content: strings.Join(synthParts, "\n\n")},
		{Role: "user", Content: fmt.Sprintf("Original request: %s\n\nStep results:\n%s", userRequest, summaryBlock)},
	}

	req := &api.ChatCompletionRequest{
		Messages:       synthMsgs,
		Stream:         true,
		EnableThinking: cfg.EnableThinking,
		// No tools, no token cap — synthesis only
	}

	events, err := complete(ctx, req)
	if err != nil {
		return "", err
	}

	// Accumulate and fire content deltas
	var content strings.Builder
	for ev := range events {
		if ev.Err != nil {
			return "", ev.Err
		}
		if ev.Done {
			break
		}
		if ev.Chunk == nil {
			continue
		}
		for _, choice := range ev.Chunk.Choices {
			if choice.Delta.Content != "" {
				content.WriteString(choice.Delta.Content)
				if cfg.OnContentDelta != nil {
					cfg.OnContentDelta(choice.Delta.Content)
				}
			}
		}
	}

	return content.String(), nil
}

// extractStepResult gets the last assistant message content, truncated.
func extractStepResult(messages []api.Message) string {
	for i := len(messages) - 1; i >= 0; i-- {
		if messages[i].Role == "assistant" && messages[i].Content != "" {
			result := messages[i].Content
			if len(result) > maxResultLen {
				result = result[:maxResultLen] + "..."
			}

			return result
		}
	}

	return "(no output)"
}

// readUserInput does a non-blocking read from the user injection channel.
func readUserInput(ch <-chan string) string {
	if ch == nil {
		return ""
	}
	select {
	case msg := <-ch:
		return msg
	default:
		return ""
	}
}

// handleInjection processes user input mid-turn. Returns the (possibly new) plan
// and the step index to continue from. Returns -1 to stop.
func handleInjection(ctx context.Context, complete StreamingCompletionFunc,
	messages []api.Message, plan *Plan, currentIdx int, input string,
	userRequest string, cfg *PlanAgentConfig) (*Plan, int) {
	lower := strings.TrimSpace(strings.ToLower(input))

	switch lower {
	case "/stop":
		// Mark remaining steps as skipped
		for i := currentIdx; i < len(plan.Steps); i++ {
			plan.Steps[i].Status = StepSkipped
		}

		return plan, -1

	case "/skip":
		plan.Steps[currentIdx].Status = StepSkipped
		if currentIdx+1 < len(plan.Steps) {
			return plan, currentIdx + 1
		}

		return plan, -1

	case "/redo":
		// Re-run the previous step
		if currentIdx > 0 {
			prev := currentIdx - 1
			plan.Steps[prev].Status = StepPending
			plan.Steps[prev].Result = ""
			plan.Steps[prev].Error = ""

			return plan, prev
		}

		return plan, currentIdx

	default:
		// Re-plan: generate new plan with user guidance
		replanRequest := fmt.Sprintf("Original request: %s\n\nCompleted steps:\n%s\nUser guidance: %s\n\nRe-plan the remaining work.",
			userRequest, formatStepSummaries(plan.Steps), input)

		replanMsgs := []api.Message{
			{Role: "system", Content: planningSystemPrompt},
			{Role: "user", Content: replanRequest},
		}

		req := &api.ChatCompletionRequest{
			Messages:       replanMsgs,
			Stream:         true,
			EnableThinking: cfg.EnableThinking,
		}

		events, err := complete(ctx, req)
		if err != nil {
			debugf("replan failed: %v", err)

			return plan, currentIdx
		}

		var content strings.Builder
		for ev := range events {
			if ev.Err != nil {
				debugf("replan stream error: %v", ev.Err)

				return plan, currentIdx
			}
			if ev.Done {
				break
			}
			if ev.Chunk == nil {
				continue
			}
			for _, choice := range ev.Chunk.Choices {
				if choice.Delta.Content != "" {
					content.WriteString(choice.Delta.Content)
				}
			}
		}

		newPlan := parsePlan(content.String())
		if newPlan == nil {
			// Replan failed, continue with current plan
			return plan, currentIdx
		}
		// Preserve completed steps
		var merged []PlanStep
		for _, s := range plan.Steps {
			if s.Status == StepDone || s.Status == StepFailed {
				merged = append(merged, s)
			}
		}
		offset := len(merged)
		for i, s := range newPlan.Steps {
			s.Index = offset + i + 1
			merged = append(merged, s)
		}
		newPlan.Steps = merged

		if cfg.OnReplan != nil {
			cfg.OnReplan(input, newPlan)
		}

		return newPlan, offset
	}
}

// buildFallbackSummary creates a text summary when synthesis LLM call fails.
func buildFallbackSummary(plan *Plan) string {
	var b strings.Builder
	b.WriteString("Task completed with the following results:\n\n")
	for _, s := range plan.Steps {
		status := s.Status.String()
		result := s.Result
		if s.Status == StepFailed {
			result = s.Error
		}
		if result == "" {
			result = "(no output)"
		}
		fmt.Fprintf(&b, "%d. [%s] %s\n   %s\n", s.Index, status, s.Description, result)
	}

	return b.String()
}
