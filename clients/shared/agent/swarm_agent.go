package agent

import (
	"context"
	"fmt"
	"strings"

	"github.com/ThatCatDev/tanrenai/shared/pkg/api"
	"github.com/ThatCatDev/tanrenai/shared/tools"
)

const (
	defaultMaxDepth      = 2
	swarmMaxResultLen    = 4000 // chars, ~1000 tokens — more than the 800 used by plan agent
	defaultWorkerMaxIter = 20
)

const swarmPlanningPrompt = `You are a planning assistant. Break the request into independent, file-level tasks for separate agents.

Output ONLY a numbered list. Format:
1. <verb> <file_path> — <what to do>
2. <verb> <file_path> — <what to do>

Rules:
- Each task should focus on one file or a small group of closely related files
- Order tasks so dependencies come first (e.g. types/interfaces before implementations)
- Use 3-8 tasks
- Workers can read files created by earlier workers using file_read
- Start each with an action verb (Create, Update, Add, Configure, Test)
- Output NOTHING except the numbered list`

const swarmStepPreamble = `You are worker %d of %d in a multi-agent team.

Original request: %s

Your task: "%s"

Prior workers completed:
%s
Complete your task using tools. The previous workers' files already exist on disk — use file_read if you need to see them. When done, summarize what you created in 1-2 sentences.`

const swarmVerifyPrompt = `You are the final verification agent. A team of workers just completed a multi-step task.

Original request: %s

Worker results:
%s

Your job: build the project, run tests, and fix any issues. Use your tools to verify everything works correctly.`

// SwarmConfig configures the recursive swarm orchestrator.
type SwarmConfig struct {
	StreamingConfig
	WorkerTools     *tools.Registry                  // tools for workers (nil = same as main)
	MaxDepth        int                              // max recursion depth (0 = default 2)
	OnPlanGenerated func(depth int, plan *Plan)       // fired when a plan is generated at any depth
	OnWorkerStart   func(depth, stepIdx int, step *PlanStep)
	OnWorkerDone    func(depth, stepIdx int, step *PlanStep)
	OnVerifyStart   func()
}

// RunSwarm executes a recursive multi-agent swarm. An orchestrator plans the
// task, then sequential workers each get a fresh context. Workers can
// recursively spawn sub-swarms up to MaxDepth.
func RunSwarm(ctx context.Context, complete StreamingCompletionFunc,
	messages []api.Message, cfg SwarmConfig) ([]api.Message, error) {
	if cfg.MaxDepth <= 0 {
		cfg.MaxDepth = defaultMaxDepth
	}
	return runSwarmAtDepth(ctx, complete, messages, cfg, 0)
}

func runSwarmAtDepth(ctx context.Context, complete StreamingCompletionFunc,
	messages []api.Message, cfg SwarmConfig, depth int) ([]api.Message, error) {

	// At max depth, degrade to a single continuous agent loop.
	if depth >= cfg.MaxDepth {
		debugf("swarm depth=%d: at max depth, using RunStreaming", depth)
		return RunStreaming(ctx, complete, messages, cfg.StreamingConfig)
	}

	// Extract the original user request.
	userRequest := ""
	for i := len(messages) - 1; i >= 0; i-- {
		if messages[i].Role == "user" {
			userRequest = messages[i].Content
			break
		}
	}
	if userRequest == "" {
		return RunStreaming(ctx, complete, messages, cfg.StreamingConfig)
	}

	// Extract system messages from the conversation.
	var systemMsgs []api.Message
	for _, m := range messages {
		if m.Role == "system" {
			systemMsgs = append(systemMsgs, m)
		}
	}

	// ── Phase 1: Plan ──────────────────────────────────────────────
	plan, err := generateSwarmPlan(ctx, complete, messages, userRequest)
	if err != nil {
		debugf("swarm depth=%d: plan failed (%v), using RunStreaming", depth, err)
		return RunStreaming(ctx, complete, messages, cfg.StreamingConfig)
	}

	debugf("swarm depth=%d: generated %d steps", depth, len(plan.Steps))

	// Single step = degrade to direct execution.
	if len(plan.Steps) <= 1 {
		debugf("swarm depth=%d: single step, using RunStreaming", depth)
		return RunStreaming(ctx, complete, messages, cfg.StreamingConfig)
	}

	if cfg.OnPlanGenerated != nil {
		cfg.OnPlanGenerated(depth, plan)
	}

	// ── Phase 2: Execute workers ───────────────────────────────────
	for i := range plan.Steps {
		if ctx.Err() != nil {
			break
		}

		step := &plan.Steps[i]
		step.Status = StepRunning
		if cfg.OnWorkerStart != nil {
			cfg.OnWorkerStart(depth, i, step)
		}

		result, stepErr := executeSwarmWorker(ctx, complete, systemMsgs, plan, i, userRequest, cfg, depth)
		if stepErr != nil {
			step.Status = StepFailed
			step.Error = stepErr.Error()
			if len(step.Error) > swarmMaxResultLen {
				step.Error = step.Error[:swarmMaxResultLen]
			}
		} else {
			step.Status = StepDone
			step.Result = extractSwarmResult(result)
		}

		if cfg.OnWorkerDone != nil {
			cfg.OnWorkerDone(depth, i, step)
		}
	}

	// ── Phase 3: Verify (only at depth 0) ──────────────────────────
	if depth == 0 {
		if cfg.OnVerifyStart != nil {
			cfg.OnVerifyStart()
		}

		verifyResult, verifyErr := runVerifier(ctx, complete, systemMsgs, plan, userRequest, cfg)
		if verifyErr != nil {
			debugf("swarm: verify failed: %v", verifyErr)
			summary := buildFallbackSummary(plan)
			messages = append(messages, api.Message{Role: "assistant", Content: summary})
			return messages, nil
		}

		messages = append(messages, api.Message{Role: "assistant", Content: verifyResult})
		return messages, nil
	}

	// At depth > 0, just return messages with a summary appended.
	summary := buildFallbackSummary(plan)
	messages = append(messages, api.Message{Role: "assistant", Content: summary})
	return messages, nil
}

// generateSwarmPlan calls the LLM with the swarm planning prompt.
func generateSwarmPlan(ctx context.Context, complete StreamingCompletionFunc,
	messages []api.Message, userRequest string) (*Plan, error) {

	// Collect system messages from the conversation.
	systemParts := []string{swarmPlanningPrompt}
	for _, m := range messages {
		if m.Role == "system" && m.Content != "" {
			systemParts = append(systemParts, m.Content)
		}
	}

	planMsgs := []api.Message{
		{Role: "system", Content: strings.Join(systemParts, "\n\n")},
		{Role: "user", Content: "Break this request into file-level tasks:\n\n" + userRequest},
	}

	req := &api.ChatCompletionRequest{
		Messages:       planMsgs,
		Stream:         true,
		EnableThinking: false, // we need the list in content, not reasoning
	}

	events, err := complete(ctx, req)
	if err != nil {
		return nil, fmt.Errorf("swarm plan completion failed: %w", err)
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

	text := content.String()
	if text == "" && reasoning.Len() > 0 {
		text = reasoning.String()
	}

	if strings.TrimSpace(text) == "" {
		return nil, fmt.Errorf("model returned empty plan")
	}

	plan := parsePlan(text)
	if plan == nil {
		return nil, fmt.Errorf("model did not produce numbered steps")
	}

	return plan, nil
}

// executeSwarmWorker runs a single worker with a fresh context. The worker
// recursively calls runSwarmAtDepth, so it may spawn sub-workers if the
// task is complex enough.
func executeSwarmWorker(ctx context.Context, complete StreamingCompletionFunc,
	systemMsgs []api.Message, plan *Plan, stepIdx int, userRequest string,
	cfg SwarmConfig, depth int) ([]api.Message, error) {

	step := &plan.Steps[stepIdx]
	total := len(plan.Steps)
	summaries := formatStepSummaries(plan.Steps[:stepIdx])

	preamble := fmt.Sprintf(swarmStepPreamble, step.Index, total, userRequest, step.Description, summaries)

	// Build fresh context for this worker.
	var systemParts []string
	for _, m := range systemMsgs {
		if m.Content != "" {
			systemParts = append(systemParts, m.Content)
		}
	}
	systemParts = append(systemParts, preamble)

	workerMsgs := []api.Message{
		{Role: "system", Content: strings.Join(systemParts, "\n\n")},
		{Role: "user", Content: step.Description},
	}

	// Build worker config with optional tool subset.
	workerCfg := cfg
	if cfg.WorkerTools != nil {
		workerCfg.StreamingConfig.Tools = cfg.WorkerTools
	}
	if cfg.MaxDepth > 0 && defaultWorkerMaxIter > 0 {
		workerCfg.StreamingConfig.MaxIterations = defaultWorkerMaxIter
	}

	// Recurse — if the sub-task is simple, this degrades to RunStreaming.
	return runSwarmAtDepth(ctx, complete, workerMsgs, workerCfg, depth+1)
}

// runVerifier runs a final agent with full tool access to build, test, and fix.
func runVerifier(ctx context.Context, complete StreamingCompletionFunc,
	systemMsgs []api.Message, plan *Plan, userRequest string, cfg SwarmConfig) (string, error) {

	summaries := formatStepSummaries(plan.Steps)
	verifyPrompt := fmt.Sprintf(swarmVerifyPrompt, userRequest, summaries)

	var systemParts []string
	for _, m := range systemMsgs {
		if m.Content != "" {
			systemParts = append(systemParts, m.Content)
		}
	}
	systemParts = append(systemParts, verifyPrompt)

	verifyMsgs := []api.Message{
		{Role: "system", Content: strings.Join(systemParts, "\n\n")},
		{Role: "user", Content: "Verify the project builds and tests pass. Fix any issues."},
	}

	// Verifier gets the full tool set and unlimited iterations.
	result, err := RunStreaming(ctx, complete, verifyMsgs, cfg.StreamingConfig)
	if err != nil {
		return "", err
	}

	// Extract the final assistant message.
	for i := len(result) - 1; i >= 0; i-- {
		if result[i].Role == "assistant" && result[i].Content != "" {
			return result[i].Content, nil
		}
	}

	return buildFallbackSummary(plan), nil
}

// extractSwarmResult gets the last assistant content, truncated for swarm use.
func extractSwarmResult(messages []api.Message) string {
	for i := len(messages) - 1; i >= 0; i-- {
		if messages[i].Role == "assistant" && messages[i].Content != "" {
			result := messages[i].Content
			if len(result) > swarmMaxResultLen {
				result = result[:swarmMaxResultLen] + "..."
			}
			return result
		}
	}
	return "(no output)"
}
