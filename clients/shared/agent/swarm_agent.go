package agent

import (
	"context"
	"fmt"
	"strings"

	"github.com/ThatCatDev/tanrenai/shared/pkg/api"
	"github.com/ThatCatDev/tanrenai/shared/tools"
)

const (
	defaultMaxDepth   = 2
	swarmMaxResultLen = 4000 // chars, ~1000 tokens
)

const swarmArchitectPrompt = `You are a software architect. Given the user's request, define the technical decisions that all developers on the team must follow.

Output a short spec (under 300 words) covering:
- Framework/library choices (e.g. "vanilla TypeScript, no React/Vue")
- Project structure (e.g. "src/components/, src/services/, src/types/")
- Shared interfaces/types that multiple files will use
- Naming conventions
- How components connect (e.g. "main.ts imports and registers all components")
- Build tool config (e.g. "Vite with TypeScript")

Be specific and decisive. Do NOT hedge or offer alternatives.
Output NOTHING except the spec.`

const swarmPlanningPrompt = `You are a planning assistant. Break the request into independent, file-level tasks for separate agents.

Output ONLY a numbered list. Format:
1. <verb> <file_path> — <what to do>
2. <verb> <file_path> — <what to do>

Rules:
- Each task should focus on one file or a small group of closely related files
- Order tasks so dependencies come first (e.g. types/interfaces before implementations)
- Workers can read files created by earlier workers using file_read
- Start each with an action verb (Create, Update, Add, Configure, Test)
- Output NOTHING except the numbered list`

const swarmStepPreamble = `You are worker %d of %d in a multi-agent team.

Original request: %s

Architecture spec (all workers follow this):
%s

Full plan (other workers handle the other tasks — do NOT do their work):
%s

Your task: "%s"

Prior workers completed:
%s
Complete ONLY your task using tools. Follow the architecture spec exactly. Do not create files assigned to other workers. The previous workers' files already exist on disk — use file_read if you need to see them. When done, summarize what you created in 1-2 sentences.`

const swarmVerifyPrompt = `You are the final verification agent. A team of workers just completed a multi-step task.

Original request: %s

Architecture spec:
%s

Worker results:
%s

Your job: build the project, run tests, and fix any issues. Use your tools to verify everything works correctly.`

// SwarmConfig configures the recursive swarm orchestrator.
type SwarmConfig struct {
	StreamingConfig
	WorkerTools     *tools.Registry                       // tools for workers (nil = same as main)
	MaxDepth        int                                   // max recursion depth (0 = default 2)
	OnArchitectSpec func(depth int, spec string)           // fired when architecture spec is generated
	OnPlanGenerated func(depth int, plan *Plan)            // fired when a plan is generated at any depth
	OnWorkerStart   func(depth, stepIdx int, step *PlanStep)
	OnWorkerDone    func(depth, stepIdx int, step *PlanStep)
	OnVerifyStart   func()
}

// RunSwarm executes a recursive multi-agent swarm. An orchestrator generates
// an architecture spec, plans file-level tasks, then sequential workers each
// get a fresh context with the spec and plan. Complex workers can recursively
// spawn sub-swarms up to MaxDepth.
func RunSwarm(ctx context.Context, complete StreamingCompletionFunc,
	messages []api.Message, cfg SwarmConfig) ([]api.Message, error) {
	if cfg.MaxDepth <= 0 {
		cfg.MaxDepth = defaultMaxDepth
	}
	return runSwarmAtDepth(ctx, complete, messages, cfg, 0, "")
}

func runSwarmAtDepth(ctx context.Context, complete StreamingCompletionFunc,
	messages []api.Message, cfg SwarmConfig, depth int, parentSpec string) ([]api.Message, error) {

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

	// ── Phase 0: Architecture spec (only at depth 0) ───────────────
	archSpec := parentSpec
	if depth == 0 {
		spec, err := generateArchitectureSpec(ctx, complete, messages, userRequest)
		if err != nil {
			debugf("swarm: architecture spec failed (%v), continuing without", err)
			archSpec = "(no architecture spec available)"
		} else {
			archSpec = spec
			debugf("swarm: architecture spec generated (%d chars)", len(spec))
		}
		if cfg.OnArchitectSpec != nil {
			cfg.OnArchitectSpec(depth, archSpec)
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

		result, stepErr := executeSwarmWorker(ctx, complete, systemMsgs, plan, i, userRequest, archSpec, cfg, depth)
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

		verifyResult, verifyErr := runVerifier(ctx, complete, systemMsgs, plan, userRequest, archSpec, cfg)
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

// generateArchitectureSpec calls the LLM to produce a shared technical spec.
func generateArchitectureSpec(ctx context.Context, complete StreamingCompletionFunc,
	messages []api.Message, userRequest string) (string, error) {

	var systemParts []string
	systemParts = append(systemParts, swarmArchitectPrompt)
	for _, m := range messages {
		if m.Role == "system" && m.Content != "" {
			systemParts = append(systemParts, m.Content)
		}
	}

	specMsgs := []api.Message{
		{Role: "system", Content: strings.Join(systemParts, "\n\n")},
		{Role: "user", Content: "Define the architecture for this project:\n\n" + userRequest},
	}

	req := &api.ChatCompletionRequest{
		Messages:       specMsgs,
		Stream:         true,
		EnableThinking: false,
	}

	events, err := complete(ctx, req)
	if err != nil {
		return "", fmt.Errorf("architecture spec failed: %w", err)
	}

	var content, reasoning strings.Builder
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
		return "", fmt.Errorf("model returned empty architecture spec")
	}

	return text, nil
}

// generateSwarmPlan calls the LLM with the swarm planning prompt.
func generateSwarmPlan(ctx context.Context, complete StreamingCompletionFunc,
	messages []api.Message, userRequest string) (*Plan, error) {

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
		EnableThinking: false,
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

// isComplexTask returns true if the task description suggests it needs
// sub-planning rather than direct execution.
func isComplexTask(desc string) bool {
	if len(desc) > 120 {
		return true
	}
	lower := strings.ToLower(desc)
	actionWords := []string{"create", "implement", "add", "build", "write", "configure", "test", "set up"}
	count := 0
	for _, w := range actionWords {
		if strings.Contains(lower, w) {
			count++
		}
	}
	return count >= 3
}

// executeSwarmWorker runs a single worker with a fresh context. Simple tasks
// run directly via RunStreaming. Complex tasks can recursively sub-plan if
// depth allows.
func executeSwarmWorker(ctx context.Context, complete StreamingCompletionFunc,
	systemMsgs []api.Message, plan *Plan, stepIdx int, userRequest, archSpec string,
	cfg SwarmConfig, depth int) ([]api.Message, error) {

	step := &plan.Steps[stepIdx]
	total := len(plan.Steps)
	summaries := formatStepSummaries(plan.Steps[:stepIdx])
	fullPlan := formatPlanList(plan)

	preamble := fmt.Sprintf(swarmStepPreamble,
		step.Index, total, userRequest, archSpec, fullPlan, step.Description, summaries)

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

	// Only sub-plan if the task is complex AND depth allows.
	if isComplexTask(step.Description) && depth+1 < cfg.MaxDepth {
		debugf("swarm depth=%d: task %d is complex, sub-planning", depth, step.Index)
		return runSwarmAtDepth(ctx, complete, workerMsgs, workerCfg, depth+1, archSpec)
	}

	// Simple tasks run directly.
	return RunStreaming(ctx, complete, workerMsgs, workerCfg.StreamingConfig)
}

// runVerifier runs a final agent with full tool access to build, test, and fix.
func runVerifier(ctx context.Context, complete StreamingCompletionFunc,
	systemMsgs []api.Message, plan *Plan, userRequest, archSpec string, cfg SwarmConfig) (string, error) {

	summaries := formatStepSummaries(plan.Steps)
	prompt := fmt.Sprintf(swarmVerifyPrompt, userRequest, archSpec, summaries)

	var systemParts []string
	for _, m := range systemMsgs {
		if m.Content != "" {
			systemParts = append(systemParts, m.Content)
		}
	}
	systemParts = append(systemParts, prompt)

	verifyMsgs := []api.Message{
		{Role: "system", Content: strings.Join(systemParts, "\n\n")},
		{Role: "user", Content: "Verify the project builds and tests pass. Fix any issues."},
	}

	result, err := RunStreaming(ctx, complete, verifyMsgs, cfg.StreamingConfig)
	if err != nil {
		return "", err
	}

	for i := len(result) - 1; i >= 0; i-- {
		if result[i].Role == "assistant" && result[i].Content != "" {
			return result[i].Content, nil
		}
	}

	return buildFallbackSummary(plan), nil
}

// formatPlanList formats the plan as a numbered list for worker context.
func formatPlanList(plan *Plan) string {
	var b strings.Builder
	for _, s := range plan.Steps {
		fmt.Fprintf(&b, "%d. %s\n", s.Index, s.Description)
	}
	return b.String()
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
