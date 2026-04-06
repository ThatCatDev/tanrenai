package agent

import (
	"context"
	"strings"
	"testing"

	"github.com/ThatCatDev/tanrenai/shared/apiclient"
	"github.com/ThatCatDev/tanrenai/shared/pkg/api"
	"github.com/ThatCatDev/tanrenai/shared/tools"
)

// mockSwarmComplete returns different responses based on the system prompt content.
func mockSwarmComplete(archSpec, planText, workerResponse, verifyResponse string) StreamingCompletionFunc {
	return func(ctx context.Context, req *api.ChatCompletionRequest) (<-chan apiclient.StreamEvent, error) {
		system := ""
		for _, m := range req.Messages {
			if m.Role == "system" {
				system = m.Content
			}
		}

		var content string
		switch {
		case strings.Contains(system, "software architect"):
			content = archSpec
		case strings.Contains(system, "Break the request"):
			content = planText
		case strings.Contains(system, "Verify the project"):
			content = verifyResponse
		default:
			content = workerResponse
		}

		ch := make(chan apiclient.StreamEvent, 2)
		go func() {
			defer close(ch)
			fr := "stop"
			ch <- apiclient.StreamEvent{
				Chunk: &api.ChatCompletionChunk{
					Choices: []api.ChunkChoice{{
						Delta:        api.MessageDelta{Role: "assistant", Content: content},
						FinishReason: &fr,
					}},
				},
			}
			ch <- apiclient.StreamEvent{Done: true}
		}()
		return ch, nil
	}
}

func TestRunSwarm_PlanAndExecute(t *testing.T) {
	complete := mockSwarmComplete(
		"Use vanilla TypeScript with Vite.",
		"1. Create types.ts\n2. Create app.ts\n3. Create tests.ts",
		"Done with this task.",
		"All verified.",
	)

	messages := []api.Message{
		{Role: "system", Content: "You are helpful."},
		{Role: "user", Content: "Build a web app with types, app, and tests"},
	}

	var archSpec string
	var planSteps int
	var workersStarted, workersDone int
	var verifyStarted bool

	cfg := SwarmConfig{
		StreamingConfig: StreamingConfig{
			Config: Config{
				MaxIterations: 5,
				Tools:         tools.NewRegistry(),
			},
		},
		MaxDepth: 1,
		OnArchitectSpec: func(depth int, spec string) {
			archSpec = spec
		},
		OnPlanGenerated: func(depth int, plan *Plan) {
			planSteps = len(plan.Steps)
		},
		OnWorkerStart: func(depth, stepIdx int, step *PlanStep) {
			workersStarted++
		},
		OnWorkerDone: func(depth, stepIdx int, step *PlanStep) {
			workersDone++
		},
		OnVerifyStart: func() {
			verifyStarted = true
		},
	}

	result, err := RunSwarm(context.Background(), complete, messages, cfg)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if archSpec == "" {
		t.Error("expected architecture spec to be generated")
	}
	if planSteps != 3 {
		t.Errorf("expected 3 plan steps, got %d", planSteps)
	}
	if workersStarted != 3 {
		t.Errorf("expected 3 workers started, got %d", workersStarted)
	}
	if workersDone != 3 {
		t.Errorf("expected 3 workers done, got %d", workersDone)
	}
	if !verifyStarted {
		t.Error("expected verify phase to start")
	}
	if len(result) == 0 {
		t.Error("expected non-empty result")
	}
}

func TestRunSwarm_PlanFailsFallback(t *testing.T) {
	complete := mockSwarmComplete(
		"Use TypeScript.",
		"I'll just do everything at once.",
		"Done.",
		"",
	)

	messages := []api.Message{
		{Role: "user", Content: "Build something complex"},
	}

	cfg := SwarmConfig{
		StreamingConfig: StreamingConfig{
			Config: Config{
				MaxIterations: 5,
				Tools:         tools.NewRegistry(),
			},
		},
	}

	var planGenerated bool
	cfg.OnPlanGenerated = func(depth int, plan *Plan) {
		planGenerated = true
	}

	_, err := RunSwarm(context.Background(), complete, messages, cfg)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if planGenerated {
		t.Error("plan should not have been generated (fallback to RunStreaming)")
	}
}

func TestRunSwarm_SingleStepFallback(t *testing.T) {
	complete := mockSwarmComplete(
		"Use TypeScript.",
		"1. Do everything",
		"Done.",
		"",
	)

	messages := []api.Message{
		{Role: "user", Content: "Simple task"},
	}

	cfg := SwarmConfig{
		StreamingConfig: StreamingConfig{
			Config: Config{
				MaxIterations: 5,
				Tools:         tools.NewRegistry(),
			},
		},
	}

	var planGenerated bool
	cfg.OnPlanGenerated = func(depth int, plan *Plan) {
		planGenerated = true
	}

	_, err := RunSwarm(context.Background(), complete, messages, cfg)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if planGenerated {
		t.Error("single-step plan should degrade to RunStreaming")
	}
}

func TestRunSwarm_MaxDepthDegrades(t *testing.T) {
	callCount := 0
	complete := func(ctx context.Context, req *api.ChatCompletionRequest) (<-chan apiclient.StreamEvent, error) {
		callCount++
		ch := make(chan apiclient.StreamEvent, 2)
		go func() {
			defer close(ch)
			fr := "stop"
			ch <- apiclient.StreamEvent{
				Chunk: &api.ChatCompletionChunk{
					Choices: []api.ChunkChoice{{
						Delta:        api.MessageDelta{Role: "assistant", Content: "Done."},
						FinishReason: &fr,
					}},
				},
			}
			ch <- apiclient.StreamEvent{Done: true}
		}()
		return ch, nil
	}

	messages := []api.Message{
		{Role: "user", Content: "Do complex work"},
	}

	cfg := SwarmConfig{
		StreamingConfig: StreamingConfig{
			Config: Config{
				MaxIterations: 5,
				Tools:         tools.NewRegistry(),
			},
		},
		MaxDepth: 1,
	}

	_, err := RunSwarm(context.Background(), complete, messages, cfg)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestRunSwarm_WorkerFailureContinues(t *testing.T) {
	complete := mockSwarmComplete(
		"Use TypeScript.",
		"1. First task\n2. Second task",
		"Worker done.",
		"Verified.",
	)

	messages := []api.Message{
		{Role: "user", Content: "Build two things"},
	}

	var doneCount int
	cfg := SwarmConfig{
		StreamingConfig: StreamingConfig{
			Config: Config{
				MaxIterations: 5,
				Tools:         tools.NewRegistry(),
			},
		},
		MaxDepth:        1,
		OnPlanGenerated: func(depth int, plan *Plan) {},
		OnWorkerStart:   func(depth, stepIdx int, step *PlanStep) {},
		OnWorkerDone: func(depth, stepIdx int, step *PlanStep) {
			doneCount++
		},
		OnVerifyStart:   func() {},
		OnArchitectSpec: func(depth int, spec string) {},
	}

	_, err := RunSwarm(context.Background(), complete, messages, cfg)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if doneCount != 2 {
		t.Errorf("expected 2 workers done, got %d", doneCount)
	}
}

func TestIsComplexTask(t *testing.T) {
	tests := []struct {
		desc string
		want bool
	}{
		{"Create package.json", false},
		{"Create tsconfig.json", false},
		{"fix the bug", false},
		// Long description = complex
		{"Build the file browser component with navigation controls, breadcrumbs, search functionality, and tree view rendering for the desktop OS application", true},
		// 3+ action words = complex
		{"Create, implement, and test the authentication module", true},
		{"Build the game, add scoring, configure settings, and write tests", true},
		// 2 action words = not complex
		{"Create and test the component", false},
	}
	for _, tt := range tests {
		got := isComplexTask(tt.desc)
		if got != tt.want {
			t.Errorf("isComplexTask(%q) = %v, want %v", tt.desc, got, tt.want)
		}
	}
}

func TestFormatPlanList(t *testing.T) {
	plan := &Plan{
		Steps: []PlanStep{
			{Index: 1, Description: "Create types.ts"},
			{Index: 2, Description: "Create app.ts"},
			{Index: 3, Description: "Write tests"},
		},
	}
	got := formatPlanList(plan)
	want := "1. Create types.ts\n2. Create app.ts\n3. Write tests\n"
	if got != want {
		t.Errorf("formatPlanList() = %q, want %q", got, want)
	}
}

func TestRunSwarm_ArchitectSpecGenerated(t *testing.T) {
	complete := mockSwarmComplete(
		"Framework: Vanilla TypeScript\nBuild: Vite\nStructure: src/components/",
		"1. Create types.ts\n2. Create app.ts",
		"Done.",
		"Verified.",
	)

	messages := []api.Message{
		{Role: "user", Content: "Build a web app"},
	}

	var specReceived string
	cfg := SwarmConfig{
		StreamingConfig: StreamingConfig{
			Config: Config{
				MaxIterations: 5,
				Tools:         tools.NewRegistry(),
			},
		},
		MaxDepth: 1,
		OnArchitectSpec: func(depth int, spec string) {
			specReceived = spec
		},
		OnPlanGenerated: func(depth int, plan *Plan) {},
		OnWorkerStart:   func(depth, stepIdx int, step *PlanStep) {},
		OnWorkerDone:    func(depth, stepIdx int, step *PlanStep) {},
		OnVerifyStart:   func() {},
	}

	_, err := RunSwarm(context.Background(), complete, messages, cfg)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !strings.Contains(specReceived, "Vanilla TypeScript") {
		t.Errorf("expected spec to contain 'Vanilla TypeScript', got %q", specReceived)
	}
}
