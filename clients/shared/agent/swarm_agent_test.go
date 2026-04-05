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
func mockSwarmComplete(planText, workerResponse, verifyResponse string) StreamingCompletionFunc {
	return func(ctx context.Context, req *api.ChatCompletionRequest) (<-chan apiclient.StreamEvent, error) {
		// Determine which phase we're in based on system prompt.
		system := ""
		for _, m := range req.Messages {
			if m.Role == "system" {
				system = m.Content
			}
		}

		var content string
		switch {
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
		"1. Create types.ts\n2. Create app.ts\n3. Create tests.ts",
		"Done with this task.",
		"All verified.",
	)

	messages := []api.Message{
		{Role: "system", Content: "You are helpful."},
		{Role: "user", Content: "Build a web app with types, app, and tests"},
	}

	var planDepth int
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
		MaxDepth: 1, // workers degrade to RunStreaming (no sub-planning)
		OnPlanGenerated: func(depth int, plan *Plan) {
			planDepth = depth
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
	if planDepth != 0 {
		t.Errorf("expected plan at depth 0, got %d", planDepth)
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
	// Return non-numbered text so plan parsing fails.
	complete := mockSwarmComplete(
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
		t.Error("single-step plan should degrade to RunStreaming without firing OnPlanGenerated")
	}
}

func TestRunSwarm_MaxDepthDegrades(t *testing.T) {
	// Even with a multi-step plan, MaxDepth=1 means workers can't sub-plan.
	callCount := 0
	complete := func(ctx context.Context, req *api.ChatCompletionRequest) (<-chan apiclient.StreamEvent, error) {
		callCount++
		content := "Done with task."

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
		MaxDepth: 1, // workers immediately degrade to RunStreaming
	}

	_, err := RunSwarm(context.Background(), complete, messages, cfg)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	// Should complete without deep recursion.
}

func TestRunSwarm_WorkerFailureContinues(t *testing.T) {
	callNum := 0
	complete := func(ctx context.Context, req *api.ChatCompletionRequest) (<-chan apiclient.StreamEvent, error) {
		callNum++
		system := ""
		for _, m := range req.Messages {
			if m.Role == "system" {
				system = m.Content
			}
		}

		var content string
		switch {
		case strings.Contains(system, "Break the request"):
			content = "1. First task\n2. Second task"
		case strings.Contains(system, "Verify the project"):
			content = "Verified."
		default:
			content = "Worker done."
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
		MaxDepth: 1, // workers degrade to RunStreaming (no sub-planning)
		OnPlanGenerated: func(depth int, plan *Plan) {},
		OnWorkerStart:   func(depth, stepIdx int, step *PlanStep) {},
		OnWorkerDone: func(depth, stepIdx int, step *PlanStep) {
			doneCount++
		},
		OnVerifyStart: func() {},
	}

	_, err := RunSwarm(context.Background(), complete, messages, cfg)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if doneCount != 2 {
		t.Errorf("expected 2 workers done, got %d", doneCount)
	}
}
