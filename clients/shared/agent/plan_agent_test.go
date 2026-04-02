package agent

import (
	"context"
	"testing"

	"github.com/ThatCatDev/tanrenai/shared/apiclient"
	"github.com/ThatCatDev/tanrenai/shared/pkg/api"
	"github.com/ThatCatDev/tanrenai/shared/tools"
)

// ── Plan Parsing Tests ─────────────────────────────────────────────────

func TestParsePlan_NumberedList(t *testing.T) {
	text := `Here's my plan:
1. Read the file main.go
2. Add a new HTTP handler
3. Write unit tests
4. Run the tests`

	plan := parsePlan(text)
	if len(plan.Steps) != 4 {
		t.Fatalf("expected 4 steps, got %d", len(plan.Steps))
	}
	if plan.Steps[0].Description != "Read the file main.go" {
		t.Errorf("step 1: got %q", plan.Steps[0].Description)
	}
	if plan.Steps[2].Description != "Write unit tests" {
		t.Errorf("step 3: got %q", plan.Steps[2].Description)
	}
	for i, s := range plan.Steps {
		if s.Index != i+1 {
			t.Errorf("step %d: index=%d", i, s.Index)
		}
		if s.Status != StepPending {
			t.Errorf("step %d: status=%v", i, s.Status)
		}
	}
}

func TestParsePlan_ParenthesisFormat(t *testing.T) {
	text := `1) First step
2) Second step`
	plan := parsePlan(text)
	if len(plan.Steps) != 2 {
		t.Fatalf("expected 2 steps, got %d", len(plan.Steps))
	}
	if plan.Steps[0].Description != "First step" {
		t.Errorf("step 1: got %q", plan.Steps[0].Description)
	}
}

func TestParsePlan_MessyOutput(t *testing.T) {
	text := `I'll help you with that. Let me think about the approach.

Here are the steps I recommend:
  1. Read the existing configuration file
  2. Modify the database settings

Some extra explanation here.
  3. Restart the service

That should do it!`

	plan := parsePlan(text)
	if len(plan.Steps) != 3 {
		t.Fatalf("expected 3 steps, got %d", len(plan.Steps))
	}
}

func TestParsePlan_ZeroSteps_ReturnsNil(t *testing.T) {
	text := "I'm not sure what you want me to do. Can you clarify?"
	plan := parsePlan(text)
	if plan != nil {
		t.Fatalf("expected nil plan for unparseable text, got %d steps", len(plan.Steps))
	}
}

// ── Format Step Summaries ──────────────────────────────────────────────

func TestFormatStepSummaries(t *testing.T) {
	steps := []PlanStep{
		{Index: 1, Description: "Read file", Status: StepDone, Result: "Found 100 lines"},
		{Index: 2, Description: "Write code", Status: StepFailed, Error: "syntax error"},
		{Index: 3, Description: "Run tests", Status: StepPending},
	}
	s := formatStepSummaries(steps)
	if s == "" {
		t.Fatal("expected non-empty summary")
	}
	if !contains(s, "Step 1 [done]") {
		t.Error("missing step 1 done")
	}
	if !contains(s, "Step 2 [failed]") {
		t.Error("missing step 2 failed")
	}
	if contains(s, "Step 3") {
		t.Error("pending step 3 should not appear")
	}
}

func contains(s, sub string) bool {
	return len(s) >= len(sub) && searchString(s, sub)
}

func searchString(s, sub string) bool {
	for i := 0; i <= len(s)-len(sub); i++ {
		if s[i:i+len(sub)] == sub {
			return true
		}
	}

	return false
}

// ── Planning Detection Tests ───────────────────────────────────────────

func TestNeedsPlanning(t *testing.T) {
	tests := []struct {
		input string
		want  bool
	}{
		{"what is Go?", false},
		{"fix the bug", false},
		{"hello", false},
		{"how does the API work?", false},
		{"Create an HTTP server with tests and deploy it to production", true},
		{"Build a web page with:\n- a navbar\n- a sidebar\n- a footer\n- a main content area", true},
		{"Create a Go HTTP server with tests, add a Dockerfile, and configure CI/CD", true},
		{"1. read the file\n2. fix the bug\n3. run tests", true},
	}
	for _, tt := range tests {
		got := needsPlanning(tt.input)
		if got != tt.want {
			t.Errorf("needsPlanning(%q) = %v, want %v", tt.input, got, tt.want)
		}
	}
}

// ── Mock Completion Function ───────────────────────────────────────────

// mockStreamComplete returns a StreamingCompletionFunc that sends a fixed response.
func mockStreamComplete(content string) StreamingCompletionFunc {
	return func(ctx context.Context, req *api.ChatCompletionRequest) (<-chan apiclient.StreamEvent, error) {
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

// ── Step Execution Test ────────────────────────────────────────────────

func TestRunPlannedStreaming_DelegatesToRunStreaming(t *testing.T) {
	// RunPlannedStreaming now delegates directly to RunStreaming (no planning).
	// Verify messages flow through correctly.
	complete := mockStreamComplete("Here is the complete implementation.")

	messages := []api.Message{
		{Role: "system", Content: "You are helpful."},
		{Role: "user", Content: "Create a Go HTTP server with tests, add a Dockerfile, and deploy it to production"},
	}

	cfg := PlanAgentConfig{
		StreamingConfig: StreamingConfig{
			Config: Config{
				MaxIterations: 5,
				Tools:         tools.NewRegistry(),
			},
		},
	}

	result, err := RunPlannedStreaming(context.Background(), complete, messages, cfg)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(result) == 0 {
		t.Fatal("expected non-empty result messages")
	}

	// Last message should be the assistant response
	last := result[len(result)-1]
	if last.Role != "assistant" {
		t.Errorf("expected last message role=assistant, got %s", last.Role)
	}
	if last.Content != "Here is the complete implementation." {
		t.Errorf("unexpected content: %s", last.Content)
	}
}

// ── User Injection Tests ───────────────────────────────────────────────

func TestHandleInjection_Stop(t *testing.T) {
	plan := &Plan{
		Steps: []PlanStep{
			{Index: 1, Status: StepDone, Result: "ok"},
			{Index: 2, Status: StepPending},
			{Index: 3, Status: StepPending},
		},
	}
	cfg := &PlanAgentConfig{}
	newPlan, idx := handleInjection(context.Background(), nil, nil, plan, 1, "/stop", "", cfg)
	if idx != -1 {
		t.Errorf("expected -1, got %d", idx)
	}
	if newPlan.Steps[1].Status != StepSkipped {
		t.Errorf("step 2 should be skipped, got %v", newPlan.Steps[1].Status)
	}
	if newPlan.Steps[2].Status != StepSkipped {
		t.Errorf("step 3 should be skipped, got %v", newPlan.Steps[2].Status)
	}
}

func TestHandleInjection_Skip(t *testing.T) {
	plan := &Plan{
		Steps: []PlanStep{
			{Index: 1, Status: StepDone, Result: "ok"},
			{Index: 2, Status: StepPending},
			{Index: 3, Status: StepPending},
		},
	}
	cfg := &PlanAgentConfig{}
	newPlan, idx := handleInjection(context.Background(), nil, nil, plan, 1, "/skip", "", cfg)
	if idx != 2 {
		t.Errorf("expected idx=2, got %d", idx)
	}
	if newPlan.Steps[1].Status != StepSkipped {
		t.Errorf("step 2 should be skipped, got %v", newPlan.Steps[1].Status)
	}
}

func TestHandleInjection_Redo(t *testing.T) {
	plan := &Plan{
		Steps: []PlanStep{
			{Index: 1, Status: StepDone, Result: "ok"},
			{Index: 2, Status: StepPending},
		},
	}
	cfg := &PlanAgentConfig{}
	newPlan, idx := handleInjection(context.Background(), nil, nil, plan, 1, "/redo", "", cfg)
	if idx != 0 {
		t.Errorf("expected idx=0, got %d", idx)
	}
	if newPlan.Steps[0].Status != StepPending {
		t.Errorf("step 1 should be reset to pending, got %v", newPlan.Steps[0].Status)
	}
}

// ── Continuous Loop Completes Without Planning ───────────────────────

func TestRunPlannedStreaming_NoPlanningPhase(t *testing.T) {
	// Verify that even complex requests go directly through RunStreaming
	// without any plan generation or step callbacks.
	complete := mockStreamComplete("I'll handle all of this in one go.")

	messages := []api.Message{
		{Role: "user", Content: "Create a new module, write the tests, and configure the CI pipeline"},
	}

	var planGenerated, stepStarted bool
	cfg := PlanAgentConfig{
		StreamingConfig: StreamingConfig{
			Config: Config{
				MaxIterations: 5,
				Tools:         tools.NewRegistry(),
			},
		},
		OnPlanGenerated: func(plan *Plan) { planGenerated = true },
		OnStepStart:     func(idx int, step *PlanStep) { stepStarted = true },
	}

	_, err := RunPlannedStreaming(context.Background(), complete, messages, cfg)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if planGenerated {
		t.Error("planning should not occur — RunPlannedStreaming delegates to RunStreaming")
	}
	if stepStarted {
		t.Error("step callbacks should not fire — RunPlannedStreaming delegates to RunStreaming")
	}
}

// ── Simple Question Passthrough ───────────────────────────────────────

func TestRunPlannedStreaming_SimpleQuestion(t *testing.T) {
	// Verify simple questions pass through to RunStreaming directly.
	complete := mockStreamComplete("Go is a programming language.")

	messages := []api.Message{
		{Role: "user", Content: "What is Go?"},
	}

	cfg := PlanAgentConfig{
		StreamingConfig: StreamingConfig{
			Config: Config{
				MaxIterations: 5,
				Tools:         tools.NewRegistry(),
			},
		},
	}

	result, err := RunPlannedStreaming(context.Background(), complete, messages, cfg)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(result) == 0 {
		t.Fatal("expected non-empty result")
	}
	last := result[len(result)-1]
	if last.Content != "Go is a programming language." {
		t.Errorf("unexpected content: %s", last.Content)
	}
}
