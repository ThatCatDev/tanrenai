package cmd

import (
	"strings"
	"testing"

	"github.com/ThatCatDev/tanrenai/shared/agent"
)

func TestSwarmToggleOnEnablesAgentMode(t *testing.T) {
	app := &tuiApp{swarmMode: false, agentMode: false}

	if !app.handleSlashCommand("/swarm on") {
		t.Fatal("expected /swarm on to be handled")
	}
	if !app.swarmMode {
		t.Error("expected swarmMode true after /swarm on")
	}
	if !app.agentMode {
		t.Error("expected agentMode true after /swarm on (swarm implies agent)")
	}
}

func TestSwarmToggleOffLeavesAgentMode(t *testing.T) {
	app := &tuiApp{swarmMode: true, agentMode: true}

	if !app.handleSlashCommand("/swarm off") {
		t.Fatal("expected /swarm off to be handled")
	}
	if app.swarmMode {
		t.Error("expected swarmMode false after /swarm off")
	}
	if !app.agentMode {
		t.Error("expected agentMode to remain true after /swarm off")
	}
}

func TestSwarmToggleRefusedDuringTurn(t *testing.T) {
	app := &tuiApp{swarmMode: false, agentMode: false, processing: true}

	if !app.handleSlashCommand("/swarm on") {
		t.Fatal("expected /swarm on to be handled (with refusal message)")
	}
	if app.swarmMode {
		t.Error("expected swarmMode to remain false while processing")
	}
	if app.agentMode {
		t.Error("expected agentMode to remain false while processing")
	}

	var found bool
	for _, line := range app.lines {
		if strings.Contains(line, "Cannot toggle /swarm") {
			found = true

			break
		}
	}
	if !found {
		t.Errorf("expected refusal message in output lines, got: %v", app.lines)
	}
}

// ── Panel helpers ─────────────────────────────────────────────────────

func mkSteps(statuses ...agent.StepStatus) []agent.PlanStep {
	steps := make([]agent.PlanStep, len(statuses))
	for i, s := range statuses {
		steps[i] = agent.PlanStep{Index: i + 1, Description: "task", Status: s}
	}

	return steps
}

func TestWindowStepsReturnsAllWhenUnderMax(t *testing.T) {
	steps := mkSteps(agent.StepDone, agent.StepRunning, agent.StepPending)
	start, end := windowSteps(steps, 4)
	if start != 0 || end != 3 {
		t.Errorf("got [%d,%d), want [0,3)", start, end)
	}
}

func TestWindowStepsCentersOnRunning(t *testing.T) {
	steps := mkSteps(
		agent.StepDone, agent.StepDone, agent.StepDone,
		agent.StepRunning, // index 3
		agent.StepPending, agent.StepPending, agent.StepPending, agent.StepPending,
	)
	start, end := windowSteps(steps, 4)
	if start != 2 || end != 6 {
		t.Errorf("got [%d,%d), want [2,6) (centered on running step at 3)", start, end)
	}
}

func TestWindowStepsFallsBackToFirstPending(t *testing.T) {
	steps := mkSteps(
		agent.StepDone, agent.StepDone,
		agent.StepPending, agent.StepPending, agent.StepPending, agent.StepPending,
	)
	start, end := windowSteps(steps, 4)
	if start != 1 || end != 5 {
		t.Errorf("got [%d,%d), want [1,5) (centered on first pending at 2)", start, end)
	}
}

func TestWindowStepsAllDoneShowsLast(t *testing.T) {
	steps := mkSteps(
		agent.StepDone, agent.StepDone, agent.StepDone,
		agent.StepDone, agent.StepDone, agent.StepDone,
	)
	start, end := windowSteps(steps, 4)
	if start != 2 || end != 6 {
		t.Errorf("got [%d,%d), want [2,6) (last 4 when all done)", start, end)
	}
}

func TestRenderStepProgressBar(t *testing.T) {
	cases := []struct {
		done, total int
		wantSuffix  string
		wantFilled  int
	}{
		{0, 4, "0/4", 0},
		{2, 4, "2/4", 10},
		{4, 4, "4/4", 20},
		{0, 0, "", 0},
		{5, 4, "4/4", 20}, // clamp
	}
	for _, c := range cases {
		got := renderStepProgressBar(c.done, c.total)
		if c.total == 0 {
			if got != "" {
				t.Errorf("renderStepProgressBar(%d,%d)=%q, want empty", c.done, c.total, got)
			}

			continue
		}
		if !strings.Contains(got, c.wantSuffix) {
			t.Errorf("renderStepProgressBar(%d,%d)=%q, want suffix %q", c.done, c.total, got, c.wantSuffix)
		}
		if strings.Count(got, "#") != c.wantFilled {
			t.Errorf("renderStepProgressBar(%d,%d): got %d filled, want %d", c.done, c.total, strings.Count(got, "#"), c.wantFilled)
		}
	}
}

func TestStepIconColorsByStatus(t *testing.T) {
	cases := map[agent.StepStatus]string{
		agent.StepPending: "gray",
		agent.StepRunning: "yellow",
		agent.StepDone:    "green",
		agent.StepFailed:  "red",
		agent.StepSkipped: "gray",
	}
	for status, color := range cases {
		got := stepIcon(status)
		if !strings.Contains(got, color) {
			t.Errorf("stepIcon(%v)=%q, expected color %q", status, got, color)
		}
	}
}

func TestRenderSwarmPanelLinesAlwaysReturnsFiveLines(t *testing.T) {
	cases := []*agent.Plan{
		nil,
		{Steps: nil},
		{Steps: mkSteps(agent.StepRunning)},
		{Steps: mkSteps(agent.StepDone, agent.StepRunning, agent.StepPending)},
		{Steps: mkSteps(agent.StepDone, agent.StepDone, agent.StepRunning,
			agent.StepPending, agent.StepPending, agent.StepPending)},
	}
	for i, plan := range cases {
		lines := renderSwarmPanelLines(plan, 0)
		if len(lines) != swarmPanelMaxSteps+1 {
			t.Errorf("case %d: got %d lines, want %d", i, len(lines), swarmPanelMaxSteps+1)
		}
	}
}

func TestRenderSwarmPanelLinesShowsRangeWhenTruncated(t *testing.T) {
	plan := &agent.Plan{Steps: mkSteps(
		agent.StepDone, agent.StepDone, agent.StepRunning,
		agent.StepPending, agent.StepPending, agent.StepPending, agent.StepPending,
	)}
	lines := renderSwarmPanelLines(plan, 0)
	header := lines[0]
	if !strings.Contains(header, "of 7") {
		t.Errorf("expected total count in header, got: %q", header)
	}
	if !strings.Contains(header, "2/7") {
		t.Errorf("expected 2/7 progress, got: %q", header)
	}
}

func TestRenderSwarmPanelLinesShowsDepth(t *testing.T) {
	plan := &agent.Plan{Steps: mkSteps(agent.StepRunning)}
	lines := renderSwarmPanelLines(plan, 2)
	if !strings.Contains(lines[0], "depth 2") {
		t.Errorf("expected depth marker in header, got: %q", lines[0])
	}
}

func TestSwarmStatusReportsCurrentState(t *testing.T) {
	cases := []struct {
		swarm bool
		want  string
	}{
		{true, "swarm mode: on"},
		{false, "swarm mode: off"},
	}
	for _, c := range cases {
		app := &tuiApp{swarmMode: c.swarm}
		if !app.handleSlashCommand("/swarm") {
			t.Fatalf("expected /swarm to be handled (swarm=%v)", c.swarm)
		}
		var found bool
		for _, line := range app.lines {
			if strings.Contains(line, c.want) {
				found = true

				break
			}
		}
		if !found {
			t.Errorf("expected %q in output for swarm=%v, got: %v", c.want, c.swarm, app.lines)
		}
	}
}
