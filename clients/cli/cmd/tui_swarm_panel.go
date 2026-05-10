package cmd

import (
	"fmt"
	"strings"

	"github.com/rivo/tview"

	"github.com/ThatCatDev/tanrenai/shared/agent"
)

const swarmPanelMaxSteps = 4

// stepIcon returns a checkbox-like glyph reflecting the step's status.
// The whole token (brackets included) is wrapped in a color tag so the
// surrounding text picks up the right hue.
func stepIcon(s agent.StepStatus) string {
	switch s {
	case agent.StepPending:
		return "[gray::-][ ][-:-:-]"
	case agent.StepRunning:
		return "[yellow::b][>][-:-:-]"
	case agent.StepDone:
		return "[green::-][x][-:-:-]"
	case agent.StepFailed:
		return "[red::-][!][-:-:-]"
	case agent.StepSkipped:
		return "[gray::-][-][-:-:-]"
	default:
		return "[gray::-][?][-:-:-]"
	}
}

// renderStepProgressBar returns a 20-char ASCII progress bar with an
// "x/y" suffix. Returns "" if total is zero.
func renderStepProgressBar(done, total int) string {
	const width = 20
	if total <= 0 {
		return ""
	}
	if done < 0 {
		done = 0
	}
	if done > total {
		done = total
	}
	filled := done * width / total
	bar := strings.Repeat("#", filled) + strings.Repeat("-", width-filled)

	return fmt.Sprintf("[gray::-][%s] %d/%d[-:-:-]", bar, done, total)
}

// windowSteps picks a sliding window of up to `max` steps. Centers on the
// running step if any, else the first pending, else the last (all done).
// Returns half-open [start, end). For len(steps) <= max, returns the full range.
func windowSteps(steps []agent.PlanStep, max int) (start, end int) {
	n := len(steps)
	if n <= max {
		return 0, n
	}
	cur := -1
	for i, s := range steps {
		if s.Status == agent.StepRunning {
			cur = i

			break
		}
	}
	if cur == -1 {
		for i, s := range steps {
			if s.Status == agent.StepPending {
				cur = i

				break
			}
		}
	}
	if cur == -1 {
		cur = n - 1
	}

	start = cur - 1
	if start < 0 {
		start = 0
	}
	end = start + max
	if end > n {
		end = n
		start = end - max
	}

	return start, end
}

// countDoneSteps counts steps in a terminal state (done/failed/skipped).
func countDoneSteps(steps []agent.PlanStep) int {
	n := 0
	for _, s := range steps {
		switch s.Status {
		case agent.StepDone, agent.StepFailed, agent.StepSkipped:
			n++
		}
	}

	return n
}

// renderSwarmPanelLines builds the swarm-panel text as a list of lines.
// Always returns swarmPanelMaxSteps+1 lines (header + 4 step rows), padding
// with empty strings if fewer steps. Pure function — no tview side effects.
func renderSwarmPanelLines(plan *agent.Plan, depth int) []string {
	lines := make([]string, 0, swarmPanelMaxSteps+1)

	if plan == nil || len(plan.Steps) == 0 {
		lines = append(lines, "[blue::b]  Swarm Plan[-:-:-]  [gray::-]planning...[-:-:-]")
		for i := 0; i < swarmPanelMaxSteps; i++ {
			lines = append(lines, "")
		}

		return lines
	}

	steps := plan.Steps
	done := countDoneSteps(steps)
	total := len(steps)
	start, end := windowSteps(steps, swarmPanelMaxSteps)

	title := "Swarm Plan"
	if depth > 0 {
		title = fmt.Sprintf("Swarm Plan (depth %d)", depth)
	}
	header := fmt.Sprintf("[blue::b]  %s[-:-:-]  %s", title, renderStepProgressBar(done, total))
	if total > swarmPanelMaxSteps {
		header += fmt.Sprintf("  [gray::-](%d-%d of %d)[-:-:-]", start+1, end, total)
	}
	lines = append(lines, header)

	for _, step := range steps[start:end] {
		desc := truncate(step.Description, 80)
		lines = append(lines, fmt.Sprintf("  %s %d. %s", stepIcon(step.Status), step.Index, tview.Escape(desc)))
	}
	for len(lines) < swarmPanelMaxSteps+1 {
		lines = append(lines, "")
	}

	return lines
}

// refreshSwarmPanel writes the current plan into the panel widget.
// Caller must hold the UI thread (i.e. inside QueueUpdateDraw).
func (t *tuiApp) refreshSwarmPanel() {
	if t.swarmPanel == nil {
		return
	}
	lines := renderSwarmPanelLines(t.swarmPlan, t.swarmDepth)
	t.swarmPanel.SetText(strings.Join(lines, "\n"))
}

// setSwarmPanelVisible toggles the panel's row reservation in the root flex.
// When visible, reserves swarmPanelMaxSteps+1 rows.
// Caller must hold the UI thread.
func (t *tuiApp) setSwarmPanelVisible(show bool) {
	if t.rootFlex == nil || t.swarmPanel == nil {
		return
	}
	if show {
		t.rootFlex.ResizeItem(t.swarmPanel, swarmPanelMaxSteps+1, 0)
	} else {
		t.rootFlex.ResizeItem(t.swarmPanel, 0, 0)
		t.swarmPanel.SetText("")
	}
}
