package agent

import (
	"fmt"
	"regexp"
	"strings"
)

// StepStatus tracks the execution state of a plan step.
type StepStatus int

const (
	StepPending StepStatus = iota
	StepRunning
	StepDone
	StepFailed
	StepSkipped
)

func (s StepStatus) String() string {
	switch s {
	case StepPending:
		return "pending"
	case StepRunning:
		return "running"
	case StepDone:
		return "done"
	case StepFailed:
		return "failed"
	case StepSkipped:
		return "skipped"
	default:
		return "unknown"
	}
}

// PlanStep is a single step in a plan.
type PlanStep struct {
	Index       int
	Description string
	Status      StepStatus
	Result      string // 1-2 sentence summary of outcome
	Error       string
}

// Plan holds the decomposed steps for a user request.
type Plan struct {
	Steps   []PlanStep
	RawText string
}

var stepPattern = regexp.MustCompile(`^\s*(\d+)[\.\)]\s+(.+)$`)

// needsPlanning returns true when the user's request is complex enough to
// benefit from plan-execute decomposition. Simple questions, short requests,
// and single-action tasks go straight to RunStreaming.
func needsPlanning(input string) bool {
	lower := strings.ToLower(input)

	// Very short inputs are never complex enough
	if len(input) < 40 {
		return false
	}

	// Questions don't need planning
	questionPrefixes := []string{
		"what ", "why ", "how does ", "how is ", "where ", "when ",
		"who ", "which ", "is ", "are ", "can you explain", "tell me about",
	}
	for _, q := range questionPrefixes {
		if strings.HasPrefix(lower, q) {
			return false
		}
	}

	// Count action signals — multiple distinct actions suggest planning
	actionVerbs := []string{
		"create", "build", "write", "add", "implement", "make",
		"fix", "update", "change", "modify", "refactor", "delete",
		"remove", "set up", "configure", "install", "test", "check",
		"debug", "deploy", "migrate", "convert",
	}
	seen := make(map[string]bool)
	for _, v := range actionVerbs {
		if strings.Contains(lower, v) {
			seen[v] = true
		}
	}

	// Multiple requirements (bullet points, numbered items)
	lines := strings.Split(input, "\n")
	listItems := 0
	for _, line := range lines {
		trimmed := strings.TrimSpace(line)
		if trimmed == "" {
			continue
		}
		if strings.HasPrefix(trimmed, "- ") || strings.HasPrefix(trimmed, "* ") ||
			(len(trimmed) > 2 && trimmed[0] >= '0' && trimmed[0] <= '9' && (trimmed[1] == '.' || trimmed[1] == ')')) {
			listItems++
		}
	}

	// Plan if: 2+ action verbs, or 3+ list items
	return len(seen) >= 2 || listItems >= 3
}

// parsePlan extracts numbered steps from LLM output. If zero steps are found,
// it wraps the original user request as a single step (degrades to current behavior).
func parsePlan(text, originalRequest string) *Plan {
	plan := &Plan{RawText: text}

	for _, line := range strings.Split(text, "\n") {
		m := stepPattern.FindStringSubmatch(line)
		if m == nil {
			continue
		}
		plan.Steps = append(plan.Steps, PlanStep{
			Index:       len(plan.Steps) + 1,
			Description: strings.TrimSpace(m[2]),
			Status:      StepPending,
		})
	}

	// Fallback: wrap entire request as a single step
	if len(plan.Steps) == 0 {
		plan.Steps = []PlanStep{{
			Index:       1,
			Description: originalRequest,
			Status:      StepPending,
		}}
	}

	return plan
}

// formatStepSummaries formats completed steps for injection into step context.
func formatStepSummaries(steps []PlanStep) string {
	var b strings.Builder
	for _, s := range steps {
		if s.Status == StepPending || s.Status == StepRunning {
			continue
		}
		status := s.Status.String()
		result := s.Result
		if s.Status == StepFailed && s.Error != "" {
			result = s.Error
		}
		if result == "" {
			result = "(no output)"
		}
		fmt.Fprintf(&b, "- Step %d [%s]: %s → %s\n", s.Index, status, s.Description, result)
	}

	return b.String()
}
