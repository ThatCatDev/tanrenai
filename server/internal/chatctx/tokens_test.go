package chatctx

import (
	"errors"
	"testing"

	"github.com/thatcatdev/tanrenai/server/pkg/api"
)

func TestEstimateEmpty(t *testing.T) {
	e := NewTokenEstimator()
	if got := e.Estimate(""); got != 0 {
		t.Errorf("Estimate(\"\") = %d, want 0", got)
	}
}

func TestEstimateText(t *testing.T) {
	e := NewTokenEstimator()

	// With default 3.5 chars/token, "hello" (5 chars) = ceil(5/3.5) = 2
	got := e.Estimate("hello")
	if got != 2 {
		t.Errorf("Estimate(\"hello\") = %d, want 2", got)
	}

	// Longer text: 35 chars = ceil(35/3.5) = 10
	got = e.Estimate("The quick brown fox jumps over dog")
	want := 10 // ceil(34/3.5) = 10
	if got != want {
		t.Errorf("Estimate(34 chars) = %d, want %d", got, want)
	}
}

func TestEstimateMessages(t *testing.T) {
	e := NewTokenEstimator()

	msgs := []api.Message{
		{Role: "system", Content: "You are helpful."},
		{Role: "user", Content: "Hello"},
	}

	total := e.EstimateMessages(msgs)
	if total <= 0 {
		t.Errorf("EstimateMessages returned %d, want > 0", total)
	}

	// Should be greater than just the content alone (due to role overhead)
	contentOnly := e.Estimate("You are helpful.") + e.Estimate("Hello")
	if total <= contentOnly {
		t.Errorf("EstimateMessages (%d) should be greater than content-only estimate (%d) due to role overhead", total, contentOnly)
	}
}

func TestEstimateMessageWithToolCalls(t *testing.T) {
	e := NewTokenEstimator()

	msg := api.Message{
		Role: "assistant",
		ToolCalls: []api.ToolCall{
			{
				ID:   "call_1",
				Type: "function",
				Function: api.ToolCallFunction{
					Name:      "file_read",
					Arguments: `{"path": "main.go"}`,
				},
			},
		},
	}

	total := e.EstimateMessages([]api.Message{msg})
	if total <= roleOverheadTokens {
		t.Errorf("Message with tool calls should estimate more than just role overhead, got %d", total)
	}
}

func TestEstimateToolResponse(t *testing.T) {
	e := NewTokenEstimator()

	msg := api.Message{
		Role:       "tool",
		Content:    "file contents here",
		ToolCallID: "call_1",
		Name:       "file_read",
	}

	total := e.EstimateMessages([]api.Message{msg})
	if total <= roleOverheadTokens {
		t.Errorf("Tool response should estimate more than just role overhead, got %d", total)
	}
}

func TestCalibrateSuccess(t *testing.T) {
	e := NewTokenEstimator()

	// Mock tokenizer: 1 token per word
	tokenizeFn := func(text string) (int, error) {
		return 100, nil // fixed count
	}

	err := e.Calibrate(tokenizeFn)
	if err != nil {
		t.Fatalf("Calibrate failed: %v", err)
	}

	if !e.Calibrated() {
		t.Error("expected Calibrated() to return true after successful calibration")
	}

	// After calibration, ratio should be len(calibrationSample)/100
	expectedRatio := float64(len(calibrationSample)) / 100.0
	if e.charsPerToken != expectedRatio {
		t.Errorf("charsPerToken = %f, want %f", e.charsPerToken, expectedRatio)
	}
}

func TestCalibrateFallback(t *testing.T) {
	e := NewTokenEstimator()

	// Mock tokenizer that fails
	tokenizeFn := func(text string) (int, error) {
		return 0, errors.New("connection refused")
	}

	err := e.Calibrate(tokenizeFn)
	if err == nil {
		t.Error("Calibrate should return error when tokenizer fails")
	}

	if e.Calibrated() {
		t.Error("expected Calibrated() to return false after failed calibration")
	}

	// Should still use default ratio
	if e.charsPerToken != defaultCharsPerToken {
		t.Errorf("charsPerToken = %f, want default %f", e.charsPerToken, defaultCharsPerToken)
	}

	// Should still work for estimation
	got := e.Estimate("hello")
	if got <= 0 {
		t.Errorf("Estimate should work with default ratio, got %d", got)
	}
}

func TestCalibrateZeroTokens(t *testing.T) {
	e := NewTokenEstimator()

	tokenizeFn := func(text string) (int, error) {
		return 0, nil
	}

	err := e.Calibrate(tokenizeFn)
	if err != nil {
		t.Fatalf("Calibrate failed: %v", err)
	}

	// Should not calibrate with zero tokens (division by zero protection)
	if e.Calibrated() {
		t.Error("should not calibrate with zero tokens")
	}
	if e.charsPerToken != defaultCharsPerToken {
		t.Errorf("charsPerToken should remain default, got %f", e.charsPerToken)
	}
}

func TestMessageText(t *testing.T) {
	msg := api.Message{
		Role:    "assistant",
		Content: "hello",
		ToolCalls: []api.ToolCall{
			{Function: api.ToolCallFunction{Name: "test", Arguments: "{}"}},
		},
	}

	text := MessageText(msg)
	if text == "" {
		t.Error("MessageText should return non-empty string")
	}
	if len(text) < len("assistant")+len("hello")+len("test")+len("{}") {
		t.Error("MessageText should include role, content, and tool call info")
	}
}

func TestEstimateJSON(t *testing.T) {
	e := NewTokenEstimator()

	got := e.EstimateJSON(map[string]string{"key": "value"})
	if got <= 0 {
		t.Errorf("EstimateJSON should return > 0, got %d", got)
	}
}
