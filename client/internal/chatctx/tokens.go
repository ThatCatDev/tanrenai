package chatctx

import (
	"encoding/json"
	"math"
	"strings"

	"github.com/ThatCatDev/tanrenai/client/pkg/api"
)

const (
	defaultCharsPerToken = 3.5
	calibrationSample    = "The quick brown fox jumps over the lazy dog. " +
		"Pack my box with five dozen liquor jugs. " +
		"How vexingly quick daft zebras jump! " +
		"The five boxing wizards jump quickly. " +
		"Sphinx of black quartz, judge my vow. " +
		"Two driven jocks help fax my big quiz. " +
		"The jay, pig, fox, zebra and my wolves quack! " +
		"Amazingly few discotheques provide jukeboxes. " +
		"Heavy boxes perform quick waltzes and jigs. " +
		"Jackdaws love my big sphinx of quartz."
	roleOverheadTokens = 4 // per-message overhead for role, separators, etc.
)

// TokenEstimator estimates token counts using a calibrated chars-per-token ratio.
// It defaults to a conservative ratio and can be calibrated against a real tokenizer.
type TokenEstimator struct {
	charsPerToken float64
	calibrated    bool
}

// NewTokenEstimator creates a TokenEstimator with the default ratio.
func NewTokenEstimator() *TokenEstimator {
	return &TokenEstimator{
		charsPerToken: defaultCharsPerToken,
	}
}

// Calibrate sends a sample string to the provided tokenize function and
// adjusts the chars-per-token ratio accordingly. If calibration fails,
// the estimator keeps using the default ratio.
func (e *TokenEstimator) Calibrate(tokenizeFn func(string) (int, error)) error {
	tokenCount, err := tokenizeFn(calibrationSample)
	if err != nil {
		return err
	}
	if tokenCount > 0 {
		e.charsPerToken = float64(len(calibrationSample)) / float64(tokenCount)
		e.calibrated = true
	}
	return nil
}

// Calibrated returns whether the estimator has been calibrated against a real tokenizer.
func (e *TokenEstimator) Calibrated() bool {
	return e.calibrated
}

// Estimate returns the estimated token count for the given text.
func (e *TokenEstimator) Estimate(text string) int {
	if text == "" {
		return 0
	}
	return int(math.Ceil(float64(len(text)) / e.charsPerToken))
}

// EstimateMessages returns the estimated total tokens for a slice of messages.
// Includes per-message overhead for role tokens and structural overhead for tool calls.
func (e *TokenEstimator) EstimateMessages(msgs []api.Message) int {
	total := 0
	for _, msg := range msgs {
		total += roleOverheadTokens
		total += e.Estimate(msg.Content)

		// Account for tool call structure
		for _, tc := range msg.ToolCalls {
			total += roleOverheadTokens // overhead for tool call structure
			total += e.Estimate(tc.Function.Name)
			total += e.Estimate(tc.Function.Arguments)
		}

		// Account for tool call ID and name in tool responses
		if msg.ToolCallID != "" {
			total += e.Estimate(msg.ToolCallID)
			total += e.Estimate(msg.Name)
		}
	}
	return total
}

// EstimateJSON estimates tokens for a JSON-serializable value by marshaling it first.
func (e *TokenEstimator) EstimateJSON(v any) int {
	data, err := json.Marshal(v)
	if err != nil {
		return 0
	}
	return e.Estimate(string(data))
}

// MessageText returns the combined text content of a message for estimation purposes.
func MessageText(msg api.Message) string {
	var b strings.Builder
	b.WriteString(msg.Role)
	b.WriteString(msg.Content)
	for _, tc := range msg.ToolCalls {
		b.WriteString(tc.Function.Name)
		b.WriteString(tc.Function.Arguments)
	}
	if msg.ToolCallID != "" {
		b.WriteString(msg.ToolCallID)
		b.WriteString(msg.Name)
	}
	return b.String()
}
