package training

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"

	"github.com/ThatCatDev/tanrenai/server/pkg/api"
)

// ABResult contains the result of an A/B test between two models.
type ABResult struct {
	BaseModel    string         `json:"base_model"`
	FTModel      string         `json:"ft_model"`
	Comparisons  []ABComparison `json:"comparisons"`
	BaseTotalMs  int64          `json:"base_total_ms"`
	FTTotalMs    int64          `json:"ft_total_ms"`
}

// ABComparison is a single prompt/response pair from both models.
type ABComparison struct {
	Prompt       string `json:"prompt"`
	BaseResponse string `json:"base_response"`
	FTResponse   string `json:"ft_response"`
	BaseLatency  int64  `json:"base_latency_ms"`
	FTLatency    int64  `json:"ft_latency_ms"`
}

// ABTest compares a base model against a fine-tuned model.
type ABTest struct {
	baseURL string // base URL of the tanrenai server
}

// NewABTest creates an ABTest using the given server URL.
func NewABTest(baseURL string) *ABTest {
	return &ABTest{baseURL: baseURL}
}

// Run sends each prompt to both models and collects responses.
func (t *ABTest) Run(ctx context.Context, baseModel, ftModel string, prompts []string) (*ABResult, error) {
	result := &ABResult{
		BaseModel: baseModel,
		FTModel:   ftModel,
	}

	for _, prompt := range prompts {
		msgs := []api.Message{
			{Role: "user", Content: prompt},
		}

		// Query base model
		baseResp, baseMs, err := t.query(ctx, baseModel, msgs)
		if err != nil {
			return nil, fmt.Errorf("base model query failed: %w", err)
		}

		// Query fine-tuned model
		ftResp, ftMs, err := t.query(ctx, ftModel, msgs)
		if err != nil {
			return nil, fmt.Errorf("fine-tuned model query failed: %w", err)
		}

		result.Comparisons = append(result.Comparisons, ABComparison{
			Prompt:       prompt,
			BaseResponse: baseResp,
			FTResponse:   ftResp,
			BaseLatency:  baseMs,
			FTLatency:    ftMs,
		})

		result.BaseTotalMs += baseMs
		result.FTTotalMs += ftMs
	}

	return result, nil
}

func (t *ABTest) query(ctx context.Context, model string, msgs []api.Message) (string, int64, error) {
	req := api.ChatCompletionRequest{
		Model:    model,
		Messages: msgs,
	}

	body, err := json.Marshal(req)
	if err != nil {
		return "", 0, fmt.Errorf("marshal request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, t.baseURL+"/v1/chat/completions", bytes.NewReader(body))
	if err != nil {
		return "", 0, fmt.Errorf("create request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")

	start := time.Now()
	resp, err := http.DefaultClient.Do(httpReq)
	if err != nil {
		return "", 0, fmt.Errorf("send request: %w", err)
	}
	defer resp.Body.Close()
	latencyMs := time.Since(start).Milliseconds()

	if resp.StatusCode != http.StatusOK {
		respBody, _ := io.ReadAll(resp.Body)
		return "", 0, fmt.Errorf("server returned %d: %s", resp.StatusCode, string(respBody))
	}

	var result api.ChatCompletionResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return "", 0, fmt.Errorf("decode response: %w", err)
	}

	if len(result.Choices) == 0 {
		return "", latencyMs, nil
	}

	return result.Choices[0].Message.Content, latencyMs, nil
}
