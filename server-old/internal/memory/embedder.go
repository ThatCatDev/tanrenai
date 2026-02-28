package memory

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net/http"
)

// EmbedFunc is a function that produces a float32 embedding vector from text.
type EmbedFunc func(ctx context.Context, text string) ([]float32, error)

// defaultMaxEmbedTokens is the fallback context size for embedding models.
const defaultMaxEmbedTokens = 256

// charsPerToken is a conservative estimate for truncation.
// Using 3.0 instead of 3.5 to avoid overflowing the embedding model's context.
const charsPerToken = 3.0

// NewLlamaEmbedFunc returns an EmbedFunc that calls a llama-server /v1/embeddings endpoint.
// maxTokens is the embedding model's context window size (e.g. 256 for MiniLM, 8192 for nomic-embed).
// Pass 0 to use the default (256).
func NewLlamaEmbedFunc(baseURL string, maxTokens int) EmbedFunc {
	if maxTokens <= 0 {
		maxTokens = defaultMaxEmbedTokens
	}
	// Use 80% of the context to leave headroom for tokenizer variance
	maxChars := int(float64(maxTokens) * charsPerToken * 0.8)

	return func(ctx context.Context, text string) ([]float32, error) {
		if len(text) > maxChars {
			text = text[:maxChars]
		}

		reqBody := embeddingRequest{
			Input: text,
			Model: "embedding",
		}
		body, err := json.Marshal(reqBody)
		if err != nil {
			return nil, fmt.Errorf("marshal embedding request: %w", err)
		}

		httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, baseURL+"/v1/embeddings", bytes.NewReader(body))
		if err != nil {
			return nil, fmt.Errorf("create embedding request: %w", err)
		}
		httpReq.Header.Set("Content-Type", "application/json")

		resp, err := http.DefaultClient.Do(httpReq)
		if err != nil {
			return nil, fmt.Errorf("embedding request failed: %w", err)
		}
		defer resp.Body.Close()

		if resp.StatusCode != http.StatusOK {
			var errBody []byte
			errBody, _ = io.ReadAll(resp.Body)
			return nil, fmt.Errorf("embedding server returned %d: %s", resp.StatusCode, string(errBody))
		}

		var result embeddingResponse
		if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
			return nil, fmt.Errorf("decode embedding response: %w", err)
		}

		if len(result.Data) == 0 {
			return nil, fmt.Errorf("embedding response contained no data")
		}

		vec := result.Data[0].Embedding
		normalizeVector(vec)
		return vec, nil
	}
}

// normalizeVector normalizes a vector to unit length in-place.
func normalizeVector(v []float32) {
	var sum float64
	for _, x := range v {
		sum += float64(x) * float64(x)
	}
	norm := float32(math.Sqrt(sum))
	if norm == 0 {
		return
	}
	for i := range v {
		v[i] /= norm
	}
}

// OpenAI-compatible embedding request/response types.
type embeddingRequest struct {
	Input string `json:"input"`
	Model string `json:"model"`
}

type embeddingResponse struct {
	Data []embeddingData `json:"data"`
}

type embeddingData struct {
	Embedding []float32 `json:"embedding"`
	Index     int       `json:"index"`
}
