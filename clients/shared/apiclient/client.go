package apiclient

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/ThatCatDev/tanrenai/shared/pkg/api"
)

// Client is a typed HTTP client that talks to the tanrenai backend server.
type Client struct {
	baseURL      string
	authToken    string       // optional Bearer token for authenticated requests
	httpClient   *http.Client // non-streaming requests (has timeout)
	streamClient *http.Client // streaming requests (no timeout)
}

// New creates a new Client for the given backend URL.
func New(baseURL string) *Client {
	return &Client{
		baseURL:      baseURL,
		httpClient:   &http.Client{Timeout: 2 * time.Minute},
		streamClient: &http.Client{},
	}
}

// SetAuthToken sets the Bearer token for authenticated requests.
func (c *Client) SetAuthToken(token string) {
	c.authToken = token
}

// applyAuth adds the Authorization header if an auth token is set.
func (c *Client) applyAuth(req *http.Request) {
	if c.authToken != "" {
		req.Header.Set("Authorization", "Bearer "+c.authToken)
	}
}

// BaseURL returns the client's base URL.
func (c *Client) BaseURL() string {
	return c.baseURL
}

// SetBaseURL updates the client's base URL.
func (c *Client) SetBaseURL(url string) {
	c.baseURL = url
}

// --- Completions (proxied through backend to GPU) ---

// StreamCompletion sends a streaming chat completion request and returns a channel of events.
func (c *Client) StreamCompletion(ctx context.Context, req *api.ChatCompletionRequest) (<-chan StreamEvent, error) {
	req.Stream = true
	body, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("marshal request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+"/v1/chat/completions", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("create request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	c.applyAuth(httpReq)

	resp, err := c.streamClient.Do(httpReq)
	if err != nil {
		return nil, connError(err)
	}

	if resp.StatusCode != http.StatusOK {
		respBody, _ := io.ReadAll(resp.Body)
		_ = resp.Body.Close()

		return nil, &StatusError{Code: resp.StatusCode, Body: string(respBody)}
	}

	events := ParseSSEStream(resp.Body)

	return wrapStreamWithCleanup(events, resp.Body), nil
}

// ChatCompletion sends a non-streaming chat completion request.
func (c *Client) ChatCompletion(ctx context.Context, req *api.ChatCompletionRequest) (*api.ChatCompletionResponse, error) {
	req.Stream = false
	body, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("marshal request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+"/v1/chat/completions", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("create request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	c.applyAuth(httpReq)

	resp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return nil, connError(err)
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		respBody, _ := io.ReadAll(resp.Body)

		return nil, &StatusError{Code: resp.StatusCode, Body: string(respBody)}
	}

	var result api.ChatCompletionResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("decode response: %w", err)
	}

	return &result, nil
}

// --- Memory (handled by backend) ---

// MemorySearch searches memories for the given query.
func (c *Client) MemorySearch(ctx context.Context, query string, limit int) (*api.MemorySearchResponse, error) {
	req := api.MemorySearchRequest{Query: query, Limit: limit}
	body, _ := json.Marshal(req)

	var result api.MemorySearchResponse
	if err := c.postJSON(ctx, "/v1/memory/search", body, &result); err != nil {
		return nil, err
	}

	return &result, nil
}

// MemoryStore stores a conversation turn in memory.
func (c *Client) MemoryStore(ctx context.Context, userMsg, assistMsg string) (string, error) {
	req := api.MemoryStoreRequest{UserMsg: userMsg, AssistMsg: assistMsg}
	body, _ := json.Marshal(req)

	var result api.MemoryStoreResponse
	if err := c.postJSON(ctx, "/v1/memory/store", body, &result); err != nil {
		return "", err
	}

	return result.ID, nil
}

// MemoryList lists recent memory entries.
func (c *Client) MemoryList(ctx context.Context, limit int) (*api.MemoryListResponse, error) {
	url := fmt.Sprintf("%s/v1/memory/list?limit=%d", c.baseURL, limit)
	var result api.MemoryListResponse
	if err := c.getJSON(ctx, url, &result); err != nil {
		return nil, err
	}

	return &result, nil
}

// MemoryDelete deletes a memory entry by ID.
func (c *Client) MemoryDelete(ctx context.Context, id string) error {
	url := fmt.Sprintf("%s/v1/memory/%s", c.baseURL, id)
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodDelete, url, nil)
	if err != nil {
		return err
	}
	c.applyAuth(httpReq)
	resp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return connError(err)
	}
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode != http.StatusOK {
		respBody, _ := io.ReadAll(resp.Body)

		return &StatusError{Code: resp.StatusCode, Body: string(respBody)}
	}

	return nil
}

// MemoryClear clears all memories.
func (c *Client) MemoryClear(ctx context.Context) error {
	url := fmt.Sprintf("%s/v1/memory", c.baseURL)
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodDelete, url, nil)
	if err != nil {
		return err
	}
	c.applyAuth(httpReq)
	resp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return connError(err)
	}
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode != http.StatusOK {
		respBody, _ := io.ReadAll(resp.Body)

		return &StatusError{Code: resp.StatusCode, Body: string(respBody)}
	}

	return nil
}

// MemoryCount returns the number of stored memories.
func (c *Client) MemoryCount(ctx context.Context) (int, error) {
	var result api.MemoryCountResponse
	url := fmt.Sprintf("%s/v1/memory/count", c.baseURL)
	if err := c.getJSON(ctx, url, &result); err != nil {
		return 0, err
	}

	return result.Count, nil
}

// --- Models (proxied through backend to GPU) ---

// LoadModel loads a model by name on the GPU server and returns the load response.
func (c *Client) LoadModel(ctx context.Context, model string) (*api.LoadResponse, error) {
	body, _ := json.Marshal(map[string]string{"model": model})
	var result api.LoadResponse
	if err := c.postJSON(ctx, "/api/load", body, &result); err != nil {
		return nil, err
	}

	return &result, nil
}

// ListModels returns available models from the GPU server.
func (c *Client) ListModels(ctx context.Context) (*api.ModelListResponse, error) {
	var result api.ModelListResponse
	url := fmt.Sprintf("%s/v1/models", c.baseURL)
	if err := c.getJSON(ctx, url, &result); err != nil {
		return nil, err
	}

	return &result, nil
}

// --- Model Pull (proxied through backend to GPU) ---

// PullModelEvent represents a parsed SSE event during a model download.
type PullModelEvent struct {
	Event api.PullEvent
	Done  bool
	Err   error
}

// PullModel starts downloading a model by URL and streams progress events.
// The returned channel is closed when the download completes, errors, or ctx is cancelled.
func (c *Client) PullModel(ctx context.Context, url string) (<-chan PullModelEvent, error) {
	req := api.PullRequest{URL: url}
	body, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("marshal request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+"/api/pull", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("create request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	c.applyAuth(httpReq)

	resp, err := c.streamClient.Do(httpReq)
	if err != nil {
		return nil, connError(err)
	}

	if resp.StatusCode != http.StatusOK {
		respBody, _ := io.ReadAll(resp.Body)
		_ = resp.Body.Close()

		return nil, &StatusError{Code: resp.StatusCode, Body: string(respBody)}
	}

	ch := make(chan PullModelEvent)
	go func() {
		defer func() { _ = resp.Body.Close() }()
		defer close(ch)
		scanner := bufio.NewScanner(resp.Body)
		for scanner.Scan() {
			line := scanner.Text()
			if !strings.HasPrefix(line, "data: ") {
				continue
			}
			data := strings.TrimPrefix(line, "data: ")
			if data == "[DONE]" {
				return
			}
			var ev api.PullEvent
			if err := json.Unmarshal([]byte(data), &ev); err != nil {
				ch <- PullModelEvent{Err: err}

				return
			}
			ch <- PullModelEvent{Event: ev}
			if ev.Status == "downloaded" || ev.Status == "error" {
				return
			}
		}
		if err := scanner.Err(); err != nil {
			ch <- PullModelEvent{Err: err}
		}
	}()

	return ch, nil
}

// --- Tokenize (proxied through backend to GPU) ---

// Tokenize returns the token count for the given text.
func (c *Client) Tokenize(ctx context.Context, text string) (int, error) {
	payload := struct {
		Content string `json:"content"`
	}{Content: text}
	body, _ := json.Marshal(payload)

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+"/tokenize", bytes.NewReader(body))
	if err != nil {
		return 0, err
	}
	httpReq.Header.Set("Content-Type", "application/json")
	c.applyAuth(httpReq)

	resp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return 0, connError(err)
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		respBody, _ := io.ReadAll(resp.Body)

		return 0, &StatusError{Code: resp.StatusCode, Body: string(respBody)}
	}

	var result struct {
		Tokens []int `json:"tokens"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return 0, err
	}

	return len(result.Tokens), nil
}

// --- Instance management (backend manages vast.ai) ---

// InstanceStatus returns the GPU instance status.
func (c *Client) InstanceStatus(ctx context.Context) (*api.InstanceStatus, error) {
	var result api.InstanceStatus
	url := fmt.Sprintf("%s/api/instance/status", c.baseURL)
	if err := c.getJSON(ctx, url, &result); err != nil {
		return nil, err
	}

	return &result, nil
}

// InstanceStart starts the GPU instance.
func (c *Client) InstanceStart(ctx context.Context) error {
	return c.postJSON(ctx, "/api/instance/start", nil, nil)
}

// InstanceStop stops the GPU instance.
func (c *Client) InstanceStop(ctx context.Context) error {
	return c.postJSON(ctx, "/api/instance/stop", nil, nil)
}

// --- Internal helpers ---

func (c *Client) postJSON(ctx context.Context, path string, body []byte, result any) error {
	var reader io.Reader
	if body != nil {
		reader = bytes.NewReader(body)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+path, reader)
	if err != nil {
		return fmt.Errorf("create request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	c.applyAuth(httpReq)

	resp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return connError(err)
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		respBody, _ := io.ReadAll(resp.Body)

		return &StatusError{Code: resp.StatusCode, Body: string(respBody)}
	}

	if result != nil {
		if err := json.NewDecoder(resp.Body).Decode(result); err != nil {
			return fmt.Errorf("decode response: %w", err)
		}
	}

	return nil
}

func (c *Client) getJSON(ctx context.Context, url string, result any) error {
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return fmt.Errorf("create request: %w", err)
	}
	c.applyAuth(httpReq)

	resp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return connError(err)
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		respBody, _ := io.ReadAll(resp.Body)

		return &StatusError{Code: resp.StatusCode, Body: string(respBody)}
	}

	if result != nil {
		if err := json.NewDecoder(resp.Body).Decode(result); err != nil {
			return fmt.Errorf("decode response: %w", err)
		}
	}

	return nil
}

// connError wraps a connection-level error as ErrServerUnavailable when appropriate.
func connError(err error) error {
	// Context cancellation is not a server-unavailable error.
	if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
		return fmt.Errorf("send request: %w", err)
	}

	return fmt.Errorf("%w: %v", ErrServerUnavailable, err)
}

// wrapStreamWithCleanup wraps a stream event channel, ensuring the HTTP response
// body is closed when the source channel is drained.
func wrapStreamWithCleanup(events <-chan StreamEvent, body io.ReadCloser) <-chan StreamEvent {
	out := make(chan StreamEvent)
	go func() {
		defer func() { _ = body.Close() }()
		defer close(out)
		for ev := range events {
			out <- ev
		}
	}()

	return out
}
