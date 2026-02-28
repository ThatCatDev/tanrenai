package vastai

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
)

const baseURL = "https://console.vast.ai/api/v0"

// Instance represents a vast.ai instance.
type Instance struct {
	ID         int     `json:"id"`
	Status     string  `json:"actual_status"` // running, exited, loading, etc.
	SSHHost    string  `json:"ssh_host"`
	SSHPort    int     `json:"ssh_port"`
	CurState   string  `json:"cur_state"`
	GPUName    string  `json:"gpu_name"`
	NumGPUs    int     `json:"num_gpus"`
	CostPerHr  float64 `json:"dph_total"`
}

// Client is a typed HTTP client for the vast.ai REST API.
type Client struct {
	apiKey     string
	httpClient *http.Client
}

// NewClient creates a new vast.ai API client.
func NewClient(apiKey string) *Client {
	return &Client{
		apiKey:     apiKey,
		httpClient: &http.Client{},
	}
}

// ListInstances returns all instances for the account.
func (c *Client) ListInstances(ctx context.Context) ([]Instance, error) {
	var result struct {
		Instances []Instance `json:"instances"`
	}
	if err := c.get(ctx, "/instances", &result); err != nil {
		return nil, err
	}
	return result.Instances, nil
}

// GetInstance returns a single instance by ID.
func (c *Client) GetInstance(ctx context.Context, id string) (*Instance, error) {
	var result struct {
		Instances []Instance `json:"instances"`
	}
	if err := c.get(ctx, "/instances?id="+id, &result); err != nil {
		return nil, err
	}
	for _, inst := range result.Instances {
		if fmt.Sprintf("%d", inst.ID) == id {
			return &inst, nil
		}
	}
	return nil, fmt.Errorf("instance %s not found", id)
}

// StartInstance starts a stopped instance.
func (c *Client) StartInstance(ctx context.Context, id string) error {
	return c.put(ctx, "/instances/"+id+"/", `{"state": "running"}`)
}

// StopInstance stops a running instance.
func (c *Client) StopInstance(ctx context.Context, id string) error {
	return c.put(ctx, "/instances/"+id+"/", `{"state": "stopped"}`)
}

// DestroyInstance permanently destroys an instance.
func (c *Client) DestroyInstance(ctx context.Context, id string) error {
	return c.delete(ctx, "/instances/"+id+"/")
}

func (c *Client) get(ctx context.Context, path string, result any) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, baseURL+path, nil)
	if err != nil {
		return err
	}
	req.Header.Set("Authorization", "Bearer "+c.apiKey)

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("vast.ai request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("vast.ai returned %d: %s", resp.StatusCode, string(body))
	}

	return json.NewDecoder(resp.Body).Decode(result)
}

func (c *Client) put(ctx context.Context, path string, body string) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodPut, baseURL+path, nil)
	if err != nil {
		return err
	}
	req.Header.Set("Authorization", "Bearer "+c.apiKey)
	req.Header.Set("Content-Type", "application/json")
	req.Body = io.NopCloser(io.Reader(jsonReader(body)))

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("vast.ai request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		respBody, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("vast.ai returned %d: %s", resp.StatusCode, string(respBody))
	}

	return nil
}

func (c *Client) delete(ctx context.Context, path string) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodDelete, baseURL+path, nil)
	if err != nil {
		return err
	}
	req.Header.Set("Authorization", "Bearer "+c.apiKey)

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("vast.ai request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		respBody, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("vast.ai returned %d: %s", resp.StatusCode, string(respBody))
	}

	return nil
}

type stringReader struct {
	s string
	i int
}

func (r *stringReader) Read(p []byte) (n int, err error) {
	if r.i >= len(r.s) {
		return 0, io.EOF
	}
	n = copy(p, r.s[r.i:])
	r.i += n
	return n, nil
}

func jsonReader(s string) io.Reader {
	return &stringReader{s: s}
}
