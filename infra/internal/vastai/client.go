package vastai

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"sort"
	"strings"
	"time"
	"unicode"
)

const baseURL = "https://console.vast.ai/api/v0"

// Client is a typed HTTP client for the vast.ai REST API.
type Client struct {
	apiKey     string
	httpClient *http.Client
}

// NewClient creates a new vast.ai API client.
func NewClient(apiKey string) *Client {
	return &Client{
		apiKey:     apiKey,
		httpClient: &http.Client{Timeout: 30 * time.Second},
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

// SearchOffers searches for available GPU offers matching the query.
// Returns offers sorted by cost (cheapest first).
func (c *Client) SearchOffers(ctx context.Context, query SearchQuery) ([]Offer, error) {
	q := map[string]any{
		"verified":    map[string]any{"eq": true},
		"rentable":    map[string]any{"eq": true},
		"rented":      map[string]any{"eq": false},
		"gpu_ram":     map[string]any{"gte": query.MinGPURAM * 1024}, // API uses MB
		"dph_total":   map[string]any{"lte": query.MaxCostPerHr},
		"disk_space":  map[string]any{"gte": query.MinDiskGB},
		"reliability": map[string]any{"gte": 0.95},
		"order":       []any{[]any{"dph_total", "asc"}},
		"type":        "ondemand",
		"limit":       100,
	}

	var result struct {
		Offers []Offer `json:"offers"`
	}
	if err := c.post(ctx, "/bundles/", q, &result); err != nil {
		return nil, err
	}

	offers := result.Offers

	// Client-side GPU name filter (supports partial match like "A100", "4090")
	if query.GPUName != "" {
		needle := normalizeGPUName(query.GPUName)
		filtered := offers[:0]
		for _, o := range offers {
			if strings.Contains(normalizeGPUName(o.GPUName), needle) {
				filtered = append(filtered, o)
			}
		}
		offers = filtered
	}

	sort.Slice(offers, func(i, j int) bool {
		return offers[i].CostPerHr < offers[j].CostPerHr
	})

	return offers, nil
}

// normalizeGPUName lowercases and strips spaces/underscores for fuzzy matching.
func normalizeGPUName(s string) string {
	return strings.Map(func(r rune) rune {
		if r == '_' || r == '-' || unicode.IsSpace(r) {
			return -1
		}

		return unicode.ToLower(r)
	}, s)
}

// CreateInstance creates a new instance from an offer.
func (c *Client) CreateInstance(ctx context.Context, offerID int, opts CreateOpts) (*Instance, error) {
	image := opts.Image
	if image == "" {
		image = "nvidia/cuda:12.4.1-devel-ubuntu22.04"
	}
	disk := opts.DiskGB
	if disk == 0 {
		disk = 50
	}

	body := map[string]any{
		"client_id": "me",
		"image":     image,
		"disk":      disk,
		"runtype":   "ssh ssh_direc ssh_proxy",
	}
	if opts.OnStart != "" {
		body["onstart"] = opts.OnStart
	}

	bodyJSON, err := json.Marshal(body)
	if err != nil {
		return nil, fmt.Errorf("marshal create body: %w", err)
	}

	if err := c.put(ctx, fmt.Sprintf("/asks/%d/", offerID), string(bodyJSON)); err != nil {
		return nil, err
	}

	// Poll for the new instance to appear
	for i := 0; i < 10; i++ {
		time.Sleep(3 * time.Second)
		instances, err := c.ListInstances(ctx)
		if err != nil {
			continue
		}
		for _, inst := range instances {
			if inst.Status == "running" || inst.Status == "loading" {
				return &inst, nil
			}
		}
	}

	return nil, fmt.Errorf("instance created but not found after polling")
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
	defer func() { _ = resp.Body.Close() }()

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
	req.Body = io.NopCloser(strings.NewReader(body))

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("vast.ai request failed: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		respBody, _ := io.ReadAll(resp.Body)

		return fmt.Errorf("vast.ai returned %d: %s", resp.StatusCode, string(respBody))
	}

	return nil
}

func (c *Client) post(ctx context.Context, path string, body any, result any) error {
	bodyJSON, err := json.Marshal(body)
	if err != nil {
		return fmt.Errorf("marshal request body: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, baseURL+path, bytes.NewReader(bodyJSON))
	if err != nil {
		return err
	}
	req.Header.Set("Authorization", "Bearer "+c.apiKey)
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("vast.ai request failed: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		respBody, _ := io.ReadAll(resp.Body)

		return fmt.Errorf("vast.ai returned %d: %s", resp.StatusCode, string(respBody))
	}

	return json.NewDecoder(resp.Body).Decode(result)
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
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		respBody, _ := io.ReadAll(resp.Body)

		return fmt.Errorf("vast.ai returned %d: %s", resp.StatusCode, string(respBody))
	}

	return nil
}
