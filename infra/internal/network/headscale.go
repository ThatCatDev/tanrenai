package network

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

// HeadscaleProvider manages Headscale network tunnel setup.
type HeadscaleProvider struct {
	baseURL    string // e.g. "https://headscale.floretos.com"
	apiKey     string
	user       string // headscale user name
	httpClient *http.Client
}

func NewHeadscaleProvider(baseURL, apiKey, user string) *HeadscaleProvider {
	if user == "" {
		user = "tanrenai"
	}

	return &HeadscaleProvider{
		baseURL:    strings.TrimRight(baseURL, "/"),
		apiKey:     apiKey,
		user:       user,
		httpClient: &http.Client{Timeout: 30 * time.Second},
	}
}

func (h *HeadscaleProvider) Name() string { return "headscale" }

// GenerateAuthKey creates a pre-auth key via the Headscale API.
// It looks up the user by name to get the numeric ID, then creates the key.
func (h *HeadscaleProvider) GenerateAuthKey(ctx context.Context) (string, error) {
	// Look up user ID by name
	userID, err := h.getUserID(ctx, h.user)
	if err != nil {
		return "", fmt.Errorf("look up headscale user %q: %w", h.user, err)
	}

	body := fmt.Sprintf(`{"user":"%d","reusable":false,"ephemeral":true,"expiration":"%s"}`,
		userID, time.Now().Add(24*time.Hour).Format(time.RFC3339))

	req, err := http.NewRequestWithContext(ctx, http.MethodPost,
		h.baseURL+"/api/v1/preauthkey", strings.NewReader(body))
	if err != nil {
		return "", err
	}
	req.Header.Set("Authorization", "Bearer "+h.apiKey)
	req.Header.Set("Content-Type", "application/json")

	resp, err := h.httpClient.Do(req)
	if err != nil {
		return "", fmt.Errorf("headscale API request failed: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		respBody, _ := io.ReadAll(resp.Body)

		return "", fmt.Errorf("headscale returned %d: %s", resp.StatusCode, string(respBody))
	}

	var result struct {
		PreAuthKey struct {
			Key string `json:"key"`
		} `json:"preAuthKey"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return "", fmt.Errorf("decode headscale response: %w", err)
	}

	return result.PreAuthKey.Key, nil
}

// getUserID looks up a Headscale user by name and returns the numeric ID.
func (h *HeadscaleProvider) getUserID(ctx context.Context, name string) (uint64, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet,
		h.baseURL+"/api/v1/user", nil)
	if err != nil {
		return 0, err
	}
	req.Header.Set("Authorization", "Bearer "+h.apiKey)

	resp, err := h.httpClient.Do(req)
	if err != nil {
		return 0, err
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		respBody, _ := io.ReadAll(resp.Body)

		return 0, fmt.Errorf("headscale returned %d: %s", resp.StatusCode, string(respBody))
	}

	var result struct {
		Users []struct {
			ID   uint64 `json:"id,string"`
			Name string `json:"name"`
		} `json:"users"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return 0, fmt.Errorf("decode users response: %w", err)
	}

	for _, u := range result.Users {
		if strings.EqualFold(u.Name, name) {
			return u.ID, nil
		}
	}

	return 0, fmt.Errorf("user %q not found in headscale (create it with: headscale users create %s)", name, name)
}

// InstallCommands returns shell commands to install Tailscale and join the Headscale server.
func (h *HeadscaleProvider) InstallCommands(authKey, hostname string) []string {
	return []string{
		"which tailscale >/dev/null 2>&1 || curl -fsSL https://tailscale.com/install.sh | sh",
		fmt.Sprintf(`bash -c '
set -e
mkdir -p /var/run/tailscale /var/lib/tailscale
killall tailscaled 2>/dev/null || true
sleep 1
rm -f /var/run/tailscale/tailscaled.sock
tailscaled --state=/var/lib/tailscale/tailscaled.state --socket=/var/run/tailscale/tailscaled.sock --tun=userspace-networking > /var/log/tailscaled.log 2>&1 &
TSPID=$!
for i in $(seq 1 15); do
  [ -S /var/run/tailscale/tailscaled.sock ] && break
  echo "waiting for tailscaled socket ($i)..."
  sleep 1
done
tailscale --socket=/var/run/tailscale/tailscaled.sock up --login-server %s --authkey %s --hostname %s
echo "tailscale connected: $(tailscale --socket=/var/run/tailscale/tailscaled.sock ip -4)"
'`, h.baseURL, authKey, hostname),
	}
}

// Node represents a machine in the Headscale network.
type Node struct {
	Name   string
	Online bool
	IPs    []string
}

// ListNodes returns all nodes in the Headscale network.
func (h *HeadscaleProvider) ListNodes(ctx context.Context) ([]Node, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet,
		h.baseURL+"/api/v1/node", nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("Authorization", "Bearer "+h.apiKey)

	resp, err := h.httpClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("headscale returned %d", resp.StatusCode)
	}

	var result struct {
		Nodes []struct {
			GivenName   string   `json:"givenName"`
			IPAddresses []string `json:"ipAddresses"`
			Online      bool     `json:"online"`
		} `json:"nodes"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, err
	}

	nodes := make([]Node, len(result.Nodes))
	for i, n := range result.Nodes {
		nodes[i] = Node{Name: n.GivenName, Online: n.Online, IPs: n.IPAddresses}
	}

	return nodes, nil
}

// WaitForPeer polls the Headscale API until a node with the given hostname appears.
func (h *HeadscaleProvider) WaitForPeer(ctx context.Context, hostname string) (string, error) {
	ticker := time.NewTicker(5 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return "", ctx.Err()
		case <-ticker.C:
			ip, err := h.findNode(ctx, hostname)
			if err != nil {
				continue
			}
			if ip != "" {
				return ip, nil
			}
		}
	}
}

func (h *HeadscaleProvider) findNode(ctx context.Context, hostname string) (string, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet,
		h.baseURL+"/api/v1/node", nil)
	if err != nil {
		return "", err
	}
	req.Header.Set("Authorization", "Bearer "+h.apiKey)

	resp, err := h.httpClient.Do(req)
	if err != nil {
		return "", err
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("headscale returned %d", resp.StatusCode)
	}

	var result struct {
		Nodes []struct {
			GivenName   string   `json:"givenName"`
			IPAddresses []string `json:"ipAddresses"`
			Online      bool     `json:"online"`
		} `json:"nodes"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return "", err
	}

	for _, node := range result.Nodes {
		if node.GivenName == hostname && node.Online && len(node.IPAddresses) > 0 {
			return node.IPAddresses[0], nil
		}
	}

	return "", nil
}
