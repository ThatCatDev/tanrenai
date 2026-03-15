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

// TailscaleProvider manages managed Tailscale network setup.
type TailscaleProvider struct {
	authKey    string // pre-generated auth key for Tailscale
	httpClient *http.Client
}

func NewTailscaleProvider(authKey string) *TailscaleProvider {
	return &TailscaleProvider{
		authKey:    authKey,
		httpClient: &http.Client{Timeout: 30 * time.Second},
	}
}

func (t *TailscaleProvider) Name() string { return "tailscale" }

// GenerateAuthKey returns the pre-configured auth key.
// For managed Tailscale, keys are generated via the admin console or API externally.
func (t *TailscaleProvider) GenerateAuthKey(ctx context.Context) (string, error) {
	if t.authKey == "" {
		return "", fmt.Errorf("tailscale auth key not configured (set TAILSCALE_AUTH_KEY)")
	}
	return t.authKey, nil
}

// InstallCommands returns shell commands to install Tailscale and join the network.
func (t *TailscaleProvider) InstallCommands(authKey, hostname string) []string {
	return []string{
		"curl -fsSL https://tailscale.com/install.sh | sh",
		fmt.Sprintf("tailscale up --authkey %s --hostname %s", authKey, hostname),
	}
}

// WaitForPeer polls the Tailscale API until the device with the given hostname appears.
func (t *TailscaleProvider) WaitForPeer(ctx context.Context, hostname string) (string, error) {
	ticker := time.NewTicker(5 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return "", ctx.Err()
		case <-ticker.C:
			ip, err := t.findDevice(ctx, hostname)
			if err != nil {
				continue
			}
			if ip != "" {
				return ip, nil
			}
		}
	}
}

func (t *TailscaleProvider) findDevice(ctx context.Context, hostname string) (string, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet,
		"https://api.tailscale.com/api/v2/tailnet/-/devices", nil)
	if err != nil {
		return "", err
	}
	req.Header.Set("Authorization", "Bearer "+t.authKey)

	resp, err := t.httpClient.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return "", fmt.Errorf("tailscale API returned %d: %s", resp.StatusCode, string(body))
	}

	var result struct {
		Devices []struct {
			Hostname  string   `json:"hostname"`
			Addresses []string `json:"addresses"`
		} `json:"devices"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return "", err
	}

	for _, dev := range result.Devices {
		if strings.EqualFold(dev.Hostname, hostname) && len(dev.Addresses) > 0 {
			return dev.Addresses[0], nil
		}
	}

	return "", nil
}
