package deploy

import (
	"bytes"
	"context"
	"encoding/json"
	"net"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/ThatCatDev/tanrenai/infra/internal/config"
	"github.com/ThatCatDev/tanrenai/infra/internal/network"
	"github.com/ThatCatDev/tanrenai/infra/internal/vastai"
)

// TestDeployRunHealthCheckFails verifies Run succeeds with a warning when health check fails.
// The health server returns 503, so health check times out (30s internal timeout),
// but Run still returns successfully (just with a warning).
func TestDeployRunHealthCheckFails(t *testing.T) {
	_, sshHost, sshPort := startDeploySSHServer(t)

	// Health server always returns 503
	healthSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusServiceUnavailable)
	}))
	defer healthSrv.Close()

	healthPort := 0
	_, portStr, _ := net.SplitHostPort(healthSrv.Listener.Addr().String())
	for _, c := range portStr {
		healthPort = healthPort*10 + int(c-'0')
	}

	inst := vastai.Instance{
		ID:        559,
		Status:    "running",
		GPUName:   "RTX 3090",
		CostPerHr: 0.3,
		SSHHost:   sshHost,
		SSHPort:   sshPort,
	}

	vastaiSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"instances": []vastai.Instance{inst},
		})
	}))
	defer vastaiSrv.Close()

	client := vastai.NewClient("test-key")
	overrideVastaiClientTransport(client, vastaiSrv)

	cfg := config.Config{
		VastaiAPIKey:   "test-key",
		VastaiInstance: "559",
		Network:        "none",
		GPUPort:        healthPort,
		Model:          "",
		MinGPURAM:      24,
		MaxCostPerHr:   1.0,
		DiskGB:         50,
	}

	var buf bytes.Buffer
	d := New(client, network.NewNoneProvider(), cfg, &buf, false)

	// Use a short context to avoid waiting 30s for health check to time out
	ctx, cancel := context.WithTimeout(context.Background(), 35*time.Second)
	defer cancel()

	// Run should succeed even when health check fails (it only warns)
	result, err := d.Run(ctx)
	if err != nil {
		t.Fatalf("Run() should succeed even when health check fails, got error: %v", err)
	}
	if result == nil {
		t.Fatal("Run() returned nil result")
	}
	if result.InstanceID != 559 {
		t.Errorf("result.InstanceID = %d, want 559", result.InstanceID)
	}
}

// TestDeployRunWaitForPeerFails verifies Run returns error when WaitForPeer fails.
// We use a mock network provider whose WaitForPeer returns an error.
func TestDeployRunWaitForPeerFails(t *testing.T) {
	_, sshHost, sshPort := startDeploySSHServer(t)

	inst := vastai.Instance{
		ID:        560,
		Status:    "running",
		GPUName:   "A100",
		CostPerHr: 1.0,
		SSHHost:   sshHost,
		SSHPort:   sshPort,
	}

	vastaiSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"instances": []vastai.Instance{inst},
		})
	}))
	defer vastaiSrv.Close()

	client := vastai.NewClient("test-key")
	overrideVastaiClientTransport(client, vastaiSrv)

	cfg := config.Config{
		VastaiAPIKey:   "test-key",
		VastaiInstance: "560",
		Network:        "headscale",
		GPUPort:        11435,
		Model:          "",
		MinGPURAM:      80,
		MaxCostPerHr:   2.0,
		DiskGB:         100,
	}

	// Mock network that returns error from WaitForPeer
	mockNet := &errorWaitNetworkProvider{name: "headscale", authKey: "key"}

	var buf bytes.Buffer
	d := New(client, mockNet, cfg, &buf, false)

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	_, err := d.Run(ctx)
	if err == nil {
		t.Error("expected error when WaitForPeer fails")
	}
	if !strings.Contains(err.Error(), "wait for peer") {
		t.Errorf("error should mention wait for peer, got: %v", err)
	}
}

// TestDeployRunInstanceNotReadyTimeout verifies Run returns error when instance doesn't become ready.
func TestDeployRunInstanceNotReadyTimeout(t *testing.T) {
	// Instance never gets SSH host/port (stays as "running" with no SSH)
	inst := vastai.Instance{
		ID:      561,
		Status:  "running",
		GPUName: "RTX 4090",
		// No SSHHost or SSHPort
	}

	vastaiSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"instances": []vastai.Instance{inst},
		})
	}))
	defer vastaiSrv.Close()

	client := vastai.NewClient("test-key")
	overrideVastaiClientTransport(client, vastaiSrv)

	cfg := config.Config{
		VastaiAPIKey:   "test-key",
		VastaiInstance: "561",
		Network:        "none",
		GPUPort:        11435,
	}

	var buf bytes.Buffer
	d := New(client, network.NewNoneProvider(), cfg, &buf, false)

	// Short timeout so the test doesn't wait 5 minutes
	ctx, cancel := context.WithTimeout(context.Background(), 8*time.Second)
	defer cancel()

	_, err := d.Run(ctx)
	if err == nil {
		t.Error("expected error when instance never becomes ready with SSH details")
	}
}

// errorWaitNetworkProvider is a network provider whose WaitForPeer always returns an error.
type errorWaitNetworkProvider struct {
	name    string
	authKey string
}

func (e *errorWaitNetworkProvider) Name() string { return e.name }
func (e *errorWaitNetworkProvider) GenerateAuthKey(_ context.Context) (string, error) {
	return e.authKey, nil
}
func (e *errorWaitNetworkProvider) InstallCommands(_, _ string) []string {
	return []string{"echo setup"}
}
func (e *errorWaitNetworkProvider) WaitForPeer(_ context.Context, _ string) (string, error) {
	return "", context.DeadlineExceeded
}
