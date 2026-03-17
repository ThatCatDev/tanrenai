package deploy

import (
	"bytes"
	"context"
	"encoding/json"
	"net"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/ThatCatDev/tanrenai/infra/internal/config"
	"github.com/ThatCatDev/tanrenai/infra/internal/network"
	"github.com/ThatCatDev/tanrenai/infra/internal/vastai"
)

// TestDeployRunVerbose tests Run with verbose=true to cover the verbose output path.
func TestDeployRunVerbose(t *testing.T) {
	_, sshHost, sshPort := startDeploySSHServer(t)

	healthSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/health" {
			w.WriteHeader(http.StatusOK)
		}
	}))
	defer healthSrv.Close()

	healthPort := 0
	_, portStr, _ := net.SplitHostPort(healthSrv.Listener.Addr().String())
	for _, c := range portStr {
		healthPort = healthPort*10 + int(c-'0')
	}

	inst := vastai.Instance{
		ID:        556,
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
		VastaiInstance: "556",
		Network:        "none",
		GPUPort:        healthPort,
		Model:          "",
		MinGPURAM:      24,
		MaxCostPerHr:   1.0,
		DiskGB:         50,
	}

	var buf bytes.Buffer
	// verbose=true exercises the different branch in stage execution
	d := New(client, network.NewNoneProvider(), cfg, &buf, true)

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	result, err := d.Run(ctx)
	if err != nil {
		t.Fatalf("Run(verbose=true) error: %v\nOutput:\n%s", err, buf.String())
	}
	if result == nil {
		t.Fatal("Run(verbose=true) returned nil result")
	}
	if result.InstanceID != 556 {
		t.Errorf("result.InstanceID = %d, want 556", result.InstanceID)
	}
}

// mockNetworkProvider is a test network provider that returns a fixed IP for WaitForPeer.
type mockNetworkProvider struct {
	name      string
	authKey   string
	peerIP    string
	waitDelay time.Duration
}

func (m *mockNetworkProvider) Name() string { return m.name }

func (m *mockNetworkProvider) GenerateAuthKey(_ context.Context) (string, error) {
	return m.authKey, nil
}

func (m *mockNetworkProvider) InstallCommands(_, _ string) []string {
	return []string{"echo network-setup"}
}

func (m *mockNetworkProvider) WaitForPeer(_ context.Context, _ string) (string, error) {
	if m.waitDelay > 0 {
		time.Sleep(m.waitDelay)
	}

	return m.peerIP, nil
}

// TestDeployRunWithNetwork tests Run with a non-none network provider to cover
// the auth key generation and WaitForPeer branches.
func TestDeployRunWithNetwork(t *testing.T) {
	_, sshHost, sshPort := startDeploySSHServer(t)

	// Health check server at a specific port — WaitForPeer returns sshHost
	healthSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/health" {
			w.WriteHeader(http.StatusOK)
		}
	}))
	defer healthSrv.Close()

	healthPort := 0
	_, portStr, _ := net.SplitHostPort(healthSrv.Listener.Addr().String())
	for _, c := range portStr {
		healthPort = healthPort*10 + int(c-'0')
	}

	inst := vastai.Instance{
		ID:        557,
		Status:    "running",
		GPUName:   "H100",
		CostPerHr: 2.0,
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
		VastaiInstance: "557",
		Network:        "headscale", // non-none network
		GPUPort:        healthPort,
		Model:          "",
		MinGPURAM:      80,
		MaxCostPerHr:   3.0,
		DiskGB:         200,
	}

	// Use a mock network that returns immediately
	// WaitForPeer returns the health server's host so the health check works
	mockNet := &mockNetworkProvider{
		name:    "headscale",
		authKey: "test-auth-key",
		peerIP:  "127.0.0.1",
	}

	var buf bytes.Buffer
	d := New(client, mockNet, cfg, &buf, false)

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	result, err := d.Run(ctx)
	if err != nil {
		t.Fatalf("Run(network=headscale) error: %v\nOutput:\n%s", err, buf.String())
	}
	if result == nil {
		t.Fatal("Run(network=headscale) returned nil result")
	}
	if result.InstanceID != 557 {
		t.Errorf("result.InstanceID = %d, want 557", result.InstanceID)
	}
}

// TestDeployRunInstanceWaitsForReady tests the polling path where the instance
// starts as "loading" and becomes "running" with SSH details on the second poll.
func TestDeployRunInstanceWaitsForReady(t *testing.T) {
	_, sshHost, sshPort := startDeploySSHServer(t)

	healthSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	}))
	defer healthSrv.Close()

	healthPort := 0
	_, portStr, _ := net.SplitHostPort(healthSrv.Listener.Addr().String())
	for _, c := range portStr {
		healthPort = healthPort*10 + int(c-'0')
	}

	callCount := 0
	// First poll: instance is running but has no SSH host yet
	loadingInst := vastai.Instance{
		ID:      558,
		Status:  "running",
		GPUName: "RTX 4090",
	}
	// Second poll: instance has SSH details
	readyInst := vastai.Instance{
		ID:        558,
		Status:    "running",
		GPUName:   "RTX 4090",
		CostPerHr: 0.5,
		SSHHost:   sshHost,
		SSHPort:   sshPort,
	}

	vastaiSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		callCount++
		if callCount == 1 {
			// First call: return loading instance
			json.NewEncoder(w).Encode(map[string]any{"instances": []vastai.Instance{loadingInst}})
		} else {
			// Subsequent calls: return ready instance
			json.NewEncoder(w).Encode(map[string]any{"instances": []vastai.Instance{readyInst}})
		}
	}))
	defer vastaiSrv.Close()

	client := vastai.NewClient("test-key")
	overrideVastaiClientTransport(client, vastaiSrv)

	cfg := config.Config{
		VastaiAPIKey:   "test-key",
		VastaiInstance: "558",
		Network:        "none",
		GPUPort:        healthPort,
		Model:          "",
		MinGPURAM:      24,
		MaxCostPerHr:   1.0,
		DiskGB:         50,
	}

	var buf bytes.Buffer
	d := New(client, network.NewNoneProvider(), cfg, &buf, false)

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	result, err := d.Run(ctx)
	if err != nil {
		t.Fatalf("Run() with waiting instance error: %v\nOutput:\n%s", err, buf.String())
	}
	if result == nil {
		t.Fatal("Run() returned nil result")
	}
	if result.InstanceID != 558 {
		t.Errorf("result.InstanceID = %d, want 558", result.InstanceID)
	}
}
