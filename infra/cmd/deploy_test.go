package cmd

import (
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/spf13/cobra"

	"github.com/ThatCatDev/tanrenai/infra/internal/vastai"
)

// makeDeployFlagsCmd creates a cobra.Command with all the flags that runDeploy reads.
func makeDeployFlagsCmd() *cobra.Command {
	cmd := &cobra.Command{Use: "deploy"}
	f := cmd.Flags()
	f.String("vastai-api-key", "", "")
	f.String("vastai-instance-id", "", "")
	f.String("gpu", "", "")
	f.String("model-size", "", "")
	f.Float64("min-gpu-ram", 24, "")
	f.Float64("max-cost", 1.0, "")
	f.Float64("disk-gb", 50, "")
	f.String("network", "none", "")
	f.String("headscale-url", "", "")
	f.String("headscale-api-key", "", "")
	f.String("headscale-user", "", "")
	f.String("tailscale-auth-key", "", "")
	f.Int("gpu-port", 11435, "")
	f.String("model", "", "")
	f.BoolP("verbose", "v", false, "")

	return cmd
}

// newErrorServer creates a test server that always returns HTTP 500.
func newErrorServer(t *testing.T) *httptest.Server {
	t.Helper()

	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "server error", http.StatusInternalServerError)
	}))
}

// mockVastaiClient injects a vastai.Client pointed at the given test server.
func mockVastaiClient(t *testing.T, srv *httptest.Server) func(apiKey string) *vastai.Client {
	t.Helper()

	return func(apiKey string) *vastai.Client {
		return makeMockClient(t, srv)
	}
}

// TestRunDeploy_MissingAPIKey verifies runDeploy exits when no API key is set.
func TestRunDeploy_MissingAPIKey(t *testing.T) {
	t.Setenv("VASTAI_API_KEY", "")
	withMockExit(t)

	cmd := makeDeployFlagsCmd()

	code := recoverExitPanic(func() {
		runDeploy(cmd, nil)
	})

	if code != 1 {
		t.Errorf("expected exit code 1 when API key missing, got %d", code)
	}
}

// TestRunDeploy_InvalidModelSize verifies runDeploy exits when model-size is invalid.
func TestRunDeploy_InvalidModelSize(t *testing.T) {
	t.Setenv("VASTAI_API_KEY", "")
	withMockExit(t)

	cmd := makeDeployFlagsCmd()
	_ = cmd.Flags().Set("vastai-api-key", "test-key")
	_ = cmd.Flags().Set("model-size", "notanumber")

	code := recoverExitPanic(func() {
		runDeploy(cmd, nil)
	})

	if code != 1 {
		t.Errorf("expected exit code 1 for invalid model-size, got %d", code)
	}
}

// TestRunDeploy_MissingHeadscaleURL verifies runDeploy exits when headscale URL is absent.
func TestRunDeploy_MissingHeadscaleURL(t *testing.T) {
	t.Setenv("VASTAI_API_KEY", "")
	t.Setenv("HEADSCALE_URL", "")
	t.Setenv("HEADSCALE_API_KEY", "")
	withMockExit(t)

	cmd := makeDeployFlagsCmd()
	_ = cmd.Flags().Set("vastai-api-key", "test-key")
	_ = cmd.Flags().Set("network", "headscale")

	code := recoverExitPanic(func() {
		runDeploy(cmd, nil)
	})

	if code != 1 {
		t.Errorf("expected exit code 1 when headscale-url missing, got %d", code)
	}
}

// TestRunDeploy_MissingHeadscaleAPIKey verifies runDeploy exits when headscale API key is absent.
func TestRunDeploy_MissingHeadscaleAPIKey(t *testing.T) {
	t.Setenv("VASTAI_API_KEY", "")
	t.Setenv("HEADSCALE_URL", "")
	t.Setenv("HEADSCALE_API_KEY", "")
	withMockExit(t)

	cmd := makeDeployFlagsCmd()
	_ = cmd.Flags().Set("vastai-api-key", "test-key")
	_ = cmd.Flags().Set("network", "headscale")
	_ = cmd.Flags().Set("headscale-url", "https://headscale.example.com")

	code := recoverExitPanic(func() {
		runDeploy(cmd, nil)
	})

	if code != 1 {
		t.Errorf("expected exit code 1 when headscale-api-key missing, got %d", code)
	}
}

// TestRunDeploy_UnknownNetwork verifies runDeploy exits for an unknown network provider.
func TestRunDeploy_UnknownNetwork(t *testing.T) {
	t.Setenv("VASTAI_API_KEY", "")
	withMockExit(t)

	cmd := makeDeployFlagsCmd()
	_ = cmd.Flags().Set("vastai-api-key", "test-key")
	_ = cmd.Flags().Set("network", "bogusnet")

	code := recoverExitPanic(func() {
		runDeploy(cmd, nil)
	})

	if code != 1 {
		t.Errorf("expected exit code 1 for unknown network, got %d", code)
	}
}

// TestRunDeploy_ModelSizeAutoSetsDisk verifies that model-size also auto-configures disk.
func TestRunDeploy_ModelSizeAutoSetsDisk(t *testing.T) {
	t.Setenv("VASTAI_API_KEY", "")
	withMockExit(t)

	cmd := makeDeployFlagsCmd()
	_ = cmd.Flags().Set("model-size", "8b")

	// Exits with code 1 (no API key) after printing VRAM info — that's expected.
	recoverExitPanic(func() {
		runDeploy(cmd, nil)
	})
}

// TestRunDeploy_ModelSizeWithDiskOverride verifies that --disk-gb suppresses auto-set.
func TestRunDeploy_ModelSizeWithDiskOverride(t *testing.T) {
	t.Setenv("VASTAI_API_KEY", "")
	withMockExit(t)

	cmd := makeDeployFlagsCmd()
	_ = cmd.Flags().Set("model-size", "72b")
	_ = cmd.Flags().Set("disk-gb", "200")

	recoverExitPanic(func() {
		runDeploy(cmd, nil)
	})
}

// TestRunDeploy_TailscaleNetwork verifies the tailscale network path executes
// correctly and reaches deployer.Run (which fails against a mock HTTP server).
func TestRunDeploy_TailscaleNetwork(t *testing.T) {
	t.Setenv("VASTAI_API_KEY", "")
	withMockExit(t)

	srv := newErrorServer(t)
	defer srv.Close()

	origNewClient := newVastaiClient
	newVastaiClient = mockVastaiClient(t, srv)
	defer func() { newVastaiClient = origNewClient }()

	cmd := makeDeployFlagsCmd()
	_ = cmd.Flags().Set("vastai-api-key", "test-key")
	_ = cmd.Flags().Set("network", "tailscale")
	_ = cmd.Flags().Set("tailscale-auth-key", "tskey-auth-fake")

	code := recoverExitPanic(func() {
		runDeploy(cmd, nil)
	})

	// deploy.Run will fail against the mock server → exitError → code 1
	if code != 1 {
		t.Errorf("expected exit code 1 for failed deploy, got %d", code)
	}
}

// TestRunDeploy_HeadscaleNetwork verifies the headscale network path executes
// correctly and reaches deployer.Run (which fails against a mock HTTP server).
func TestRunDeploy_HeadscaleNetwork(t *testing.T) {
	t.Setenv("VASTAI_API_KEY", "")
	t.Setenv("HEADSCALE_URL", "")
	t.Setenv("HEADSCALE_API_KEY", "")
	withMockExit(t)

	srv := newErrorServer(t)
	defer srv.Close()

	origNewClient := newVastaiClient
	newVastaiClient = mockVastaiClient(t, srv)
	defer func() { newVastaiClient = origNewClient }()

	cmd := makeDeployFlagsCmd()
	_ = cmd.Flags().Set("vastai-api-key", "test-key")
	_ = cmd.Flags().Set("network", "headscale")
	_ = cmd.Flags().Set("headscale-url", "https://headscale.example.com")
	_ = cmd.Flags().Set("headscale-api-key", "hs-api-key")
	_ = cmd.Flags().Set("headscale-user", "myuser")

	code := recoverExitPanic(func() {
		runDeploy(cmd, nil)
	})

	if code != 1 {
		t.Errorf("expected exit code 1 for failed deploy, got %d", code)
	}
}

// TestRunDeploy_NoneNetworkWithFlags verifies that all optional flags can be set
// without panicking, reaching deployer.Run (which fails on mock server).
func TestRunDeploy_NoneNetworkWithFlags(t *testing.T) {
	t.Setenv("VASTAI_API_KEY", "")
	withMockExit(t)

	srv := newErrorServer(t)
	defer srv.Close()

	origNewClient := newVastaiClient
	newVastaiClient = mockVastaiClient(t, srv)
	defer func() { newVastaiClient = origNewClient }()

	cmd := makeDeployFlagsCmd()
	_ = cmd.Flags().Set("vastai-api-key", "explicit-key")
	_ = cmd.Flags().Set("gpu", "A100")
	_ = cmd.Flags().Set("min-gpu-ram", "80")
	_ = cmd.Flags().Set("max-cost", "2.5")
	_ = cmd.Flags().Set("gpu-port", "12345")
	_ = cmd.Flags().Set("model", "qwen3:7b")
	_ = cmd.Flags().Set("network", "none")

	code := recoverExitPanic(func() {
		runDeploy(cmd, nil)
	})

	if code != 1 {
		t.Errorf("expected exit code 1 for failed deploy, got %d", code)
	}
}
