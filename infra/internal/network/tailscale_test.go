package network

import (
	"context"
	"strings"
	"testing"
)

func TestTailscaleProviderName(t *testing.T) {
	p := NewTailscaleProvider("tskey-123")
	if p.Name() != "tailscale" {
		t.Errorf("Name() = %q, want \"tailscale\"", p.Name())
	}
}

func TestTailscaleProviderGenerateAuthKeyWithKey(t *testing.T) {
	p := NewTailscaleProvider("tskey-auth-abc123")
	key, err := p.GenerateAuthKey(context.Background())
	if err != nil {
		t.Fatalf("GenerateAuthKey() unexpected error: %v", err)
	}
	if key != "tskey-auth-abc123" {
		t.Errorf("GenerateAuthKey() = %q, want \"tskey-auth-abc123\"", key)
	}
}

func TestTailscaleProviderGenerateAuthKeyEmpty(t *testing.T) {
	p := NewTailscaleProvider("")
	_, err := p.GenerateAuthKey(context.Background())
	if err == nil {
		t.Error("expected error when auth key is empty")
	}
	if !strings.Contains(err.Error(), "TAILSCALE_AUTH_KEY") {
		t.Errorf("error should mention TAILSCALE_AUTH_KEY, got: %v", err)
	}
}

func TestTailscaleInstallCommandsCount(t *testing.T) {
	p := NewTailscaleProvider("tskey-123")
	cmds := p.InstallCommands("tskey-abc", "gpu-node")
	if len(cmds) != 2 {
		t.Fatalf("InstallCommands() returned %d commands, want 2", len(cmds))
	}
}

func TestTailscaleInstallCommandsInstallScript(t *testing.T) {
	p := NewTailscaleProvider("tskey-123")
	cmds := p.InstallCommands("tskey-abc", "gpu-node")

	if !strings.Contains(cmds[0], "tailscale.com/install.sh") {
		t.Errorf("cmd[0] should install tailscale, got: %q", cmds[0])
	}
	if !strings.Contains(cmds[0], "which tailscale") {
		t.Errorf("cmd[0] should check for existing tailscale, got: %q", cmds[0])
	}
}

func TestTailscaleInstallCommandsAuthKeyAndHostname(t *testing.T) {
	p := NewTailscaleProvider("tskey-123")
	cmds := p.InstallCommands("tskey-xyz", "my-hostname")

	if !strings.Contains(cmds[1], "--authkey tskey-xyz") {
		t.Errorf("cmd[1] should contain --authkey tskey-xyz, got: %q", cmds[1])
	}
	if !strings.Contains(cmds[1], "--hostname my-hostname") {
		t.Errorf("cmd[1] should contain --hostname my-hostname, got: %q", cmds[1])
	}
}

func TestTailscaleInstallCommandsDaemon(t *testing.T) {
	p := NewTailscaleProvider("tskey-123")
	cmds := p.InstallCommands("tskey-abc", "node")

	if !strings.Contains(cmds[1], "tailscaled") {
		t.Errorf("cmd[1] should start tailscaled daemon")
	}
	if !strings.Contains(cmds[1], "userspace-networking") {
		t.Errorf("cmd[1] should use userspace-networking")
	}
}

func TestTailscaleInstallCommandsNoLoginServer(t *testing.T) {
	// Tailscale (managed) should NOT have --login-server flag
	p := NewTailscaleProvider("tskey-123")
	cmds := p.InstallCommands("tskey-abc", "node")

	if strings.Contains(cmds[1], "--login-server") {
		t.Errorf("managed Tailscale cmd should not have --login-server, got: %q", cmds[1])
	}
}

func TestTailscaleInstallCommandsDifferentFromHeadscale(t *testing.T) {
	ts := NewTailscaleProvider("tskey-123")
	hs := NewHeadscaleProvider("https://hs.example.com", "key", "user")

	tsCmds := ts.InstallCommands("tskey-abc", "node")
	hsCmds := hs.InstallCommands("key-abc", "node")

	// Headscale should have --login-server, Tailscale should not
	if strings.Contains(tsCmds[1], "--login-server") {
		t.Error("Tailscale should not have --login-server")
	}
	if !strings.Contains(hsCmds[1], "--login-server") {
		t.Error("Headscale should have --login-server")
	}
}
