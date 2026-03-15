package network

import (
	"context"
	"strings"
	"testing"
)

func TestNoneProvider(t *testing.T) {
	p := NewNoneProvider()

	if p.Name() != "none" {
		t.Errorf("Name() = %q, want \"none\"", p.Name())
	}

	key, err := p.GenerateAuthKey(context.Background())
	if err != nil {
		t.Fatalf("GenerateAuthKey() error: %v", err)
	}
	if key != "" {
		t.Errorf("GenerateAuthKey() = %q, want empty", key)
	}

	cmds := p.InstallCommands("", "test-host")
	if cmds != nil {
		t.Errorf("InstallCommands() = %v, want nil", cmds)
	}

	ip, err := p.WaitForPeer(context.Background(), "test-host")
	if err != nil {
		t.Fatalf("WaitForPeer() error: %v", err)
	}
	if ip != "" {
		t.Errorf("WaitForPeer() = %q, want empty", ip)
	}
}

func TestHeadscaleProviderInstallCommands(t *testing.T) {
	p := NewHeadscaleProvider("https://hs.example.com", "api-key", "")

	if p.Name() != "headscale" {
		t.Errorf("Name() = %q, want \"headscale\"", p.Name())
	}

	cmds := p.InstallCommands("auth-key-123", "gpu-node")
	if len(cmds) != 2 {
		t.Fatalf("InstallCommands() returned %d commands, want 2", len(cmds))
	}
	if !strings.Contains(cmds[0], "tailscale.com/install.sh") {
		t.Errorf("cmd[0] should install tailscale, got %q", cmds[0])
	}
	if !strings.Contains(cmds[1], "--login-server https://hs.example.com") {
		t.Errorf("cmd[1] should contain login-server, got %q", cmds[1][:80])
	}
	if !strings.Contains(cmds[1], "--authkey auth-key-123") {
		t.Errorf("cmd[1] should contain authkey")
	}
	if !strings.Contains(cmds[1], "--hostname gpu-node") {
		t.Errorf("cmd[1] should contain hostname")
	}
}

func TestTailscaleProviderInstallCommands(t *testing.T) {
	p := NewTailscaleProvider("tskey-123")

	if p.Name() != "tailscale" {
		t.Errorf("Name() = %q, want \"tailscale\"", p.Name())
	}

	cmds := p.InstallCommands("tskey-123", "gpu-node")
	if len(cmds) != 2 {
		t.Fatalf("InstallCommands() returned %d commands, want 2", len(cmds))
	}
	if !strings.Contains(cmds[1], "--authkey tskey-123") {
		t.Errorf("cmd[1] should contain authkey")
	}
	if !strings.Contains(cmds[1], "--hostname gpu-node") {
		t.Errorf("cmd[1] should contain hostname")
	}
}

func TestTailscaleProviderNoKey(t *testing.T) {
	p := NewTailscaleProvider("")

	_, err := p.GenerateAuthKey(context.Background())
	if err == nil {
		t.Error("expected error when auth key is empty")
	}
}
