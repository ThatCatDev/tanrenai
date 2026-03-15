package network

import "context"

// Provider abstracts network tunnel setup for connecting backend to GPU server.
type Provider interface {
	// Name returns the provider name (e.g. "headscale", "tailscale", "none").
	Name() string

	// GenerateAuthKey creates a pre-auth key for joining the network.
	GenerateAuthKey(ctx context.Context) (string, error)

	// InstallCommands returns shell commands to install and join the network on a remote host.
	InstallCommands(authKey, hostname string) []string

	// WaitForPeer polls until the given hostname appears in the network, returning its IP.
	WaitForPeer(ctx context.Context, hostname string) (string, error)
}
