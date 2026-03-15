package network

import "context"

// NoneProvider is a no-op network provider that uses the instance's direct IP.
type NoneProvider struct{}

func NewNoneProvider() *NoneProvider {
	return &NoneProvider{}
}

func (n *NoneProvider) Name() string { return "none" }

func (n *NoneProvider) GenerateAuthKey(ctx context.Context) (string, error) {
	return "", nil
}

func (n *NoneProvider) InstallCommands(authKey, hostname string) []string {
	return nil
}

func (n *NoneProvider) WaitForPeer(ctx context.Context, hostname string) (string, error) {
	// No tunnel — the deployer will use the SSH host IP directly.
	return "", nil
}
