package cmd

import (
	"context"

	"github.com/spf13/cobra"
)

// startLocalServersFn is the constructor used by resolveBackend when --local
// is set. Real callers use the default; tests swap it for a fake that
// returns a stub URL without booting real GPU/backend processes.
var startLocalServersFn = startLocalServers

// resolveBackend picks the backend URL a non-`run` command should talk to
// based on the persistent --local flag (defined on rootCmd). Returns a
// cleanup that callers must defer; it is a no-op in the non-local case so
// it's always safe to defer unconditionally.
//
// Centralizing the decision means every command that hits the backend
// honors --local consistently — earlier the list command silently fell
// through to the remote default, which is the bug this exists to prevent.
func resolveBackend(ctx context.Context, cmd *cobra.Command) (string, func(), error) {
	local, _ := cmd.Flags().GetBool("local")
	if !local {
		return serverURL, func() {}, nil
	}

	gpuLayers, _ := cmd.Flags().GetInt("gpu-layers")
	flashAttn, _ := cmd.Flags().GetBool("flash-attn")
	url, cleanup, err := startLocalServersFn(ctx, localOpts{
		GPULayers:      gpuLayers,
		FlashAttention: flashAttn,
	}, &startupLog{})
	if err != nil {
		return "", nil, err
	}
	return url, cleanup, nil
}
