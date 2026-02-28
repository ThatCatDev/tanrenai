package memory

import (
	"context"

	"github.com/ThatCatDev/tanrenai/server/internal/gpuclient"
)

// EmbedFunc is a function that produces a float32 embedding vector from text.
type EmbedFunc func(ctx context.Context, text string) ([]float32, error)

// NewRemoteEmbedFunc returns an EmbedFunc that calls the GPU server's /v1/embeddings
// endpoint via the gpuclient. This replaces the old NewLlamaEmbedFunc which spawned
// a local llama-server subprocess.
func NewRemoteEmbedFunc(gpu *gpuclient.Client) EmbedFunc {
	return func(ctx context.Context, text string) ([]float32, error) {
		return gpu.Embed(ctx, text)
	}
}
