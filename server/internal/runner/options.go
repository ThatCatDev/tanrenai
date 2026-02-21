package runner

// Options configures how a runner loads and serves a model.
type Options struct {
	// Port for the llama-server subprocess to listen on.
	Port int

	// GPULayers is the number of layers to offload to GPU (-1 = auto/all).
	GPULayers int

	// CtxSize is the context window size in tokens.
	CtxSize int

	// BinDir is the directory containing llama-server binaries.
	BinDir string

	// Threads is the number of CPU threads to use (0 = auto).
	Threads int

	// FlashAttention enables flash attention if supported.
	FlashAttention bool
}

// DefaultOptions returns Options with sensible defaults.
func DefaultOptions() Options {
	return Options{
		Port:           18080,
		GPULayers:      -1,
		CtxSize:        4096,
		Threads:        0,
		FlashAttention: true,
	}
}
