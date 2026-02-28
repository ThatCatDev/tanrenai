package runner

import "time"

// Options configures how a runner loads and serves a model.
type Options struct {
	// Port for the llama-server subprocess to listen on.
	// 0 means auto-allocate a free port.
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

	// ChatTemplateFile is an optional path to a Jinja chat template file.
	// When set, llama-server uses this template instead of the GGUF-embedded one.
	ChatTemplateFile string

	// ReasoningFormat specifies the reasoning/thinking format for llama-server
	// (e.g. "deepseek" for Qwen3.5 thinking mode).
	ReasoningFormat string

	// Quiet suppresses subprocess stdout/stderr output.
	Quiet bool

	// HealthTimeout is how long to wait for the subprocess to become healthy.
	// 0 means use the default (120s for inference, 60s for embedding).
	HealthTimeout time.Duration
}

// DefaultOptions returns Options with sensible defaults.
// Port defaults to 0 (auto-allocate) to avoid conflicts when running
// multiple instances (e.g., serve + run, inference + embedding).
func DefaultOptions() Options {
	return Options{
		Port:           0,
		GPULayers:      -1,
		CtxSize:        4096,
		Threads:        0,
		FlashAttention: true,
	}
}
