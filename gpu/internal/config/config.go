package config

// Config holds the GPU server configuration.
type Config struct {
	Host             string
	Port             int
	ModelsDir        string
	BinDir           string
	GPULayers        int
	CtxSize          int
	ChatTemplateFile string // optional Jinja chat template override
	EmbeddingModel   string // optional embedding model name/path
	ReasoningFormat  string // optional reasoning format (e.g. "deepseek" for Qwen3.5 thinking mode)
	FlashAttention   bool   // enable flash attention (default true)
}

// DefaultConfig returns a Config with sensible defaults.
func DefaultConfig() *Config {
	return &Config{
		Host:      "127.0.0.1",
		Port:      11435,
		ModelsDir: ModelsDir(),
		BinDir:    BinDir(),
		GPULayers:      -1, // auto
		CtxSize:        4096,
		FlashAttention: true,
	}
}
