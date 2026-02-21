package config

// Config holds the server configuration.
type Config struct {
	Host      string
	Port      int
	ModelsDir string
	BinDir    string
	GPULayers int
	CtxSize   int
}

// DefaultConfig returns a Config with sensible defaults.
func DefaultConfig() *Config {
	return &Config{
		Host:      "127.0.0.1",
		Port:      11435,
		ModelsDir: ModelsDir(),
		BinDir:    BinDir(),
		GPULayers: -1, // auto
		CtxSize:   4096,
	}
}
