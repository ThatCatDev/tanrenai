package models

import "testing"

func TestResolveBareNameToURI(t *testing.T) {
	tests := []struct {
		name string
		want string
	}{
		{"Qwen3.5-122B-A10B-UD-Q4_K_XL", "hf://unsloth/Qwen3.5-122B-A10B-GGUF/UD-Q4_K_XL"},
		{"Qwen2.5-7B-Instruct-Q4_K_M", "hf://unsloth/Qwen2.5-7B-Instruct-GGUF/Q4_K_M"},
		{"Llama-3-8B-Instruct-Q5_K_S", "hf://unsloth/Llama-3-8B-Instruct-GGUF/Q5_K_S"},
		{"Mistral-7B-Instruct-v0.3-IQ2_XS", "hf://unsloth/Mistral-7B-Instruct-v0.3-GGUF/IQ2_XS"},
		{"gemma-2-27b-it-BF16", "hf://unsloth/gemma-2-27b-it-GGUF/BF16"},
		{"Qwen2.5-0.5B-UD-Q2_K_XL", "hf://unsloth/Qwen2.5-0.5B-GGUF/UD-Q2_K_XL"},
		{"Qwen3.6-35B-A3B-Q4_K_M", "hf://unsloth/Qwen3.6-35B-A3B-GGUF/Q4_K_M"},
		// URIs pass through unchanged.
		{"hf://unsloth/foo/Q4_K_M", "hf://unsloth/foo/Q4_K_M"},
		{"https://huggingface.co/x/y.gguf", "https://huggingface.co/x/y.gguf"},
		{"http://example.com/m.gguf", "http://example.com/m.gguf"},
		// No quant suffix — caller should error rather than misroute.
		{"Qwen2.5-7B-Instruct", ""},
		{"just-a-plain-name", ""},
		{"", ""},
	}
	for _, tc := range tests {
		got := ResolveBareNameToURI(tc.name)
		if got != tc.want {
			t.Errorf("ResolveBareNameToURI(%q) = %q, want %q", tc.name, got, tc.want)
		}
	}
}

func TestIsURI(t *testing.T) {
	tests := []struct {
		in   string
		want bool
	}{
		{"hf://unsloth/x/q", true},
		{"https://example.com", true},
		{"http://example.com", true},
		{"Qwen3-Q4_K_M", false},
		{"", false},
	}
	for _, tc := range tests {
		if got := IsURI(tc.in); got != tc.want {
			t.Errorf("IsURI(%q) = %v, want %v", tc.in, got, tc.want)
		}
	}
}
