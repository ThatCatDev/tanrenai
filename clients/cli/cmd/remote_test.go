package cmd

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
		// No quant suffix — can't guess, caller falls through to explicit-URI error.
		{"Qwen2.5-7B-Instruct", ""},
		{"just-a-plain-name", ""},
		{"", ""},
	}
	for _, tc := range tests {
		got := resolveBareNameToURI(tc.name)
		if got != tc.want {
			t.Errorf("resolveBareNameToURI(%q) = %q, want %q", tc.name, got, tc.want)
		}
	}
}
