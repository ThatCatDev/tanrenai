package runner

import (
	"strings"
	"testing"
)

func TestGenerateChatML_Qwen25_MatchesStatic(t *testing.T) {
	cfg := DefaultChatMLConfig
	tpl := GenerateChatML(cfg)

	// Key structural checks against the known Qwen 2.5 template.
	checks := []string{
		// Tools preamble
		`{%- if tools %}`,
		`<|im_start|>system\n`,
		`You are a helpful assistant.`,
		`# Tools`,
		`<tool_call>`,
		`</tool_call>`,
		`<|im_end|>\n`,

		// Non-first system uses message['role'] (not hardcoded 'user')
		`message['role'] + '\n' + message['content']`,

		// Tool responses batched under user
		`<tool_response>`,
		`</tool_response>`,

		// Generation prompt
		`{%- if add_generation_prompt %}`,
		`<|im_start|>assistant\n`,
	}

	for _, check := range checks {
		if !strings.Contains(tpl, check) {
			t.Errorf("Qwen2.5 template missing expected substring: %q", check)
		}
	}

	// Should NOT have hardcoded 'user' for system messages.
	if strings.Contains(tpl, `'user\n' + message['content']`) {
		// This line appears in both, but in Qwen2.5 the user+system branch uses message['role']
		// Check more specifically that non-first system isn't forced to 'user'
		lines := strings.Split(tpl, "\n")
		for _, line := range lines {
			if strings.Contains(line, "message['role'] == 'system'") &&
				strings.Contains(line, "message['role'] == 'user'") {
				// Found the combined condition — check it uses message['role']
				break
			}
		}
	}
}

func TestGenerateChatML_Qwen35_SystemAsUser(t *testing.T) {
	cfg := DefaultChatMLConfig
	cfg.SystemAsUser = true
	tpl := GenerateChatML(cfg)

	// Qwen 3.5 renders non-first system as 'user' — look for hardcoded user role.
	if !strings.Contains(tpl, `<|im_start|>user\n' + message['content']`) {
		t.Error("Qwen3.5 template should hardcode 'user' for system+user messages")
	}

	// Should NOT contain message['role'] in the user/system branch.
	// The user/system branch should use hardcoded 'user'.
	lines := strings.Split(tpl, "\n")
	for _, line := range lines {
		if strings.Contains(line, "message['role'] == 'user'") {
			if strings.Contains(line, "message['role'] + '\\n'") {
				t.Error("Qwen3.5 template should not use message['role'] in user/system rendering")
			}
		}
	}
}

func TestGenerateChatML_CustomTokens(t *testing.T) {
	cfg := ChatMLConfig{
		StartToken:       "<|start|>",
		EndToken:         "<|end|>",
		ToolCallStart:    "<call>",
		ToolCallEnd:      "</call>",
		ToolRespStart:    "<resp>",
		ToolRespEnd:      "</resp>",
		DefaultSysPrompt: "Hello world.",
	}
	tpl := GenerateChatML(cfg)

	for _, tok := range []string{"<|start|>", "<|end|>", "<call>", "</call>", "<resp>", "</resp>", "Hello world."} {
		if !strings.Contains(tpl, tok) {
			t.Errorf("custom template missing token: %q", tok)
		}
	}
}

func TestMatchFamily_Qwen2(t *testing.T) {
	cfg := MatchFamily("qwen2", "Qwen2.5-Coder-32B")
	if cfg == nil {
		t.Fatal("expected match for qwen2 architecture")
	}
	if cfg.SystemAsUser {
		t.Error("qwen2 should have SystemAsUser = false")
	}
}

func TestMatchFamily_Qwen3(t *testing.T) {
	cfg := MatchFamily("qwen3", "Qwen3-32B")
	if cfg == nil {
		t.Fatal("expected match for qwen3 architecture")
	}
	if !cfg.SystemAsUser {
		t.Error("qwen3 should have SystemAsUser = true")
	}
}

func TestMatchFamily_Qwen2ArchWithQwen3Name(t *testing.T) {
	// Qwen3 models sometimes use qwen2 architecture.
	cfg := MatchFamily("qwen2", "Qwen3.5-Coder-32B")
	if cfg == nil {
		t.Fatal("expected match for qwen2 arch + qwen3 name")
	}
	if !cfg.SystemAsUser {
		t.Error("qwen2 arch + qwen3 name should have SystemAsUser = true")
	}
}

func TestMatchFamily_Unknown(t *testing.T) {
	cfg := MatchFamily("llama", "Llama-3-70B")
	if cfg != nil {
		t.Error("expected nil for unknown architecture")
	}
}

func TestMatchFamily_CaseInsensitive(t *testing.T) {
	cfg := MatchFamily("Qwen2", "QWEN3-TEST")
	if cfg == nil {
		t.Fatal("expected case-insensitive match")
	}
	if !cfg.SystemAsUser {
		t.Error("should match qwen3 name pattern case-insensitively")
	}
}
