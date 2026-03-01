package runner

import (
	"fmt"
	"strings"
)

// ChatMLConfig defines the parameters for generating a ChatML-style Jinja template.
type ChatMLConfig struct {
	StartToken       string // e.g. "<|im_start|>"
	EndToken         string // e.g. "<|im_end|>"
	ToolCallStart    string // e.g. "<tool_call>"
	ToolCallEnd      string // e.g. "</tool_call>"
	ToolRespStart    string // e.g. "<tool_response>"
	ToolRespEnd      string // e.g. "</tool_response>"
	DefaultSysPrompt string // e.g. "You are a helpful assistant."
	// SystemAsUser renders non-first system messages as the 'user' role instead
	// of the original role. Required for models like Qwen 3.5 that enforce
	// system messages only at the beginning.
	SystemAsUser bool
}

// DefaultChatMLConfig returns a ChatMLConfig with standard ChatML tokens.
var DefaultChatMLConfig = ChatMLConfig{
	StartToken:       "<|im_start|>",
	EndToken:         "<|im_end|>",
	ToolCallStart:    "<tool_call>",
	ToolCallEnd:      "</tool_call>",
	ToolRespStart:    "<tool_response>",
	ToolRespEnd:      "</tool_response>",
	DefaultSysPrompt: "You are a helpful assistant.",
}

// GenerateChatML produces a complete Jinja chat template from the given config.
// The output is functionally equivalent to the templates previously stored as
// static string constants (qwen25ChatTemplate / qwen35ChatTemplate).
func GenerateChatML(cfg ChatMLConfig) string {
	var b strings.Builder

	// --- Tools preamble ---
	// When tools are provided, emit a system message with tool descriptions.
	fmt.Fprintf(&b, `{%%- if tools %%}
    {{- '%s' }}`, startRole(cfg, "system"))
	fmt.Fprintf(&b, `
    {%%- if messages[0]['role'] == 'system' %%}
        {{- messages[0]['content'] }}
    {%%- else %%}
        {{- '%s' }}
    {%%- endif %%}`, cfg.DefaultSysPrompt)

	// Tool description preamble — uses the config's wrapper tokens.
	toolPreamble := fmt.Sprintf(
		`\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>`,
	)
	toolSuffix := fmt.Sprintf(
		`\n</tools>\n\nFor each function call, return a json object with function name and arguments within %s%s XML tags:\n%s\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n%s`,
		cfg.ToolCallStart, cfg.ToolCallEnd, cfg.ToolCallStart, cfg.ToolCallEnd,
	)

	fmt.Fprintf(&b, `
    {{- "%s" }}
    {%%- for tool in tools %%}
        {{- "\n" }}
        {{- tool | tojson }}
    {%%- endfor %%}
    {{- "%s" }}
    {{- '%s' }}
{%%- else %%}
    {%%- if messages[0]['role'] == 'system' %%}
        {{- '%s' + messages[0]['content'] + '%s' }}
    {%%- endif %%}
{%%- endif %%}`,
		escapeJinja(toolPreamble),
		escapeJinja(toolSuffix),
		endRole(cfg),
		startRole(cfg, "system"), endRole(cfg))

	// --- Message loop ---
	b.WriteString("\n")

	// The role used for non-first system messages and user messages.
	if cfg.SystemAsUser {
		// Qwen 3.5 style: both user and non-first system render as 'user'.
		fmt.Fprintf(&b, `{%%- for message in messages %%}
    {%%- if message['role'] == 'user' or (message['role'] == 'system' and (not tools or not loop.first)) %%}
        {{- '%suser\n' + message['content'] + '%s' }}`,
			cfg.StartToken, endRole(cfg))
	} else {
		// Qwen 2.5 style: non-first system keeps its original role.
		fmt.Fprintf(&b, `{%%- for message in messages %%}
    {%%- if (message['role'] == 'user') or (message['role'] == 'system' and (not tools or not loop.first)) %%}
        {{- '%s' + message['role'] + '\n' + message['content'] + '%s' }}`,
			cfg.StartToken, endRole(cfg))
	}

	// Assistant messages (with optional tool calls).
	fmt.Fprintf(&b, `
    {%%- elif message['role'] == 'assistant' %%}
        {%%- if message.get('tool_calls') %%}
            {{- '%sassistant\n' }}
            {%%- if message['content'] %%}
                {{- message['content'] }}
            {%%- endif %%}
            {%%- for tool_call in message['tool_calls'] %%}
                {%%- if tool_call.get('function') %%}
                    {{- '\n%s\n{"name": "' + tool_call['function']['name'] + '", "arguments": ' + tool_call['function']['arguments'] + '}\n%s' }}
                {%%- endif %%}
            {%%- endfor %%}
            {{- '%s' }}
        {%%- else %%}
            {{- '%sassistant\n' + message['content'] + '%s' }}
        {%%- endif %%}`,
		cfg.StartToken,
		cfg.ToolCallStart, cfg.ToolCallEnd,
		endRole(cfg),
		cfg.StartToken, endRole(cfg))

	// Tool response messages — batched under a user tag.
	fmt.Fprintf(&b, `
    {%%- elif message['role'] == 'tool' %%}
        {%%- if (loop.index0 == 0) or (messages[loop.index0 - 1]['role'] != 'tool') %%}
            {{- '%suser' }}
        {%%- endif %%}
        {{- '\n%s\n' }}
        {{- message['content'] }}
        {{- '\n%s' }}
        {%%- if loop.last or (messages[loop.index0 + 1]['role'] != 'tool') %%}
            {{- '%s' }}
        {%%- endif %%}`,
		cfg.StartToken,
		cfg.ToolRespStart,
		cfg.ToolRespEnd,
		endRole(cfg))

	b.WriteString("\n    {%- endif %}\n{%- endfor %}")

	// --- Generation prompt ---
	fmt.Fprintf(&b, `
{%%- if add_generation_prompt %%}
    {{- '%sassistant\n' }}
{%%- endif %%}`, cfg.StartToken)

	return b.String()
}

// startRole returns the start-of-role token: "<|im_start|>role\n"
func startRole(cfg ChatMLConfig, role string) string {
	return cfg.StartToken + role + `\n`
}

// endRole returns the end-of-role token: "<|im_end|>\n"
func endRole(cfg ChatMLConfig) string {
	return cfg.EndToken + `\n`
}

// escapeJinja escapes double quotes for use inside Jinja {{- "..." }} expressions.
func escapeJinja(s string) string {
	return strings.ReplaceAll(s, `"`, `\"`)
}

// FamilyEntry maps a model architecture (and optional name substring) to a ChatMLConfig.
type FamilyEntry struct {
	Architecture string   // e.g. "qwen2", "qwen3"
	NameContains []string // optional: require name to contain one of these substrings (case-insensitive)
	Config       ChatMLConfig
}

// familyRegistry is searched in order; more specific entries should come first.
var familyRegistry = []FamilyEntry{
	{
		// Qwen3 models running on qwen2 architecture but with "qwen3" in name
		Architecture: "qwen2",
		NameContains: []string{"qwen3"},
		Config: func() ChatMLConfig {
			c := DefaultChatMLConfig
			c.SystemAsUser = true
			return c
		}(),
	},
	{
		// Native qwen3 architecture
		Architecture: "qwen3",
		Config: func() ChatMLConfig {
			c := DefaultChatMLConfig
			c.SystemAsUser = true
			return c
		}(),
	},
	{
		// Standard qwen2 architecture
		Architecture: "qwen2",
		Config:       DefaultChatMLConfig,
	},
}

// MatchFamily finds the ChatMLConfig for the given architecture and model name.
// Returns nil if no entry matches.
func MatchFamily(arch, name string) *ChatMLConfig {
	archLower := strings.ToLower(arch)
	nameLower := strings.ToLower(name)

	for _, entry := range familyRegistry {
		if strings.ToLower(entry.Architecture) != archLower {
			continue
		}
		if len(entry.NameContains) > 0 {
			matched := false
			for _, sub := range entry.NameContains {
				if strings.Contains(nameLower, strings.ToLower(sub)) {
					matched = true
					break
				}
			}
			if !matched {
				continue
			}
		}
		cfg := entry.Config
		return &cfg
	}
	return nil
}
