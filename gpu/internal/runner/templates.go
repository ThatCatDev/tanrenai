package runner

import (
	"os"
	"path/filepath"
)

// Qwen2.5 chat template with native tool support.
// This replaces the Hermes 2 Pro fallback that llama-server uses when the
// GGUF template doesn't "natively describe tools". The native format produces
// more reliable tool calls (including multiple calls per response) and uses
// fewer tokens for tool descriptions.
const qwen25ChatTemplate = `{%- if tools %}
    {{- '<|im_start|>system\n' }}
    {%- if messages[0]['role'] == 'system' %}
        {{- messages[0]['content'] }}
    {%- else %}
        {{- 'You are a helpful assistant.' }}
    {%- endif %}
    {{- "\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>" }}
    {%- for tool in tools %}
        {{- "\n" }}
        {{- tool | tojson }}
    {%- endfor %}
    {{- "\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>" }}
    {{- '<|im_end|>\n' }}
{%- else %}
    {%- if messages[0]['role'] == 'system' %}
        {{- '<|im_start|>system\n' + messages[0]['content'] + '<|im_end|>\n' }}
    {%- endif %}
{%- endif %}
{%- for message in messages %}
    {%- if (message['role'] == 'user') or (message['role'] == 'system' and (not tools or not loop.first)) %}
        {{- '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n' }}
    {%- elif message['role'] == 'assistant' %}
        {%- if message.get('tool_calls') %}
            {{- '<|im_start|>assistant\n' }}
            {%- if message['content'] %}
                {{- message['content'] }}
            {%- endif %}
            {%- for tool_call in message['tool_calls'] %}
                {%- if tool_call.get('function') %}
                    {{- '\n<tool_call>\n{"name": "' + tool_call['function']['name'] + '", "arguments": ' + tool_call['function']['arguments'] + '}\n</tool_call>' }}
                {%- endif %}
            {%- endfor %}
            {{- '<|im_end|>\n' }}
        {%- else %}
            {{- '<|im_start|>assistant\n' + message['content'] + '<|im_end|>\n' }}
        {%- endif %}
    {%- elif message['role'] == 'tool' %}
        {%- if (loop.index0 == 0) or (messages[loop.index0 - 1]['role'] != 'tool') %}
            {{- '<|im_start|>user' }}
        {%- endif %}
        {{- '\n<tool_response>\n' }}
        {{- message['content'] }}
        {{- '\n</tool_response>' }}
        {%- if loop.last or (messages[loop.index0 + 1]['role'] != 'tool') %}
            {{- '<|im_end|>\n' }}
        {%- endif %}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
{%- endif %}`

// WriteQwen25Template writes the Qwen2.5 chat template to a temp file
// and returns its path. The caller should clean up the file when done.
func WriteQwen25Template() (string, error) {
	dir := os.TempDir()
	path := filepath.Join(dir, "tanrenai-qwen25-chat.jinja")
	if err := os.WriteFile(path, []byte(qwen25ChatTemplate), 0644); err != nil {
		return "", err
	}
	return path, nil
}
