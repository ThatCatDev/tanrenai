package cmd

import "testing"

func TestStripToolCallXML(t *testing.T) {
	tests := []struct {
		name  string
		input string
		want  string
	}{
		{
			name:  "no tool calls",
			input: "Here is the answer to your question.",
			want:  "Here is the answer to your question.",
		},
		{
			name: "single tool call block",
			input: `Let me read that file.
<tool_call>
<function=file_read>
<parameter=path>
./src/index.ts
</parameter>
</function>
</tool_call>
I can see the project structure.`,
			want: `Let me read that file.
I can see the project structure.`,
		},
		{
			name: "multiple tool call blocks",
			input: `Checking files.
<tool_call>
<function=file_read>
<parameter=path>a.ts</parameter>
</function>
</tool_call>
Found first file.
<tool_call>
<function=file_read>
<parameter=path>b.ts</parameter>
</function>
</tool_call>
Found second file.`,
			want: `Checking files.
Found first file.
Found second file.`,
		},
		{
			name:  "only tool call block",
			input: "<tool_call>\n<function=list_dir>\n<parameter=path>.</parameter>\n</function>\n</tool_call>",
			want:  "",
		},
		{
			name:  "empty string",
			input: "",
			want:  "",
		},
		{
			name:  "no xml just angle brackets",
			input: "Use <b>bold</b> text here",
			want:  "Use <b>bold</b> text here",
		},
		{
			name: "qwen3 format with equals",
			input: "Good, I've seen the core system.ts. Let me continue reading the other core files.\n\n<tool_call>\n<function=file_read>\n<parameter=path>\nsrc/core/tasks.ts\n</parameter>\n</function>\n</tool_call>\nLet me continue reading the core files.",
			want:  "Good, I've seen the core system.ts. Let me continue reading the other core files.\n\nLet me continue reading the core files.",
		},
		{
			name:  "only qwen3 tool call",
			input: "<tool_call>\n<function=file_read>\n<parameter=path>\nsrc/core/taskbar.ts\n</parameter>\n</function>\n</tool_call>",
			want:  "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := stripToolCallXML(tt.input)
			if got != tt.want {
				t.Errorf("stripToolCallXML() = %q, want %q", got, tt.want)
			}
		})
	}
}
