package tools

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"os/exec"
	"time"
)

const maxGitOutput = 32 * 1024 // 32KB

// GitInfoTool provides git operations like log, diff, status.
type GitInfoTool struct{}

type gitInfoArgs struct {
	Command string `json:"command"`
	Args    string `json:"args"`
}

func (t *GitInfoTool) Name() string { return "git_info" }

func (t *GitInfoTool) Description() string {
	return `Run read-only git commands. Available commands:
- "status" — show working tree status
- "log" — show recent commits (args: number of commits, default 10)
- "diff" — show unstaged changes (args: optional file path)
- "diff_staged" — show staged changes
- "show" — show a specific commit (args: commit hash)
- "blame" — show line-by-line authorship (args: file path)
- "branch" — list branches`
}

func (t *GitInfoTool) Parameters() json.RawMessage {
	return Schema{
		Type: "object",
		Properties: map[string]SchemaProperty{
			"command": {Type: "string", Description: "Git command: status, log, diff, diff_staged, show, blame, branch"},
			"args":    {Type: "string", Description: "Additional arguments (e.g. file path for diff/blame, commit hash for show, number for log)"},
		},
		Required: []string{"command"},
	}.MustMarshal()
}

func (t *GitInfoTool) Execute(ctx context.Context, arguments string) (*ToolResult, error) {
	var args gitInfoArgs
	if err := json.Unmarshal([]byte(arguments), &args); err != nil {
		return ErrorResult(fmt.Sprintf("invalid arguments: %v", err)), nil
	}

	var gitArgs []string

	switch args.Command {
	case "status":
		gitArgs = []string{"status", "--short"}

	case "log":
		n := "10"
		if args.Args != "" {
			n = args.Args
		}
		gitArgs = []string{"log", "--oneline", "-n", n}

	case "diff":
		gitArgs = []string{"diff"}
		if args.Args != "" {
			gitArgs = append(gitArgs, args.Args)
		}

	case "diff_staged":
		gitArgs = []string{"diff", "--staged"}
		if args.Args != "" {
			gitArgs = append(gitArgs, args.Args)
		}

	case "show":
		if args.Args == "" {
			return ErrorResult("show requires a commit hash in args"), nil
		}
		gitArgs = []string{"show", "--stat", args.Args}

	case "blame":
		if args.Args == "" {
			return ErrorResult("blame requires a file path in args"), nil
		}
		gitArgs = []string{"blame", args.Args}

	case "branch":
		gitArgs = []string{"branch", "-a"}

	default:
		return ErrorResult(fmt.Sprintf("unknown git command: %q. Use: status, log, diff, diff_staged, show, blame, branch", args.Command)), nil
	}

	ctx, cancel := context.WithTimeout(ctx, 15*time.Second)
	defer cancel()

	cmd := exec.CommandContext(ctx, "git", gitArgs...)
	var buf bytes.Buffer
	cmd.Stdout = &buf
	cmd.Stderr = &buf

	if err := cmd.Run(); err != nil {
		return ErrorResult(fmt.Sprintf("git %s failed: %v\n%s", args.Command, err, buf.String())), nil
	}

	output := buf.String()
	if output == "" {
		output = "(no output)"
	}
	if len(output) > maxGitOutput {
		output = output[:maxGitOutput] + "\n[truncated]"
	}

	return &ToolResult{Output: output}, nil
}
