package cmd

import (
	"bufio"
	"context"
	"fmt"
	"io"
	"log/slog"
	"os"
	"os/signal"
	"path/filepath"
	"strings"
	"time"

	"github.com/ThatCatDev/tanrenai/client/internal/chatctx"
	"github.com/ThatCatDev/tanrenai/shared/agent"
	"github.com/ThatCatDev/tanrenai/shared/apiclient"
	"github.com/ThatCatDev/tanrenai/shared/pkg/api"
	"github.com/ThatCatDev/tanrenai/shared/scrolls"
	"github.com/ThatCatDev/tanrenai/shared/tools"
)

const pipeDelimiter = "---END---"

// pipeStatus writes a bracketed status line to stderr.
func pipeStatus(format string, args ...any) {
	msg := fmt.Sprintf(format, args...)
	fmt.Fprintf(os.Stderr, "[%s]\n", msg)
}

// emitPipeGenRateSummary prints a one-line generation-rate summary to
// stderr at the end of each turn, e.g. `[gen: 192 tokens, 42 t/s]`. Silent
// when fewer than two tokens streamed — one sample can't yield a rate, and
// the line would just be noise for non-generation turns (e.g. tool-only).
func emitPipeGenRateSummary() {
	tokens, tps := pipeGenRate.Snapshot()
	if tps <= 0 {
		return
	}
	pipeStatus("gen: %d tokens, %.1f t/s", tokens, tps)
}

// startPipe is the entry point for non-interactive pipe mode.
func startPipe(ctx context.Context, p runParams) error {
	ctx, stop := signal.NotifyContext(ctx, os.Interrupt)
	defer stop()

	// Set up file-only logging (no TUI handler).
	logFile, logErr := openLogFile()
	if logErr != nil {
		slog.SetDefault(slog.New(slog.NewTextHandler(io.Discard, nil)))
	} else {
		defer logFile.Close()
		slog.SetDefault(slog.New(slog.NewTextHandler(logFile, &slog.HandlerOptions{Level: slog.LevelDebug})))
	}

	log := &startupLog{tui: nil} // prints to stdout/stderr
	deps, err := setupSession(ctx, p, log)
	if err != nil {
		return err
	}
	if deps.cleanupFn != nil {
		defer deps.cleanupFn()
	}

	return runPipeLoop(ctx, deps)
}

// runPipeLoop reads messages from stdin and runs turns until EOF.
func runPipeLoop(ctx context.Context, deps *sessionDeps) error {
	scanner := bufio.NewScanner(os.Stdin)
	scanner.Buffer(make([]byte, 0, 1024*1024), 1024*1024)

	for {
		input, eof := readPipeMessage(scanner)
		input = strings.TrimSpace(input)
		if input == "" && eof {
			return nil
		}
		if input == "" {
			if eof {
				return nil
			}
			continue
		}

		if err := runPipeTurn(ctx, deps, input); err != nil {
			pipeStatus("error: %v", err)
		}

		// Signal turn complete.
		fmt.Fprintln(os.Stdout, pipeDelimiter)

		if eof {
			return nil
		}
	}
}

// readPipeMessage accumulates lines from the scanner until it sees the
// delimiter line or EOF. Returns the message and whether EOF was reached.
func readPipeMessage(scanner *bufio.Scanner) (string, bool) {
	var lines []string
	for scanner.Scan() {
		line := scanner.Text()
		if line == pipeDelimiter {
			return strings.Join(lines, "\n"), false
		}
		lines = append(lines, line)
	}
	// EOF
	return strings.Join(lines, "\n"), true
}

// pipeGenRate is the per-process generation-rate tracker for pipe mode.
// Shared by both runPipeSimpleTurn and the agent/swarm configs so the
// summary line at the end of each turn covers whichever path ran. Reset
// per turn in runPipeTurn before dispatching to the path.
var pipeGenRate = &apiclient.TokenRateTracker{}

// runPipeTurn executes a single user turn through the agent or chat path.
func runPipeTurn(ctx context.Context, deps *sessionDeps, input string) error {
	pipeGenRate.Reset()
	defer emitPipeGenRateSummary()

	deps.mgr.Append(api.Message{Role: "user", Content: input})

	// Match scrolls.
	if deps.scrollsEnabled {
		matched := scrolls.Match(deps.allScrolls, input, 3)
		if len(matched) > 0 {
			var scrollMsgs []api.Message
			var names []string
			for _, s := range matched {
				content := fmt.Sprintf("[Scroll: %s]\n%s", s.Name, s.Content)
				scrollMsgs = append(scrollMsgs, api.Message{Role: "system", Content: content})
				names = append(names, s.Name)
			}
			deps.mgr.SetScrolls(scrollMsgs)
			fmt.Fprintf(os.Stderr, "[scrolls] matched: %s\n", strings.Join(names, ", "))
		} else {
			deps.mgr.ClearScrolls()
		}
	}

	// Memory search.
	if deps.memoryEnabled {
		results, err := deps.client.MemorySearch(ctx, input, 3)
		if err == nil && len(results.Results) > 0 {
			var memMsgs []api.Message
			for _, r := range results.Results {
				userMsg := truncate(r.Entry.UserMsg, 200)
				assistMsg := truncate(r.Entry.AssistMsg, 500)
				memContent := fmt.Sprintf("[Memory from %s] User asked: %s\nAssistant replied: %s",
					r.Entry.Timestamp.Format("2006-01-02"), userMsg, assistMsg)
				memMsgs = append(memMsgs, api.Message{Role: "system", Content: memContent})
			}
			deps.mgr.SetMemories(memMsgs)
		} else {
			deps.mgr.ClearMemories()
		}
	}

	// Summarise if needed.
	if deps.mgr.NeedsSummary() {
		_ = deps.mgr.Summarize(ctx, chatctx.CompletionFunc(deps.completeFn))
	}

	windowedMsgs := deps.mgr.Messages()

	if !deps.agentMode {
		return runPipeSimpleTurn(ctx, deps, windowedMsgs)
	}

	var result []api.Message
	var err error
	if deps.swarmMode {
		cfg := buildPipeSwarmConfig(deps)
		result, err = agent.RunSwarm(ctx, deps.streamFn, windowedMsgs, cfg)
	} else {
		cfg := buildPipeAgentConfig(deps)
		result, err = agent.RunPlannedStreaming(ctx, deps.streamFn, windowedMsgs, cfg)
	}
	if err != nil {
		return err
	}

	// Persist new messages.
	if len(result) > len(windowedMsgs) {
		newMsgs := result[len(windowedMsgs):]
		deps.mgr.AppendMany(newMsgs)
		persistPipeMemory(ctx, deps, newMsgs)
	}

	return nil
}

// buildPipeAgentConfig wires agent hooks to stdout/stderr for pipe mode.
func buildPipeAgentConfig(deps *sessionDeps) agent.PlanAgentConfig {
	var contentFilt xmlFilter
	var reasoningBuf strings.Builder

	flushReasoning := func() {
		if reasoningBuf.Len() > 0 {
			fmt.Fprintf(os.Stderr, "[reasoning] %s\n", strings.TrimSpace(reasoningBuf.String()))
			reasoningBuf.Reset()
		}
	}

	flushContent := func() {
		flushReasoning()
		if contentFilt.len() > 0 {
			os.Stdout.WriteString(contentFilt.string())
		}
	}

	return agent.PlanAgentConfig{
		StreamingConfig: agent.StreamingConfig{
			Config: agent.Config{
				MaxIterations:     deps.maxIterations,
				MaxResponseTokens: deps.maxTokens,
				EnableThinking:    deps.enableThinking,
				Tools:             deps.registry,
				Hooks: agent.Hooks{
					OnToolCall: func(call api.ToolCall) {
						flushContent()
						args := call.Function.Arguments
						if len(args) > 200 {
							args = args[:200] + "..."
						}
						pipeStatus("tool_call: %s: %s", call.Function.Name, args)
					},
					OnToolResult: func(call api.ToolCall, result *tools.ToolResult) {
						tag := "tool_result"
						if result.IsError {
							tag = "tool_result_error"
						}
						preview := strings.TrimSpace(result.Output)
						if len(preview) > 200 {
							preview = preview[:200] + "..."
						}
						pipeStatus("%s: %s: %s", tag, call.Function.Name, preview)
					},
					OnToolApproval: func(call api.ToolCall) agent.ApprovalAction {
						return agent.ApprovalAllow
					},
					OnAssistantMessage: func(content string) {},
				},
			},
			OnIterationStart: func(iteration, maxIter int, messages []api.Message) {
				flushContent()
				if maxIter > 0 {
					pipeStatus("iteration %d/%d", iteration+1, maxIter)
				} else {
					pipeStatus("iteration %d", iteration+1)
				}
			},
			OnContentDelta: func(delta string) {
				pipeGenRate.Record()
				contentFilt.write(delta)
				if text := contentFilt.string(); text != "" {
					os.Stdout.WriteString(text)
				}
			},
			OnReasoningDelta: func(delta string) {
				pipeGenRate.Record()
				// Accumulate reasoning tokens; flush on sentence boundaries.
				reasoningBuf.WriteString(delta)
				if strings.ContainsAny(delta, ".\n") && reasoningBuf.Len() > 40 {
					fmt.Fprintf(os.Stderr, "[reasoning] %s\n", strings.TrimSpace(reasoningBuf.String()))
					reasoningBuf.Reset()
				}
			},
			OnThinking: func() {
				pipeStatus("thinking")
			},
			OnThinkingDone: func() {
				flushReasoning()
				pipeStatus("generating")
			},
		},
		OnPlanningStart: func() {
			pipeStatus("planning")
		},
		OnPlanGenerated: func(plan *agent.Plan) {
			for _, step := range plan.Steps {
				pipeStatus("plan_step %d: %s", step.Index, step.Description)
			}
		},
		OnStepStart: func(stepIdx int, step *agent.PlanStep) {
			flushContent()
			pipeStatus("step_start %d: %s", step.Index, step.Description)
		},
		OnStepDone: func(stepIdx int, step *agent.PlanStep) {
			flushContent()
			pipeStatus("step_done %d: %s", step.Index, step.Status.String())
		},
		OnReplan: func(reason string, newPlan *agent.Plan) {
			pipeStatus("replan: %s", reason)
			for _, step := range newPlan.Steps {
				pipeStatus("plan_step %d: %s", step.Index, step.Description)
			}
		},
		OnSynthesize: func() {
			flushContent()
			pipeStatus("synthesizing")
		},
	}
}

// runPipeSimpleTurn handles a non-agent (plain chat) streaming turn.
func runPipeSimpleTurn(ctx context.Context, deps *sessionDeps, messages []api.Message) error {
	req := &api.ChatCompletionRequest{
		Model:    deps.modelName,
		Messages: messages,
		Stream:   true,
	}
	events, err := deps.client.StreamCompletion(ctx, req)
	if err != nil {
		return err
	}

	content, err := streamSimpleChat(events, chatStreamHooks{
		OnThinking:     func() { pipeStatus("thinking") },
		OnThinkingDone: func() { pipeStatus("generating") },
		OnContentDelta: func(delta string) {
			pipeGenRate.Record()
			os.Stdout.WriteString(delta)
		},
	})
	if err != nil {
		return err
	}

	if content != "" {
		deps.mgr.Append(api.Message{Role: "assistant", Content: content})
		if !strings.HasSuffix(content, "\n") {
			fmt.Fprintln(os.Stdout)
		}
	}

	return nil
}

// persistPipeMemory stores a memory entry when memory is enabled (pipe mode).
func persistPipeMemory(ctx context.Context, deps *sessionDeps, newMsgs []api.Message) {
	if !deps.memoryEnabled {
		return
	}

	var assistContent string
	for _, msg := range newMsgs {
		if msg.Role == "assistant" && msg.Content != "" {
			if assistContent != "" {
				assistContent += "\n"
			}
			assistContent += msg.Content
		}
	}
	if len(assistContent) > 2000 {
		assistContent = assistContent[:2000]
	}

	var userInput string
	for i := len(newMsgs) - 1; i >= 0; i-- {
		if newMsgs[i].Role == "user" {
			userInput = newMsgs[i].Content
			break
		}
	}

	if assistContent == "" || userInput == "" {
		return
	}

	storeCtx, cancel := context.WithTimeout(ctx, 10*time.Second)
	defer cancel()
	if _, err := deps.client.MemoryStore(storeCtx, userInput, assistContent); err != nil {
		slog.Error("failed to store memory", "error", err)
	}
}

// buildPipeSwarmConfig wires swarm hooks to stderr for pipe mode.
func buildPipeSwarmConfig(deps *sessionDeps) agent.SwarmConfig {
	var contentFilt xmlFilter
	var reasoningBuf strings.Builder

	flushReasoning := func() {
		if reasoningBuf.Len() > 0 {
			fmt.Fprintf(os.Stderr, "[reasoning] %s\n", strings.TrimSpace(reasoningBuf.String()))
			reasoningBuf.Reset()
		}
	}

	flushContent := func() {
		flushReasoning()
		if contentFilt.len() > 0 {
			os.Stdout.WriteString(contentFilt.string())
		}
	}

	// Worker tools: filesystem + shell, no web_search or git_info.
	var workerTools *tools.Registry
	if deps.registry != nil {
		workerTools = deps.registry.Subset(
			"file_read", "file_write", "patch_file",
			"list_dir", "grep_search", "shell_exec",
		)
	}

	return agent.SwarmConfig{
		StreamingConfig: agent.StreamingConfig{
			Config: agent.Config{
				MaxIterations:     deps.maxIterations,
				MaxResponseTokens: deps.maxTokens,
				EnableThinking:    deps.enableThinking,
				Tools:             deps.registry,
				Hooks: agent.Hooks{
					OnToolCall: func(call api.ToolCall) {
						flushContent()
						args := call.Function.Arguments
						if len(args) > 200 {
							args = args[:200] + "..."
						}
						pipeStatus("tool_call: %s: %s", call.Function.Name, args)
					},
					OnToolResult: func(call api.ToolCall, result *tools.ToolResult) {
						tag := "tool_result"
						if result.IsError {
							tag = "tool_result_error"
						}
						preview := strings.TrimSpace(result.Output)
						if len(preview) > 200 {
							preview = preview[:200] + "..."
						}
						pipeStatus("%s: %s: %s", tag, call.Function.Name, preview)
					},
					OnToolApproval: func(call api.ToolCall) agent.ApprovalAction {
						return agent.ApprovalAllow
					},
					OnAssistantMessage: func(content string) {},
				},
			},
			OnIterationStart: func(iteration, maxIter int, messages []api.Message) {
				flushContent()
			},
			OnContentDelta: func(delta string) {
				pipeGenRate.Record()
				contentFilt.write(delta)
				if text := contentFilt.string(); text != "" {
					os.Stdout.WriteString(text)
				}
			},
			OnReasoningDelta: func(delta string) {
				pipeGenRate.Record()
				reasoningBuf.WriteString(delta)
				if strings.ContainsAny(delta, ".\n") && reasoningBuf.Len() > 40 {
					fmt.Fprintf(os.Stderr, "[reasoning] %s\n", strings.TrimSpace(reasoningBuf.String()))
					reasoningBuf.Reset()
				}
			},
			OnThinking: func() {
				pipeStatus("thinking")
			},
			OnThinkingDone: func() {
				flushReasoning()
				pipeStatus("generating")
			},
		},
		WorkerTools:   workerTools,
		ArchitectFile: filepath.Join(".tanrenai", "architect.md"),
		OnArchitectSpec: func(depth int, spec string) {
			pipeStatus("swarm_architect d=%d: %s", depth, strings.ReplaceAll(strings.TrimSpace(spec), "\n", " | "))
		},
		OnPlanGenerated: func(depth int, plan *agent.Plan) {
			for _, step := range plan.Steps {
				pipeStatus("swarm_plan d=%d: %d. %s", depth, step.Index, step.Description)
			}
		},
		OnWorkerStart: func(depth, stepIdx int, step *agent.PlanStep) {
			flushContent()
			pipeStatus("swarm_worker_start d=%d %d: %s", depth, step.Index, step.Description)
		},
		OnWorkerDone: func(depth, stepIdx int, step *agent.PlanStep) {
			flushContent()
			pipeStatus("swarm_worker_done d=%d %d: %s", depth, step.Index, step.Status.String())
		},
		OnVerifyStart: func() {
			flushContent()
			pipeStatus("swarm_verify")
		},
	}
}
