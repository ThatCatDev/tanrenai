package cmd

import (
	"bufio"
	"context"
	"encoding/json"
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

// pipeEmitter abstracts how a pipe-mode session reports turn events.
// `textEmitter` reproduces the original human-friendly format
// (assistant content on stdout, bracketed status on stderr, ---END--- between
// turns); `jsonEmitter` writes one JSONL event per line on stdout and routes
// every status/log line to stderr so stdout stays parseable. This is the seam
// programmatic hosts (IDE plugins, agent runners, CI scripts) hook into to
// consume turn output without parsing free-form text.
type pipeEmitter interface {
	TextDelta(text string)
	ReasoningDelta(text string)
	ToolCall(call api.ToolCall)
	ToolResult(call api.ToolCall, result *tools.ToolResult)
	Iteration(current, max int)
	Status(format string, args ...any)
	Error(err error)
	GenRate(tokens int, tps float64)
	TurnEnd(reason string)
}

// newPipeEmitter selects an emitter based on the --format flag value.
// Unknown values default to text mode (parseRunParams already rejects them at
// flag-parse time, so this is just a defensive fallback).
func newPipeEmitter(format string) pipeEmitter {
	if format == "json" {
		return &jsonEmitter{}
	}
	return &textEmitter{}
}

// ── text emitter (original behavior) ────────────────────────────────────

// textEmitter is the human-facing pipe output: assistant content streams to
// stdout, every other signal (tool calls, reasoning, status, errors) goes to
// stderr inside `[bracketed]` tags, and turns are separated by `---END---`
// on its own line in stdout.
type textEmitter struct{}

func (textEmitter) TextDelta(text string) {
	os.Stdout.WriteString(text)
}

func (textEmitter) ReasoningDelta(text string) {
	fmt.Fprintf(os.Stderr, "[reasoning] %s\n", text)
}

func (textEmitter) ToolCall(call api.ToolCall) {
	args := call.Function.Arguments
	if len(args) > 200 {
		args = args[:200] + "..."
	}
	fmt.Fprintf(os.Stderr, "[tool_call: %s: %s]\n", call.Function.Name, args)
}

func (textEmitter) ToolResult(call api.ToolCall, result *tools.ToolResult) {
	tag := "tool_result"
	if result.IsError {
		tag = "tool_result_error"
	}
	preview := strings.TrimSpace(result.Output)
	if len(preview) > 200 {
		preview = preview[:200] + "..."
	}
	fmt.Fprintf(os.Stderr, "[%s: %s: %s]\n", tag, call.Function.Name, preview)
}

func (textEmitter) Iteration(current, max int) {
	if max > 0 {
		fmt.Fprintf(os.Stderr, "[iteration %d/%d]\n", current, max)
	} else {
		fmt.Fprintf(os.Stderr, "[iteration %d]\n", current)
	}
}

func (textEmitter) Status(format string, args ...any) {
	fmt.Fprintf(os.Stderr, "[%s]\n", fmt.Sprintf(format, args...))
}

func (textEmitter) Error(err error) {
	fmt.Fprintf(os.Stderr, "[error: %v]\n", err)
}

func (textEmitter) GenRate(tokens int, tps float64) {
	if tps <= 0 {
		return
	}
	fmt.Fprintf(os.Stderr, "[gen: %d tokens, %.1f t/s]\n", tokens, tps)
}

func (textEmitter) TurnEnd(_ string) {
	fmt.Fprintln(os.Stdout, pipeDelimiter)
}

// ── json emitter ────────────────────────────────────────────────────────

// jsonEmitter writes JSONL events to stdout (one event per line), keeping the
// stream cleanly parseable for programmatic hosts. The protocol intentionally
// mirrors the canonical events OD's `json-event-stream` parser already
// translates for other adapters (Codex, Gemini, OpenCode, Cursor) so the
// daemon-side translator stays small.
//
// Wire shape (all events have a `type` discriminator):
//
//	{"type":"text_delta","delta":"..."}
//	{"type":"reasoning_delta","delta":"..."}
//	{"type":"tool_call","id":"<id>","name":"<tool>","arguments":<any>}
//	{"type":"tool_result","tool_call_id":"<id>","output":"...","is_error":false}
//	{"type":"iteration","current":<n>,"max":<n|0 for unlimited>}
//	{"type":"status","label":"thinking"}
//	{"type":"gen_rate","tokens":<n>,"tps":<f>}
//	{"type":"error","message":"..."}
//	{"type":"turn_end","reason":"stop"}
//
// `turn_end` is the load-bearing signal: hosts treat it as the unambiguous
// "this turn is complete, you may submit the next user message" boundary,
// independent of process lifetime. (Text mode uses the `---END---` line
// for the same purpose, but stdin closure / EOF is the only termination
// signal the host can trust there.)
type jsonEmitter struct{}

// emit writes one JSONL line to stdout. Marshal failures fall back to a
// stringified `error` event so the host always sees *something* parseable —
// silently dropping events would let a malformed tool output stall the run.
func (jsonEmitter) emit(event any) {
	b, err := json.Marshal(event)
	if err != nil {
		fallback, _ := json.Marshal(map[string]any{
			"type":    "error",
			"message": fmt.Sprintf("json marshal failed: %v", err),
		})
		_, _ = os.Stdout.Write(fallback)
		_, _ = os.Stdout.Write([]byte("\n"))
		return
	}
	_, _ = os.Stdout.Write(b)
	_, _ = os.Stdout.Write([]byte("\n"))
}

func (e jsonEmitter) TextDelta(text string) {
	e.emit(map[string]any{"type": "text_delta", "delta": text})
}

func (e jsonEmitter) ReasoningDelta(text string) {
	e.emit(map[string]any{"type": "reasoning_delta", "delta": text})
}

func (e jsonEmitter) ToolCall(call api.ToolCall) {
	// Tanrenai's `api.ToolCall.Function.Arguments` is the raw JSON string the
	// model emitted. Parse it so the host receives a structured value
	// (matches Codex/OpenCode shapes); fall back to the raw string if the
	// model emitted invalid JSON so the host still sees the attempt.
	var args any
	if call.Function.Arguments != "" {
		if err := json.Unmarshal([]byte(call.Function.Arguments), &args); err != nil {
			args = call.Function.Arguments
		}
	}
	e.emit(map[string]any{
		"type":      "tool_call",
		"id":        call.ID,
		"name":      call.Function.Name,
		"arguments": args,
	})
}

func (e jsonEmitter) ToolResult(call api.ToolCall, result *tools.ToolResult) {
	e.emit(map[string]any{
		"type":         "tool_result",
		"tool_call_id": call.ID,
		"name":         call.Function.Name,
		"output":       result.Output,
		"is_error":     result.IsError,
	})
}

func (e jsonEmitter) Iteration(current, max int) {
	e.emit(map[string]any{
		"type":    "iteration",
		"current": current,
		"max":     max,
	})
}

func (e jsonEmitter) Status(format string, args ...any) {
	e.emit(map[string]any{
		"type":  "status",
		"label": fmt.Sprintf(format, args...),
	})
}

func (e jsonEmitter) Error(err error) {
	e.emit(map[string]any{
		"type":    "error",
		"message": err.Error(),
	})
}

func (e jsonEmitter) GenRate(tokens int, tps float64) {
	if tps <= 0 {
		return
	}
	e.emit(map[string]any{
		"type":   "gen_rate",
		"tokens": tokens,
		"tps":    tps,
	})
}

func (e jsonEmitter) TurnEnd(reason string) {
	if reason == "" {
		reason = "stop"
	}
	e.emit(map[string]any{"type": "turn_end", "reason": reason})
}

// ── pipe entry point ────────────────────────────────────────────────────

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

	// In JSON mode, startupLog's default behaviour (Info → stdout) would
	// corrupt the JSONL stream with "Loading model X...", "Local GPU server
	// ready", etc. Route every startup line to stderr instead so stdout
	// remains pure JSON from the very first byte.
	var log *startupLog
	if p.pipeFormat == "json" {
		log = &startupLog{emit: func(level, msg string) {
			fmt.Fprintf(os.Stderr, "%s\n", msg)
		}}
	} else {
		log = &startupLog{tui: nil} // text-mode default: stdout/stderr
	}

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
	em := newPipeEmitter(deps.pipeFormat)

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

		turnErr := runPipeTurn(ctx, deps, em, input)
		if turnErr != nil {
			em.Error(turnErr)
		}

		// Signal turn complete — text mode prints ---END---, json mode
		// emits {"type":"turn_end"}. Either way the host gets an
		// unambiguous boundary it can wait on.
		reason := "stop"
		if turnErr != nil {
			reason = "error"
		}
		em.TurnEnd(reason)

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
// summary emission at the end of each turn covers whichever path ran.
// Reset per turn in runPipeTurn before dispatching to the path.
var pipeGenRate = &apiclient.TokenRateTracker{}

// runPipeTurn executes a single user turn through the agent or chat path.
func runPipeTurn(ctx context.Context, deps *sessionDeps, em pipeEmitter, input string) error {
	pipeGenRate.Reset()
	defer func() {
		tokens, tps := pipeGenRate.Snapshot()
		em.GenRate(tokens, tps)
	}()

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
			em.Status("scrolls matched: %s", strings.Join(names, ", "))
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
		return runPipeSimpleTurn(ctx, deps, em, windowedMsgs)
	}

	var result []api.Message
	var err error
	if deps.swarmMode {
		cfg := buildPipeSwarmConfig(deps, em)
		result, err = agent.RunSwarm(ctx, deps.streamFn, windowedMsgs, cfg)
	} else {
		cfg := buildPipeAgentConfig(deps, em)
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

// buildPipeAgentConfig wires agent hooks to the supplied emitter.
//
// In text mode the emitter is `textEmitter`, which reproduces the original
// stdout/stderr split (assistant content → stdout, everything else → stderr,
// XML-filtered through `contentFilt`, reasoning buffered until sentence
// boundaries). In JSON mode the emitter is `jsonEmitter`, which flushes
// every delta directly as a JSONL event with no buffering — programmatic
// consumers want raw token-level events and will assemble their own UX.
func buildPipeAgentConfig(deps *sessionDeps, em pipeEmitter) agent.PlanAgentConfig {
	jsonMode := deps.pipeFormat == "json"

	// Text mode keeps an XML filter on assistant content and buffers
	// reasoning until a sentence boundary so the stderr output reads as
	// prose rather than per-token spam. JSON consumers expect raw deltas,
	// so those buffers are skipped entirely there.
	var contentFilt xmlFilter
	var reasoningBuf strings.Builder

	flushReasoning := func() {
		if jsonMode {
			return
		}
		if reasoningBuf.Len() > 0 {
			em.ReasoningDelta(strings.TrimSpace(reasoningBuf.String()))
			reasoningBuf.Reset()
		}
	}

	flushContent := func() {
		flushReasoning()
		if jsonMode {
			return
		}
		if contentFilt.len() > 0 {
			em.TextDelta(contentFilt.string())
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
						em.ToolCall(call)
					},
					OnToolResult: func(call api.ToolCall, result *tools.ToolResult) {
						em.ToolResult(call, result)
					},
					OnToolApproval: func(call api.ToolCall) agent.ApprovalAction {
						return agent.ApprovalAllow
					},
					OnAssistantMessage: func(content string) {},
				},
			},
			OnIterationStart: func(iteration, maxIter int, messages []api.Message) {
				flushContent()
				em.Iteration(iteration+1, maxIter)
			},
			OnContentDelta: func(delta string) {
				pipeGenRate.Record()
				if jsonMode {
					em.TextDelta(delta)
					return
				}
				contentFilt.write(delta)
				if text := contentFilt.string(); text != "" {
					em.TextDelta(text)
				}
			},
			OnReasoningDelta: func(delta string) {
				pipeGenRate.Record()
				if jsonMode {
					em.ReasoningDelta(delta)
					return
				}
				reasoningBuf.WriteString(delta)
				if strings.ContainsAny(delta, ".\n") && reasoningBuf.Len() > 40 {
					em.ReasoningDelta(strings.TrimSpace(reasoningBuf.String()))
					reasoningBuf.Reset()
				}
			},
			OnThinking: func() {
				em.Status("thinking")
			},
			OnThinkingDone: func() {
				flushReasoning()
				em.Status("generating")
			},
		},
		OnPlanningStart: func() {
			em.Status("planning")
		},
		OnPlanGenerated: func(plan *agent.Plan) {
			for _, step := range plan.Steps {
				em.Status("plan_step %d: %s", step.Index, step.Description)
			}
		},
		OnStepStart: func(stepIdx int, step *agent.PlanStep) {
			flushContent()
			em.Status("step_start %d: %s", step.Index, step.Description)
		},
		OnStepDone: func(stepIdx int, step *agent.PlanStep) {
			flushContent()
			em.Status("step_done %d: %s", step.Index, step.Status.String())
		},
		OnReplan: func(reason string, newPlan *agent.Plan) {
			em.Status("replan: %s", reason)
			for _, step := range newPlan.Steps {
				em.Status("plan_step %d: %s", step.Index, step.Description)
			}
		},
		OnSynthesize: func() {
			flushContent()
			em.Status("synthesizing")
		},
	}
}

// runPipeSimpleTurn handles a non-agent (plain chat) streaming turn.
func runPipeSimpleTurn(ctx context.Context, deps *sessionDeps, em pipeEmitter, messages []api.Message) error {
	jsonMode := deps.pipeFormat == "json"
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
		OnThinking:     func() { em.Status("thinking") },
		OnThinkingDone: func() { em.Status("generating") },
		OnContentDelta: func(delta string) {
			pipeGenRate.Record()
			em.TextDelta(delta)
		},
	})
	if err != nil {
		return err
	}

	if content != "" {
		deps.mgr.Append(api.Message{Role: "assistant", Content: content})
		if !jsonMode && !strings.HasSuffix(content, "\n") {
			// Text mode: ensure the assistant message ends with a newline so
			// the next ---END--- delimiter line stands on its own. JSON mode
			// has no such requirement — turn_end is its own event.
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

// buildPipeSwarmConfig wires swarm hooks to the supplied emitter, mirroring
// `buildPipeAgentConfig` for the orchestrator/worker swarm path. The same
// jsonMode branches apply: text mode buffers reasoning + filters content,
// JSON mode flushes deltas raw.
func buildPipeSwarmConfig(deps *sessionDeps, em pipeEmitter) agent.SwarmConfig {
	jsonMode := deps.pipeFormat == "json"
	var contentFilt xmlFilter
	var reasoningBuf strings.Builder

	flushReasoning := func() {
		if jsonMode {
			return
		}
		if reasoningBuf.Len() > 0 {
			em.ReasoningDelta(strings.TrimSpace(reasoningBuf.String()))
			reasoningBuf.Reset()
		}
	}

	flushContent := func() {
		flushReasoning()
		if jsonMode {
			return
		}
		if contentFilt.len() > 0 {
			em.TextDelta(contentFilt.string())
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
						em.ToolCall(call)
					},
					OnToolResult: func(call api.ToolCall, result *tools.ToolResult) {
						em.ToolResult(call, result)
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
				if jsonMode {
					em.TextDelta(delta)
					return
				}
				contentFilt.write(delta)
				if text := contentFilt.string(); text != "" {
					em.TextDelta(text)
				}
			},
			OnReasoningDelta: func(delta string) {
				pipeGenRate.Record()
				if jsonMode {
					em.ReasoningDelta(delta)
					return
				}
				reasoningBuf.WriteString(delta)
				if strings.ContainsAny(delta, ".\n") && reasoningBuf.Len() > 40 {
					em.ReasoningDelta(strings.TrimSpace(reasoningBuf.String()))
					reasoningBuf.Reset()
				}
			},
			OnThinking: func() {
				em.Status("thinking")
			},
			OnThinkingDone: func() {
				flushReasoning()
				em.Status("generating")
			},
		},
		WorkerTools:   workerTools,
		ArchitectFile: filepath.Join(".tanrenai", "architect.md"),
		OnArchitectSpec: func(depth int, spec string) {
			em.Status("swarm_architect d=%d: %s", depth, strings.ReplaceAll(strings.TrimSpace(spec), "\n", " | "))
		},
		OnPlanGenerated: func(depth int, plan *agent.Plan) {
			for _, step := range plan.Steps {
				em.Status("swarm_plan d=%d: %d. %s", depth, step.Index, step.Description)
			}
		},
		OnWorkerStart: func(depth, stepIdx int, step *agent.PlanStep) {
			flushContent()
			em.Status("swarm_worker_start d=%d %d: %s", depth, step.Index, step.Description)
		},
		OnWorkerDone: func(depth, stepIdx int, step *agent.PlanStep) {
			flushContent()
			em.Status("swarm_worker_done d=%d %d: %s", depth, step.Index, step.Status.String())
		},
		OnVerifyStart: func() {
			flushContent()
			em.Status("swarm_verify")
		},
	}
}
