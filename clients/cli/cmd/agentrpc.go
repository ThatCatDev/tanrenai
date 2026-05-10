package cmd

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/spf13/cobra"

	"github.com/ThatCatDev/tanrenai/client/internal/chatctx"
	"github.com/ThatCatDev/tanrenai/shared/agent"
	"github.com/ThatCatDev/tanrenai/shared/pkg/api"
	"github.com/ThatCatDev/tanrenai/shared/scrolls"
	"github.com/ThatCatDev/tanrenai/shared/tools"
)

// AgentRPCProtocolVersion is bumped on any breaking IPC change. The extension
// advertises the version it speaks in `init`; the CLI rejects mismatches.
const AgentRPCProtocolVersion = 1

// ── Inbound message types (extension → CLI) ────────────────────────────

type rpcInitMsg struct {
	Type             string   `json:"type"`
	ProtocolVersion  int      `json:"protocolVersion"`
	Model            string   `json:"model"`
	AgentMode        bool     `json:"agentMode"`
	SwarmMode        bool     `json:"swarmMode"`
	EnableMemory     bool     `json:"enableMemory"`
	EnableScrolls    bool     `json:"enableScrolls"`
	InterceptedTools []string `json:"interceptedTools"`
	WorkspaceRoot    string   `json:"workspaceRoot"`
	MaxIterations    int      `json:"maxIterations"`
	SystemPrompt     string   `json:"systemPrompt"`
}

type rpcUserMessageMsg struct {
	Type    string `json:"type"`
	Content string `json:"content"`
}

type rpcToolResultMsg struct {
	Type    string `json:"type"`
	ID      string `json:"id"`
	OK      bool   `json:"ok"`
	Content string `json:"content,omitempty"`
	Error   string `json:"error,omitempty"`
}

type rpcApprovalResponseMsg struct {
	Type   string `json:"type"`
	ID     string `json:"id"`
	Action string `json:"action"` // "allow" | "deny" | "always"
}

// ── Outbound message types (CLI → extension) ───────────────────────────

type rpcReadyMsg struct {
	Type            string    `json:"type"`
	ProtocolVersion int       `json:"protocolVersion"`
	Tools           []rpcTool `json:"tools"`
	Model           string    `json:"model"`
}

type rpcTool struct {
	Name        string          `json:"name"`
	Description string          `json:"description"`
	Schema      json.RawMessage `json:"schema"`
}

type rpcContentDeltaMsg struct {
	Type string `json:"type"`
	Text string `json:"text"`
}

type rpcIterationStartMsg struct {
	Type          string `json:"type"`
	Iteration     int    `json:"iteration"`
	MaxIterations int    `json:"maxIterations"`
}

type rpcToolCallMsg struct {
	Type      string `json:"type"`
	ID        string `json:"id"`
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

type rpcToolResultLocalMsg struct {
	Type    string `json:"type"`
	ID      string `json:"id"`
	OK      bool   `json:"ok"`
	Content string `json:"content,omitempty"`
	Error   string `json:"error,omitempty"`
}

type rpcApprovalRequiredMsg struct {
	Type      string `json:"type"`
	ID        string `json:"id"`
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

type rpcTurnDoneMsg struct {
	Type   string `json:"type"`
	OK     bool   `json:"ok"`
	Reason string `json:"reason,omitempty"`
}

type rpcErrorMsg struct {
	Type    string `json:"type"`
	Message string `json:"message"`
	Fatal   bool   `json:"fatal"`
}

// ── RPC server ────────────────────────────────────────────────────────

// rpcServer owns stdout writes (serialized) and the pending-tool-call map.
// Inbound messages are routed by the main loop; this type is the writer side
// plus the Tool-execution hand-off used by RPCTool and approval prompts.
type rpcServer struct {
	enc            *json.Encoder
	encMu          sync.Mutex
	pendingTools   sync.Map // id → chan rpcToolResultMsg
	pendingApprove sync.Map // id → chan rpcApprovalResponseMsg
	nextID         atomic.Uint64
	interceptedSet map[string]bool
}

func newRPCServer(out io.Writer, intercepted []string) *rpcServer {
	set := make(map[string]bool, len(intercepted))
	for _, n := range intercepted {
		set[n] = true
	}

	return &rpcServer{
		enc:            json.NewEncoder(out),
		interceptedSet: set,
	}
}

func (s *rpcServer) write(msg any) error {
	s.encMu.Lock()
	defer s.encMu.Unlock()

	return s.enc.Encode(msg)
}

func (s *rpcServer) writeError(message string, fatal bool) {
	_ = s.write(rpcErrorMsg{Type: "error", Message: message, Fatal: fatal})
}

// RequestTool implements tools.RPCRequester for intercepted tools.
func (s *rpcServer) RequestTool(ctx context.Context, name, arguments string) (*tools.ToolResult, error) {
	id := fmt.Sprintf("rpc_%d", s.nextID.Add(1))
	ch := make(chan rpcToolResultMsg, 1)
	s.pendingTools.Store(id, ch)
	defer s.pendingTools.Delete(id)

	if err := s.write(rpcToolCallMsg{
		Type:      "tool_call_request",
		ID:        id,
		Name:      name,
		Arguments: arguments,
	}); err != nil {
		return nil, fmt.Errorf("rpc: write tool_call_request: %w", err)
	}

	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	case res := <-ch:
		if !res.OK {
			return tools.ErrorResult(res.Error), nil
		}

		return &tools.ToolResult{Output: res.Content}, nil
	}
}

func (s *rpcServer) handleToolResult(msg rpcToolResultMsg) {
	if v, ok := s.pendingTools.Load(msg.ID); ok {
		if ch, _ := v.(chan rpcToolResultMsg); ch != nil {
			ch <- msg
		}
	}
}

// requestApproval emits approval_required and blocks on approval_response.
func (s *rpcServer) requestApproval(ctx context.Context, call api.ToolCall) agent.ApprovalAction {
	id := fmt.Sprintf("appr_%d", s.nextID.Add(1))
	ch := make(chan rpcApprovalResponseMsg, 1)
	s.pendingApprove.Store(id, ch)
	defer s.pendingApprove.Delete(id)

	if err := s.write(rpcApprovalRequiredMsg{
		Type:      "approval_required",
		ID:        id,
		Name:      call.Function.Name,
		Arguments: call.Function.Arguments,
	}); err != nil {
		return agent.ApprovalBlock
	}

	select {
	case <-ctx.Done():
		return agent.ApprovalBlock
	case res := <-ch:
		switch res.Action {
		case "allow":
			return agent.ApprovalAllow
		case "always":
			return agent.ApprovalAlwaysAllow
		default:
			return agent.ApprovalBlock
		}
	}
}

func (s *rpcServer) handleApprovalResponse(msg rpcApprovalResponseMsg) {
	if v, ok := s.pendingApprove.Load(msg.ID); ok {
		if ch, _ := v.(chan rpcApprovalResponseMsg); ch != nil {
			ch <- msg
		}
	}
}

// ── Cobra command ─────────────────────────────────────────────────────

var agentRPCCmd = &cobra.Command{
	Use:   "agent-rpc",
	Short: "Run the agent driven by an external client over JSON-stdio (used by editor extensions)",
	Long: `agent-rpc reads NDJSON messages from stdin and writes NDJSON to stdout,
exposing the agent loop and tools to an external UI such as a VS Code
extension. The first message must be 'init'; the response is 'ready' with the
tool catalogue. See clients/vscode for the consuming extension.`,
	Args: cobra.NoArgs,
	RunE: func(cmd *cobra.Command, args []string) error {
		return runAgentRPC(cmd.Context())
	},
}

func init() {
	rootCmd.AddCommand(agentRPCCmd)
}

// ── Main loop ─────────────────────────────────────────────────────────

func runAgentRPC(ctx context.Context) error {
	// File-only logging — stdout is reserved for NDJSON.
	logFile, logErr := openLogFile()
	if logErr != nil {
		slog.SetDefault(slog.New(slog.NewTextHandler(io.Discard, nil)))
	} else {
		defer logFile.Close()
		slog.SetDefault(slog.New(slog.NewTextHandler(logFile, &slog.HandlerOptions{Level: slog.LevelDebug})))
	}

	dec := json.NewDecoder(os.Stdin)

	// Phase 1: handshake.
	var initRaw json.RawMessage
	if err := dec.Decode(&initRaw); err != nil {
		return fmt.Errorf("agent-rpc: failed to read init: %w", err)
	}
	var init rpcInitMsg
	if err := json.Unmarshal(initRaw, &init); err != nil {
		return fmt.Errorf("agent-rpc: failed to parse init: %w", err)
	}
	if init.Type != "init" {
		return fmt.Errorf("agent-rpc: first message must be 'init', got %q", init.Type)
	}
	if init.ProtocolVersion != AgentRPCProtocolVersion {
		return fmt.Errorf("agent-rpc: protocol version mismatch (extension=%d, cli=%d) — update both", init.ProtocolVersion, AgentRPCProtocolVersion)
	}
	if init.Model == "" {
		return errors.New("agent-rpc: init.model is required")
	}

	// Honour workspaceRoot so relative paths in tool args resolve correctly.
	if init.WorkspaceRoot != "" {
		if err := os.Chdir(init.WorkspaceRoot); err != nil {
			return fmt.Errorf("agent-rpc: chdir to workspaceRoot %q: %w", init.WorkspaceRoot, err)
		}
	}

	srv := newRPCServer(os.Stdout, init.InterceptedTools)

	// Build session deps via the same setup the TUI/pipe use.
	p := runParams{
		model:         init.Model,
		systemPrompt:  init.SystemPrompt,
		agentMode:     init.AgentMode || init.SwarmMode,
		swarmMode:     init.SwarmMode,
		memoryEnabled: init.EnableMemory,
		maxIterations: init.MaxIterations,
		thinking:      true,
		noScrolls:     !init.EnableScrolls,
	}
	deps, err := setupSession(ctx, p, &startupLog{tui: nil})
	if err != nil {
		srv.writeError(fmt.Sprintf("setup failed: %v", err), true)

		return err
	}
	if deps.cleanupFn != nil {
		defer deps.cleanupFn()
	}

	// Swap intercepted tools to RPCTool wrappers, preserving description+schema.
	if deps.registry != nil {
		for _, name := range init.InterceptedTools {
			orig := deps.registry.Get(name)
			if orig == nil {
				continue
			}
			rpcT := tools.NewRPCTool(name, orig.Description(), orig.Parameters(), srv)
			deps.registry.Replace(name, rpcT)
		}
	}

	// Send ready with the tool catalogue.
	if err := srv.write(buildReadyMsg(deps, init.Model)); err != nil {
		return err
	}

	// Phase 2: message loop.
	type turnState struct {
		cancel  context.CancelFunc
		running atomic.Bool
	}
	var turn turnState

	for {
		var raw json.RawMessage
		if err := dec.Decode(&raw); err != nil {
			if errors.Is(err, io.EOF) {
				return nil
			}

			return err
		}
		var env struct {
			Type string `json:"type"`
		}
		if err := json.Unmarshal(raw, &env); err != nil {
			srv.writeError(fmt.Sprintf("malformed message: %v", err), false)

			continue
		}

		switch env.Type {
		case "user_message":
			if turn.running.Load() {
				srv.writeError("a turn is already in progress", false)

				continue
			}
			var msg rpcUserMessageMsg
			if err := json.Unmarshal(raw, &msg); err != nil {
				srv.writeError(fmt.Sprintf("invalid user_message: %v", err), false)

				continue
			}
			turnCtx, cancel := context.WithCancel(ctx)
			turn.cancel = cancel
			turn.running.Store(true)
			go func(input string) {
				defer turn.running.Store(false)
				defer cancel()
				runRPCTurn(turnCtx, deps, srv, input)
			}(msg.Content)

		case "tool_result":
			var msg rpcToolResultMsg
			if err := json.Unmarshal(raw, &msg); err != nil {
				srv.writeError(fmt.Sprintf("invalid tool_result: %v", err), false)

				continue
			}
			srv.handleToolResult(msg)

		case "approval_response":
			var msg rpcApprovalResponseMsg
			if err := json.Unmarshal(raw, &msg); err != nil {
				srv.writeError(fmt.Sprintf("invalid approval_response: %v", err), false)

				continue
			}
			srv.handleApprovalResponse(msg)

		case "cancel":
			if turn.cancel != nil {
				turn.cancel()
			}

		case "shutdown":
			if turn.cancel != nil {
				turn.cancel()
			}

			return nil

		default:
			srv.writeError(fmt.Sprintf("unknown message type %q", env.Type), false)
		}
	}
}

// buildReadyMsg builds the handshake response advertising every tool the
// agent has access to (with its JSON schema).
func buildReadyMsg(deps *sessionDeps, model string) rpcReadyMsg {
	out := rpcReadyMsg{
		Type:            "ready",
		ProtocolVersion: AgentRPCProtocolVersion,
		Model:           model,
	}
	if deps.registry == nil {
		return out
	}
	for _, t := range deps.registry.APITools() {
		out.Tools = append(out.Tools, rpcTool{
			Name:        t.Function.Name,
			Description: t.Function.Description,
			Schema:      t.Function.Parameters,
		})
	}

	return out
}

// runRPCTurn runs a single turn: scrolls + memory + agent loop, emitting
// IPC events as the turn progresses. Errors are reported via turn_done.
func runRPCTurn(ctx context.Context, deps *sessionDeps, srv *rpcServer, input string) {
	deps.mgr.Append(api.Message{Role: "user", Content: input})

	if deps.scrollsEnabled {
		matched := scrolls.Match(deps.allScrolls, input, 3)
		if len(matched) > 0 {
			var scrollMsgs []api.Message
			for _, s := range matched {
				content := fmt.Sprintf("[Scroll: %s]\n%s", s.Name, s.Content)
				scrollMsgs = append(scrollMsgs, api.Message{Role: "system", Content: content})
			}
			deps.mgr.SetScrolls(scrollMsgs)
		} else {
			deps.mgr.ClearScrolls()
		}
	}

	if deps.memoryEnabled {
		if results, err := deps.client.MemorySearch(ctx, input, 3); err == nil && len(results.Results) > 0 {
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

	if deps.mgr.NeedsSummary() {
		_ = deps.mgr.Summarize(ctx, chatctx.CompletionFunc(deps.completeFn))
	}

	windowedMsgs := deps.mgr.Messages()

	var result []api.Message
	var err error
	switch {
	case !deps.agentMode:
		// Plain streaming chat — no agent, no tools.
		result, err = runRPCSimpleTurn(ctx, deps, srv, windowedMsgs)
	case deps.swarmMode:
		cfg := buildRPCSwarmConfig(deps, srv)
		result, err = agent.RunSwarm(ctx, deps.streamFn, windowedMsgs, cfg)
	default:
		cfg := buildRPCAgentConfig(deps, srv)
		result, err = agent.RunPlannedStreaming(ctx, deps.streamFn, windowedMsgs, cfg)
	}

	if err != nil {
		_ = srv.write(rpcTurnDoneMsg{Type: "turn_done", OK: false, Reason: err.Error()})

		return
	}

	if len(result) > len(windowedMsgs) {
		newMsgs := result[len(windowedMsgs):]
		deps.mgr.AppendMany(newMsgs)
		persistRPCMemory(ctx, deps, newMsgs)
	}

	_ = srv.write(rpcTurnDoneMsg{Type: "turn_done", OK: true})
}

// runRPCSimpleTurn is the non-agent path: one streaming completion, content
// goes out as content_delta events.
func runRPCSimpleTurn(ctx context.Context, deps *sessionDeps, srv *rpcServer, messages []api.Message) ([]api.Message, error) {
	req := &api.ChatCompletionRequest{
		Model:    deps.modelName,
		Messages: messages,
		Stream:   true,
	}
	events, err := deps.client.StreamCompletion(ctx, req)
	if err != nil {
		return messages, err
	}

	content, err := streamSimpleChat(events, chatStreamHooks{
		OnContentDelta: func(delta string) {
			_ = srv.write(rpcContentDeltaMsg{Type: "content_delta", Text: delta})
		},
	})
	if err != nil {
		return messages, err
	}

	if content != "" {
		return append(messages, api.Message{Role: "assistant", Content: content}), nil
	}

	return messages, nil
}

// buildRPCAgentConfig wires the streaming agent's hooks to IPC events.
// Mirror of buildPipeAgentConfig but emitting NDJSON instead of stderr lines.
func buildRPCAgentConfig(deps *sessionDeps, srv *rpcServer) agent.PlanAgentConfig {
	return agent.PlanAgentConfig{
		StreamingConfig: agent.StreamingConfig{
			Config: agent.Config{
				MaxIterations:     deps.maxIterations,
				MaxResponseTokens: deps.maxTokens,
				EnableThinking:    deps.enableThinking,
				Tools:             deps.registry,
				Hooks: agent.Hooks{
					OnToolCall: func(call api.ToolCall) {
						// For intercepted tools, RPCTool emits tool_call_request
						// itself — skip the informational tool_call to avoid
						// duplicate events.
						if srv.interceptedSet[call.Function.Name] {
							return
						}
						_ = srv.write(rpcToolCallMsg{
							Type:      "tool_call",
							ID:        call.ID,
							Name:      call.Function.Name,
							Arguments: call.Function.Arguments,
						})
					},
					OnToolResult: func(call api.ToolCall, result *tools.ToolResult) {
						if srv.interceptedSet[call.Function.Name] {
							return
						}
						_ = srv.write(rpcToolResultLocalMsg{
							Type:    "tool_result_local",
							ID:      call.ID,
							OK:      !result.IsError,
							Content: result.Output,
						})
					},
					OnToolApproval: func(call api.ToolCall) agent.ApprovalAction {
						return srv.requestApproval(context.Background(), call)
					},
					OnAssistantMessage: func(content string) {},
				},
			},
			OnIterationStart: func(iteration, maxIter int, _ []api.Message) {
				_ = srv.write(rpcIterationStartMsg{
					Type:          "iteration_start",
					Iteration:     iteration + 1,
					MaxIterations: maxIter,
				})
			},
			OnContentDelta: func(delta string) {
				_ = srv.write(rpcContentDeltaMsg{Type: "content_delta", Text: delta})
			},
			OnReasoningDelta: func(delta string) {
				_ = srv.write(rpcContentDeltaMsg{Type: "reasoning_delta", Text: delta})
			},
		},
	}
}

// buildRPCSwarmConfig is the swarm equivalent. Swarm-specific events
// (architect spec, plan, worker start/done, verify) are folded into
// content_delta lines for v1; a richer protocol for swarm UI lands in v1.1.
func buildRPCSwarmConfig(deps *sessionDeps, srv *rpcServer) agent.SwarmConfig {
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
						if srv.interceptedSet[call.Function.Name] {
							return
						}
						_ = srv.write(rpcToolCallMsg{
							Type:      "tool_call",
							ID:        call.ID,
							Name:      call.Function.Name,
							Arguments: call.Function.Arguments,
						})
					},
					OnToolResult: func(call api.ToolCall, result *tools.ToolResult) {
						if srv.interceptedSet[call.Function.Name] {
							return
						}
						_ = srv.write(rpcToolResultLocalMsg{
							Type:    "tool_result_local",
							ID:      call.ID,
							OK:      !result.IsError,
							Content: result.Output,
						})
					},
					OnToolApproval: func(call api.ToolCall) agent.ApprovalAction {
						return srv.requestApproval(context.Background(), call)
					},
					OnAssistantMessage: func(content string) {},
				},
			},
			OnIterationStart: func(iteration, maxIter int, _ []api.Message) {
				_ = srv.write(rpcIterationStartMsg{
					Type:          "iteration_start",
					Iteration:     iteration + 1,
					MaxIterations: maxIter,
				})
			},
			OnContentDelta: func(delta string) {
				_ = srv.write(rpcContentDeltaMsg{Type: "content_delta", Text: delta})
			},
			OnReasoningDelta: func(delta string) {
				_ = srv.write(rpcContentDeltaMsg{Type: "reasoning_delta", Text: delta})
			},
		},
		WorkerTools:   workerTools,
		ArchitectFile: filepath.Join(".tanrenai", "architect.md"),
		OnPlanGenerated: func(depth int, plan *agent.Plan) {
			var sb strings.Builder
			fmt.Fprintf(&sb, "[swarm plan d=%d]\n", depth)
			for _, step := range plan.Steps {
				fmt.Fprintf(&sb, "  %d. %s\n", step.Index, step.Description)
			}
			_ = srv.write(rpcContentDeltaMsg{Type: "content_delta", Text: sb.String()})
		},
		OnWorkerStart: func(depth, _ int, step *agent.PlanStep) {
			_ = srv.write(rpcContentDeltaMsg{
				Type: "content_delta",
				Text: fmt.Sprintf("[swarm worker d=%d %d] %s\n", depth, step.Index, step.Description),
			})
		},
		OnWorkerDone: func(depth, _ int, step *agent.PlanStep) {
			_ = srv.write(rpcContentDeltaMsg{
				Type: "content_delta",
				Text: fmt.Sprintf("[swarm done d=%d %d] %s\n", depth, step.Index, step.Status.String()),
			})
		},
	}
}

// persistRPCMemory mirrors persistPipeMemory.
func persistRPCMemory(ctx context.Context, deps *sessionDeps, newMsgs []api.Message) {
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
		slog.Error("agent-rpc: failed to store memory", "error", err)
	}
}
