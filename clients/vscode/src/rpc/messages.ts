// IPC message types for the agent-rpc protocol. Mirrors the Go-side
// definitions in clients/cli/cmd/agentrpc.go. Bump PROTOCOL_VERSION when
// changing the wire format.

export const PROTOCOL_VERSION = 1;

// ── Outbound (extension → CLI) ────────────────────────────────────────

export interface InitMsg {
  type: 'init';
  protocolVersion: number;
  model: string;
  agentMode: boolean;
  swarmMode: boolean;
  enableMemory: boolean;
  enableScrolls: boolean;
  interceptedTools: string[];
  workspaceRoot: string;
  maxIterations: number;
  systemPrompt: string;
}

export interface UserMessageMsg {
  type: 'user_message';
  content: string;
  /** Optional per-turn mode override; empty/undefined = use init's mode. */
  mode?: 'chat' | 'agent' | 'swarm';
  /** Optional image attachments as data: / http(s) URLs. Sent to the
   *  model as multimodal content_parts. Requires a vision-capable model. */
  images?: string[];
}

export interface ClearHistoryMsg {
  type: 'clear_history';
}

export interface ToolResultMsg {
  type: 'tool_result';
  id: string;
  ok: boolean;
  content?: string;
  error?: string;
}

export interface ApprovalResponseMsg {
  type: 'approval_response';
  id: string;
  action: 'allow' | 'deny' | 'always';
}

export interface CancelMsg {
  type: 'cancel';
}

export interface ShutdownMsg {
  type: 'shutdown';
}

/** Manual compact request from the VS Code Footer menu. CLI runs the same
 *  Summarize path the auto-compactor uses and emits matching compaction
 *  events; the ack lets the GUI know success/failure for a toast. */
export interface CompactRequestMsg {
  type: 'compact_request';
  requestId: string;
}

export interface ContextListReqMsg {
  type: 'context_list';
  requestId: string;
}

export interface ContextAddReqMsg {
  type: 'context_add';
  requestId: string;
  path: string;
}

export interface ContextClearReqMsg {
  type: 'context_clear';
  requestId: string;
}

export interface MemoryListReqMsg {
  type: 'memory_list';
  requestId: string;
  limit: number;
}

export interface MemorySearchReqMsg {
  type: 'memory_search';
  requestId: string;
  query: string;
  limit: number;
}

export interface MemoryForgetReqMsg {
  type: 'memory_forget';
  requestId: string;
  id: string;
}

export interface MemoryClearReqMsg {
  type: 'memory_clear';
  requestId: string;
}

export interface ScrollsListReqMsg {
  type: 'scrolls_list';
  requestId: string;
}

export interface ScrollsShowReqMsg {
  type: 'scrolls_show';
  requestId: string;
  name: string;
}

export type OutboundMsg =
  | InitMsg
  | UserMessageMsg
  | ToolResultMsg
  | ApprovalResponseMsg
  | CancelMsg
  | ShutdownMsg
  | ClearHistoryMsg
  | CompactRequestMsg
  | ContextListReqMsg
  | ContextAddReqMsg
  | ContextClearReqMsg
  | MemoryListReqMsg
  | MemorySearchReqMsg
  | MemoryForgetReqMsg
  | MemoryClearReqMsg
  | ScrollsListReqMsg
  | ScrollsShowReqMsg;

// ── Inbound (CLI → extension) ─────────────────────────────────────────

export interface ToolDescriptor {
  name: string;
  description: string;
  schema: unknown;
}

export interface ReadyMsg {
  type: 'ready';
  protocolVersion: number;
  tools: ToolDescriptor[];
  model: string;
}

export interface ConnectingProgressMsg {
  type: 'connecting_progress';
  level: 'info' | 'warn';
  message: string;
}

export interface HistoryClearedMsg {
  type: 'history_cleared';
}

export interface ContentDeltaMsg {
  type: 'content_delta';
  text: string;
}

export interface ReasoningDeltaMsg {
  type: 'reasoning_delta';
  text: string;
}

export interface IterationStartMsg {
  type: 'iteration_start';
  iteration: number;
  maxIterations: number;
}

export interface ToolCallMsg {
  type: 'tool_call';
  id: string;
  name: string;
  arguments: string;
}

export interface ToolCallRequestMsg {
  type: 'tool_call_request';
  id: string;
  name: string;
  arguments: string;
}

export interface ToolResultLocalMsg {
  type: 'tool_result_local';
  id: string;
  ok: boolean;
  content?: string;
  error?: string;
}

export interface ApprovalRequiredMsg {
  type: 'approval_required';
  id: string;
  name: string;
  arguments: string;
}

export interface TurnDoneMsg {
  type: 'turn_done';
  ok: boolean;
  reason?: string;
}

/**
 * Streamed throughput update for the current turn. The CLI sends one of
 * these every ~500ms while content/reasoning deltas are arriving, plus a
 * final one just before turn_done so the UI can land on a stable number.
 * `tokens` is the count of streamed deltas; `tps` is the rate computed
 * over the window between the first and last delta (excludes prompt-eval
 * latency, so it reflects pure generation speed).
 */
export interface TokenRateMsg {
  type: 'token_rate';
  tokens: number;
  tps: number;
}

// ── Swarm events ──────────────────────────────────────────────────────
// Structured progress events emitted by the swarm orchestrator. The
// webview reduces these into a SwarmActivity entry per depth and renders
// the plan as a step list with live status; see webview/state.ts and
// components/SwarmActivity.tsx. Pre-v2 these were folded into
// content_delta strings which left no rendering hook.

export interface SwarmPlanStep {
  index: number;
  description: string;
}

export interface SwarmArchitectMsg {
  type: 'swarm_architect';
  depth: number;
  spec: string;
}

export interface SwarmPlanMsg {
  type: 'swarm_plan';
  depth: number;
  steps: SwarmPlanStep[];
}

export interface SwarmWorkerStartMsg {
  type: 'swarm_worker_start';
  depth: number;
  stepIndex: number;
  description: string;
}

export interface SwarmWorkerDoneMsg {
  type: 'swarm_worker_done';
  depth: number;
  stepIndex: number;
  /** agent.StepStatus stringified — typically "done", "error", "skipped". */
  status: string;
  result?: string;
  error?: string;
}

export interface SwarmVerifyMsg {
  type: 'swarm_verify';
  depth: number;
}

export interface ErrorMsg {
  type: 'error';
  message: string;
  fatal: boolean;
}

/**
 * Snapshot of the prompt budget. CLI emits one on `ready`, after every
 * `turn_done`, after `clear_history`, and after any GUI op that mutates
 * pinned context. Mirrors chatctx.BudgetInfo on the Go side. The webview
 * footer shows `used / total` plus a popover with the full breakdown.
 */
export interface ContextUsageMsg {
  type: 'context_usage';
  total: number;
  system: number;
  scrolls: number;
  memory: number;
  summary: number;
  history: number;
  available: number;
  historyCount: number;
  totalHistory: number;
}

/**
 * Auto-compaction lifecycle event. Phase progresses `start` → `done` |
 * `error`. The webview shows a transient banner while a `start` is open
 * and inserts a persistent transcript divider at each phase.
 */
export interface CompactionMsg {
  type: 'compaction';
  /** `noop` means there was nothing to compact (NeedsSummary was false
   *  or the eviction cutoff found no candidates). Distinct from `done`
   *  so the UI doesn't say "Compacted 0 messages into summary". */
  phase: 'start' | 'done' | 'error' | 'noop';
  messages?: number;
  error?: string;
}

// ── Reply envelopes for GUI ops ───────────────────────────────────────
// Each reply carries the `requestId` from the original request so the
// extension can resolve the right pending promise.

export interface ContextFilesMsg {
  type: 'context_files';
  requestId: string;
  files: string[];
}

export interface MemoryRow {
  id: string;
  userMsg: string;
  assistMsg: string;
  timestamp: string;
}

export interface MemoryListReplyMsg {
  type: 'memory_list_reply';
  requestId: string;
  entries: MemoryRow[];
  total: number;
}

export interface MemorySearchResult {
  entry: MemoryRow;
  combinedScore: number;
}

export interface MemorySearchReplyMsg {
  type: 'memory_search_reply';
  requestId: string;
  results: MemorySearchResult[];
}

export interface ScrollRow {
  name: string;
  description: string;
  source: string;
}

export interface ScrollsListReplyMsg {
  type: 'scrolls_list_reply';
  requestId: string;
  scrolls: ScrollRow[];
}

export interface ScrollReplyMsg {
  type: 'scroll_reply';
  requestId: string;
  name: string;
  description: string;
  content: string;
  source?: string;
}

/** Generic ack for ops that don't return data (clear, forget, compact). */
export interface AckMsg {
  type: 'ack';
  requestId: string;
  op: string;
  ok: boolean;
  error?: string;
}

export type InboundMsg =
  | ReadyMsg
  | ConnectingProgressMsg
  | HistoryClearedMsg
  | ContentDeltaMsg
  | ReasoningDeltaMsg
  | IterationStartMsg
  | ToolCallMsg
  | ToolCallRequestMsg
  | ToolResultLocalMsg
  | ApprovalRequiredMsg
  | TurnDoneMsg
  | TokenRateMsg
  | ContextUsageMsg
  | CompactionMsg
  | SwarmArchitectMsg
  | SwarmPlanMsg
  | SwarmWorkerStartMsg
  | SwarmWorkerDoneMsg
  | SwarmVerifyMsg
  | ErrorMsg
  | ContextFilesMsg
  | MemoryListReplyMsg
  | MemorySearchReplyMsg
  | ScrollsListReplyMsg
  | ScrollReplyMsg
  | AckMsg;
