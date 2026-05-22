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

export type OutboundMsg =
  | InitMsg
  | UserMessageMsg
  | ToolResultMsg
  | ApprovalResponseMsg
  | CancelMsg
  | ShutdownMsg
  | ClearHistoryMsg;

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
  | SwarmArchitectMsg
  | SwarmPlanMsg
  | SwarmWorkerStartMsg
  | SwarmWorkerDoneMsg
  | SwarmVerifyMsg
  | ErrorMsg;
