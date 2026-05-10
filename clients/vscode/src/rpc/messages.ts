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
  | ShutdownMsg;

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

export interface ErrorMsg {
  type: 'error';
  message: string;
  fatal: boolean;
}

export type InboundMsg =
  | ReadyMsg
  | ContentDeltaMsg
  | ReasoningDeltaMsg
  | IterationStartMsg
  | ToolCallMsg
  | ToolCallRequestMsg
  | ToolResultLocalMsg
  | ApprovalRequiredMsg
  | TurnDoneMsg
  | ErrorMsg;
