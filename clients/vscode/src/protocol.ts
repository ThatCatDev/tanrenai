// Shared types for the host ↔ webview message protocol. Imported by both
// src/* (extension host) and webview/* (browser bundle).

export interface ProgressLine {
  message: string;
  level: 'info' | 'warn';
}

export type Mode = 'chat' | 'agent' | 'swarm';

export type ConnectionState =
  | { status: 'idle' }
  | { status: 'no_credentials' }
  | { status: 'connecting'; progress: ProgressLine[] }
  | { status: 'connected'; model: string; toolCount: number }
  | { status: 'error'; message: string };

export type WebviewInbound =
  | { type: 'send'; content: string; attachments?: SelectionAttachment[] }
  | { type: 'attach_request' }
  | { type: 'attach_clear' }
  | { type: 'cancel' }
  | { type: 'cancel_connect' }
  | { type: 'pick_model' }
  | { type: 'clear_chat' }
  | { type: 'set_mode'; mode: Mode }
  | { type: 'approval_decision'; id: string; action: 'allow' | 'deny' | 'always' }
  | { type: 'login' }
  | { type: 'logout' }
  | { type: 'reconnect' };

export interface SelectionAttachment {
  /** Display label, e.g. "src/foo.ts:12-34". */
  label: string;
  /** Workspace-relative or absolute path. */
  path: string;
  /** Detected language id (e.g. "typescript"). */
  languageId: string;
  /** 1-indexed line range. */
  startLine: number;
  endLine: number;
  /** The selected text. */
  text: string;
}

export type WebviewOutbound =
  | { type: 'state'; state: ConnectionState }
  | { type: 'turn_start' }
  | { type: 'turn_end'; ok: boolean; reason?: string }
  | { type: 'iteration_start'; iteration: number; maxIterations: number }
  | { type: 'message_start'; role: 'user' | 'assistant' | 'tool'; id: string; meta?: string }
  | { type: 'message_delta'; id: string; text: string; channel?: 'content' | 'reasoning' }
  | { type: 'message_end'; id: string }
  | { type: 'tool_call'; id: string; name: string; arguments: string; intercepted: boolean }
  | { type: 'tool_call_streaming'; index: number; name: string; argsDelta: string }
  | { type: 'tool_result'; id: string; ok: boolean; content?: string }
  | { type: 'approval_required'; id: string; name: string; arguments: string }
  | { type: 'approval_resolved'; id: string }
  | { type: 'clear_chat' }
  | { type: 'mode'; mode: Mode }
  | { type: 'attach_selection'; selection: SelectionAttachment };
