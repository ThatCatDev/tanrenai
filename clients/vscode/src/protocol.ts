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
  | {
      type: 'send';
      content: string;
      attachments?: SelectionAttachment[];
      images?: ImageAttachment[];
    }
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
  | { type: 'reconnect' }
  | { type: 'stop_gpu' }
  | { type: 'destroy_gpu' }
  | { type: 'show_gpu_status' };

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

export interface ImageAttachment {
  /** Display label, e.g. "screenshot.png". */
  label: string;
  /** MIME type, e.g. "image/png". */
  mimeType: string;
  /** Full data URL: `data:<mime>;base64,<payload>` — sent to the model. */
  dataUrl: string;
  /** Byte size (raw, not the base64-encoded length). */
  size: number;
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
  | { type: 'attach_selection'; selection: SelectionAttachment }
  /** Live preview of the active editor's selection (or null when there is none).
   *  Webview renders a faint hint above the composer; user clicks to attach. */
  | { type: 'available_selection'; selection: SelectionAttachment | null }
  /** Streamed throughput readout for the current turn. The host throttles
   *  these to one per ~500ms; `tps` is the rate over the window between
   *  the first and last delta (excludes prompt-eval latency). Webview
   *  surfaces it in the status panel during streaming and on the final
   *  emitted value after the turn closes. */
  | { type: 'token_rate'; tokens: number; tps: number }
  /** Swarm orchestrator lifecycle events. Forwarded as-is from the CLI;
   *  the webview reducer turns them into a per-depth SwarmActivity entry
   *  rendered as a step list with live status updates. */
  | { type: 'swarm_architect'; depth: number; spec: string }
  | {
      type: 'swarm_plan';
      depth: number;
      steps: Array<{ index: number; description: string }>;
    }
  | { type: 'swarm_worker_start'; depth: number; stepIndex: number; description: string }
  | {
      type: 'swarm_worker_done';
      depth: number;
      stepIndex: number;
      status: string;
      result?: string;
      error?: string;
    }
  | { type: 'swarm_verify'; depth: number };
