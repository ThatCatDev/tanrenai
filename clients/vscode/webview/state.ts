// State model the App reduces from inbound messages.

import type {
  ConnectionState,
  ImageAttachment,
  Mode,
  SelectionAttachment,
  WebviewOutbound,
} from '../src/protocol';

export interface AssistantMsg {
  kind: 'assistant';
  id: string;
  content: string;
  reasoning: string;
  /** True while deltas are still arriving for this entry. */
  open: boolean;
}

export interface UserMsg {
  kind: 'user';
  id: string;
  content: string;
}

export interface ToolMsg {
  kind: 'tool';
  id: string;
  name: string;
  args: string;
  intercepted: boolean;
  result?: { ok: boolean; content?: string };
}

export interface ApprovalMsg {
  kind: 'approval';
  id: string;
  name: string;
  args: string;
  resolved: boolean;
}

export interface ErrorMsg {
  kind: 'error';
  id: string;
  text: string;
}

/** A single step within a swarm plan. Status transitions:
 *  `pending` (after swarm_plan) → `running` (after swarm_worker_start) →
 *  `done` | `error` (after swarm_worker_done). */
export interface SwarmStep {
  index: number;
  description: string;
  status: 'pending' | 'running' | 'done' | 'error' | string;
  result?: string;
  error?: string;
}

/** Swarm orchestrator activity for a single nesting depth. One entry
 *  per depth per turn — depth 0 is the top-level orchestrator, depth>0
 *  is a child swarm spawned by a worker. */
export interface SwarmActivityMsg {
  kind: 'swarm';
  id: string;
  depth: number;
  /** Architecture spec the orchestrator loaded/produced for this depth.
   *  Optional — only present when the agent emits one. */
  architectSpec?: string;
  steps: SwarmStep[];
  /** Set after swarm_verify lands; UI surfaces a "verifying" indicator. */
  verifying: boolean;
}

export type Entry =
  | AssistantMsg
  | UserMsg
  | ToolMsg
  | ApprovalMsg
  | ErrorMsg
  | SwarmActivityMsg;

/** A tool call still being streamed by the model — not yet finalised. */
export interface StreamingTool {
  index: number;
  name: string;
  argsLength: number;
}

/** Throughput readout for the current (or most recent) turn. Updated as
 *  the CLI streams `token_rate` messages. `tokens` accumulates while the
 *  turn is live; we keep both values around after `turn_end` so the panel
 *  can show the final rate until the next turn starts. */
export interface TokenRate {
  tokens: number;
  tps: number;
}

export interface AppState {
  connection: ConnectionState;
  mode: Mode;
  entries: Entry[];
  turnRunning: boolean;
  iteration: number; // 0 when no turn or pre-iteration
  maxIterations: number;
  streamingTools: StreamingTool[];
  /** Selection attachments queued for the next user message. */
  pendingAttachments: SelectionAttachment[];
  /** Image attachments queued for the next user message. */
  pendingImages: ImageAttachment[];
  /** Live preview of the active editor's selection (null = no selection). */
  availableSelection: SelectionAttachment | null;
  /** Most recent throughput reading (null until the first delta). */
  tokenRate: TokenRate | null;
}

export const initialState: AppState = {
  connection: { status: 'idle' },
  mode: 'agent',
  entries: [],
  turnRunning: false,
  iteration: 0,
  maxIterations: 0,
  streamingTools: [],
  pendingAttachments: [],
  pendingImages: [],
  availableSelection: null,
  tokenRate: null,
};

/** Return the swarm activity that should pin to the dock, or null if
 *  there isn't one. We pick the deepest active entry — when a worker
 *  spawns a child swarm at depth+1, the user wants to see *that* one
 *  ticking, not the parent waiting. Falls back to the most recent
 *  swarm entry overall so the dock stays useful after turn_end (the
 *  user can still see what just finished).
 */
export function activeSwarm(state: AppState): SwarmActivityMsg | null {
  const swarms = state.entries.filter(
    (e): e is SwarmActivityMsg => e.kind === 'swarm',
  );
  if (swarms.length === 0) return null;
  // Prefer one with a currently-running step at the deepest depth.
  const running = swarms.filter((s) => s.steps.some((step) => step.status === 'running'));
  if (running.length > 0) {
    return running.reduce((a, b) => (b.depth > a.depth ? b : a));
  }
  // No running step — show the most recent (last in entries) so the
  // dock surfaces "just finished" until the next turn starts.
  return swarms[swarms.length - 1];
}

export type Activity =
  | { kind: 'idle' }
  | { kind: 'thinking' }
  | { kind: 'generating' }
  | { kind: 'preparing'; name: string; chars: number }
  | { kind: 'tool'; name: string }
  | { kind: 'awaiting_approval'; name: string };

/** Derive what the agent is currently doing from observable state. */
export function deriveActivity(state: AppState): Activity {
  if (!state.turnRunning) {
    return { kind: 'idle' };
  }
  // Pending approval is the most blocking — surface first.
  for (let i = state.entries.length - 1; i >= 0; i--) {
    const e = state.entries[i];
    if (e.kind === 'approval' && !e.resolved) {
      return { kind: 'awaiting_approval', name: e.name };
    }
  }
  // Tool currently executing.
  for (let i = state.entries.length - 1; i >= 0; i--) {
    const e = state.entries[i];
    if (e.kind === 'tool' && !e.result) {
      return { kind: 'tool', name: e.name };
    }
    if (e.kind === 'assistant' && e.open) {
      if (e.content.length > 0) {
        return { kind: 'generating' };
      }

      break;
    }
  }
  // If the model is mid-tool-call streaming, surface that — was the
  // dead-air case before this hook existed.
  if (state.streamingTools.length > 0) {
    const last = state.streamingTools[state.streamingTools.length - 1];

    return { kind: 'preparing', name: last.name || '…', chars: last.argsLength };
  }
  // Otherwise: still in an open assistant bubble (reasoning) or pre-iteration.
  for (let i = state.entries.length - 1; i >= 0; i--) {
    const e = state.entries[i];
    if (e.kind === 'assistant' && e.open) {
      return { kind: 'thinking' };
    }
  }

  return { kind: 'thinking' };
}

/**
 * Internal actions dispatched by the webview itself (not the host) — for
 * UI-only state transitions like removing an attachment chip.
 */
export type InternalAction =
  | { type: 'attach_remove'; index: number }
  | { type: 'attach_clear_pending' }
  | { type: 'image_attach'; image: ImageAttachment }
  | { type: 'image_remove'; index: number }
  | { type: 'image_clear_pending' };

export type Action = WebviewOutbound | InternalAction;

/**
 * Apply a single inbound action. Returns the next state. Pure — caller
 * sets it onto a useState/signal.
 */
export function reduce(state: AppState, msg: Action): AppState {
  switch (msg.type) {
    case 'state':
      return { ...state, connection: msg.state };
    case 'mode':
      return { ...state, mode: msg.mode };
    case 'turn_start':
      return {
        ...state,
        turnRunning: true,
        iteration: 0,
        maxIterations: 0,
        streamingTools: [],
        tokenRate: null,
      };
    case 'token_rate':
      return {
        ...state,
        tokenRate: { tokens: msg.tokens, tps: msg.tps },
      };
    case 'turn_end': {
      const entries = msg.ok || !msg.reason
        ? closeOpenAssistants(state.entries)
        : [
            ...closeOpenAssistants(state.entries),
            { kind: 'error' as const, id: `e_${Date.now()}`, text: msg.reason },
          ];

      return {
        ...state,
        turnRunning: false,
        entries,
        iteration: 0,
        maxIterations: 0,
        streamingTools: [],
      };
    }
    case 'iteration_start':
      return {
        ...state,
        iteration: msg.iteration,
        maxIterations: msg.maxIterations,
        streamingTools: [],
      };
    case 'message_start': {
      if (msg.role === 'user') {
        return appendOrUpdate(state, {
          kind: 'user',
          id: msg.id,
          content: '',
        });
      }
      // 'assistant' (or 'tool', currently unused)
      return appendOrUpdate(state, {
        kind: 'assistant',
        id: msg.id,
        content: '',
        reasoning: '',
        open: true,
      });
    }
    case 'message_delta': {
      const entries = state.entries.map((e) => {
        if (e.id !== msg.id) return e;
        if (e.kind === 'user') {
          return { ...e, content: e.content + msg.text };
        }
        if (e.kind === 'assistant') {
          if (msg.channel === 'reasoning') {
            return { ...e, reasoning: e.reasoning + msg.text };
          }

          return { ...e, content: e.content + msg.text };
        }

        return e;
      });
      // Synthesize an entry if delta arrived without start.
      if (!entries.some((e) => e.id === msg.id)) {
        const created: AssistantMsg = {
          kind: 'assistant',
          id: msg.id,
          content: msg.channel === 'reasoning' ? '' : msg.text,
          reasoning: msg.channel === 'reasoning' ? msg.text : '',
          open: true,
        };

        return { ...state, entries: [...entries, created] };
      }

      return { ...state, entries };
    }
    case 'message_end': {
      const entries = state.entries.map((e) =>
        e.id === msg.id && e.kind === 'assistant' ? { ...e, open: false } : e,
      );

      return { ...state, entries };
    }
    case 'tool_call':
      // Tool call finalised — clear any matching streaming tool.
      return appendOrUpdate(
        { ...state, streamingTools: state.streamingTools.filter((t) => t.name !== msg.name) },
        {
          kind: 'tool',
          id: msg.id,
          name: msg.name,
          args: msg.arguments,
          intercepted: msg.intercepted,
        },
      );
    case 'tool_call_streaming': {
      // Track in-progress tool args by index. Update name once known,
      // accumulate args length so the activity bar can show "preparing
      // file_write (1.2k chars)…".
      const existing = state.streamingTools.find((t) => t.index === msg.index);
      const next: StreamingTool[] = existing
        ? state.streamingTools.map((t) =>
            t.index === msg.index
              ? { ...t, name: msg.name || t.name, argsLength: t.argsLength + msg.argsDelta.length }
              : t,
          )
        : [...state.streamingTools, { index: msg.index, name: msg.name, argsLength: msg.argsDelta.length }];

      return { ...state, streamingTools: next };
    }
    case 'tool_result': {
      const entries = state.entries.map((e) =>
        e.id === msg.id && e.kind === 'tool'
          ? { ...e, result: { ok: msg.ok, content: msg.content } }
          : e,
      );

      return { ...state, entries };
    }
    case 'approval_required':
      return appendOrUpdate(state, {
        kind: 'approval',
        id: msg.id,
        name: msg.name,
        args: msg.arguments,
        resolved: false,
      });
    case 'approval_resolved': {
      const entries = state.entries.map((e) =>
        e.kind === 'approval' && e.id === msg.id ? { ...e, resolved: true } : e,
      );

      return { ...state, entries };
    }
    case 'clear_chat':
      return { ...state, entries: [], streamingTools: [] };
    case 'attach_selection':
      // Don't add an exact duplicate of an existing pending attachment.
      if (
        state.pendingAttachments.some(
          (a) => a.path === msg.selection.path && a.text === msg.selection.text,
        )
      ) {
        return state;
      }

      return {
        ...state,
        pendingAttachments: [...state.pendingAttachments, msg.selection],
      };
    case 'available_selection':
      return { ...state, availableSelection: msg.selection };
    case 'attach_remove':
      return {
        ...state,
        pendingAttachments: state.pendingAttachments.filter((_, i) => i !== msg.index),
      };
    case 'attach_clear_pending':
      return { ...state, pendingAttachments: [] };
    case 'image_attach':
      return { ...state, pendingImages: [...state.pendingImages, msg.image] };
    case 'image_remove':
      return {
        ...state,
        pendingImages: state.pendingImages.filter((_, i) => i !== msg.index),
      };
    case 'image_clear_pending':
      return { ...state, pendingImages: [] };

    // ── Swarm events ────────────────────────────────────────────────
    // Each depth gets one SwarmActivity entry per turn. We use a stable
    // id derived from depth so worker-start/done events can locate and
    // mutate the right plan, and so a re-emitted plan (rare, but the
    // agent can replan) updates in place rather than appending a dupe.
    case 'swarm_architect':
      return upsertSwarm(state, msg.depth, (a) => ({ ...a, architectSpec: msg.spec }));
    case 'swarm_plan':
      return upsertSwarm(state, msg.depth, (a) => ({
        ...a,
        steps: msg.steps.map((s) => ({
          index: s.index,
          description: s.description,
          status: 'pending' as const,
        })),
      }));
    case 'swarm_worker_start':
      return upsertSwarm(state, msg.depth, (a) => ({
        ...a,
        steps: updateStep(a.steps, msg.stepIndex, (s) => ({ ...s, status: 'running' })),
      }));
    case 'swarm_worker_done':
      return upsertSwarm(state, msg.depth, (a) => ({
        ...a,
        steps: updateStep(a.steps, msg.stepIndex, (s) => ({
          ...s,
          // Trust whatever string the agent reported; only normalise the
          // happy/error cases the UI styles. Anything else (skipped,
          // cancelled, etc.) passes through and renders as plain text.
          status:
            msg.status === 'done' || msg.status === 'error'
              ? msg.status
              : (msg.status as SwarmStep['status']),
          result: msg.result,
          error: msg.error,
        })),
      }));
    case 'swarm_verify':
      return upsertSwarm(state, msg.depth, (a) => ({ ...a, verifying: true }));

    default:
      return state;
  }
}

/** Update the SwarmActivity entry for a depth (creating it if missing).
 *  Per-depth identity means worker-start/done events landing before the
 *  plan event still find a parent to attach to — handy because plan
 *  generation and the first worker_start can race on the agent side. */
function upsertSwarm(
  state: AppState,
  depth: number,
  mutate: (a: SwarmActivityMsg) => SwarmActivityMsg,
): AppState {
  const id = `swarm_d${depth}`;
  const idx = state.entries.findIndex((e) => e.kind === 'swarm' && e.id === id);
  const base: SwarmActivityMsg =
    idx === -1
      ? { kind: 'swarm', id, depth, steps: [], verifying: false }
      : (state.entries[idx] as SwarmActivityMsg);
  const next = mutate(base);
  if (idx === -1) {
    return { ...state, entries: [...state.entries, next] };
  }
  const entries = state.entries.slice();
  entries[idx] = next;

  return { ...state, entries };
}

/** Replace the first step matching `stepIndex`, or append if missing.
 *  Append happens when worker_start lands before swarm_plan — we still
 *  want to render the in-flight worker rather than drop the event. */
function updateStep(
  steps: SwarmStep[],
  stepIndex: number,
  mutate: (s: SwarmStep) => SwarmStep,
): SwarmStep[] {
  const idx = steps.findIndex((s) => s.index === stepIndex);
  if (idx === -1) {
    return [
      ...steps,
      mutate({ index: stepIndex, description: '', status: 'pending' }),
    ];
  }
  const next = steps.slice();
  next[idx] = mutate(next[idx]);

  return next;
}

function appendOrUpdate(state: AppState, entry: Entry): AppState {
  const idx = state.entries.findIndex((e) => e.id === entry.id);
  if (idx === -1) {
    return { ...state, entries: [...state.entries, entry] };
  }
  const entries = state.entries.slice();
  entries[idx] = entry;

  return { ...state, entries };
}

function closeOpenAssistants(entries: Entry[]): Entry[] {
  return entries.map((e) => (e.kind === 'assistant' && e.open ? { ...e, open: false } : e));
}
