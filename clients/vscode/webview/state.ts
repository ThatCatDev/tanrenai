// State model the App reduces from inbound messages.

import type { ConnectionState, Mode, WebviewOutbound } from '../src/protocol';

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

export interface ErrorMsg {
  kind: 'error';
  id: string;
  text: string;
}

export type Entry = AssistantMsg | UserMsg | ToolMsg | ErrorMsg;

export interface AppState {
  connection: ConnectionState;
  mode: Mode;
  entries: Entry[];
  turnRunning: boolean;
  iteration: number; // 0 when no turn or pre-iteration
  maxIterations: number;
}

export const initialState: AppState = {
  connection: { status: 'idle' },
  mode: 'agent',
  entries: [],
  turnRunning: false,
  iteration: 0,
  maxIterations: 0,
};

export type Activity =
  | { kind: 'idle' }
  | { kind: 'thinking' }
  | { kind: 'generating' }
  | { kind: 'tool'; name: string };

/** Derive what the agent is currently doing from observable state. */
export function deriveActivity(state: AppState): Activity {
  if (!state.turnRunning) {
    return { kind: 'idle' };
  }
  // Walk entries from newest backwards.
  for (let i = state.entries.length - 1; i >= 0; i--) {
    const e = state.entries[i];
    if (e.kind === 'tool' && !e.result) {
      return { kind: 'tool', name: e.name };
    }
    if (e.kind === 'assistant' && e.open) {
      // Reasoning got more text and content is empty → still thinking.
      // Content has any chars → generating.
      if (e.content.length > 0) {
        return { kind: 'generating' };
      }

      return { kind: 'thinking' };
    }
  }

  return { kind: 'thinking' };
}

/**
 * Apply a single inbound message. Returns the next state. Pure — caller
 * sets it onto a useState/signal.
 */
export function reduce(state: AppState, msg: WebviewOutbound): AppState {
  switch (msg.type) {
    case 'state':
      return { ...state, connection: msg.state };
    case 'mode':
      return { ...state, mode: msg.mode };
    case 'turn_start':
      return { ...state, turnRunning: true, iteration: 0, maxIterations: 0 };
    case 'turn_end': {
      const entries = msg.ok || !msg.reason
        ? closeOpenAssistants(state.entries)
        : [
            ...closeOpenAssistants(state.entries),
            { kind: 'error' as const, id: `e_${Date.now()}`, text: msg.reason },
          ];

      return { ...state, turnRunning: false, entries, iteration: 0, maxIterations: 0 };
    }
    case 'iteration_start':
      return { ...state, iteration: msg.iteration, maxIterations: msg.maxIterations };
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
      return appendOrUpdate(state, {
        kind: 'tool',
        id: msg.id,
        name: msg.name,
        args: msg.arguments,
        intercepted: msg.intercepted,
      });
    case 'tool_result': {
      const entries = state.entries.map((e) =>
        e.id === msg.id && e.kind === 'tool'
          ? { ...e, result: { ok: msg.ok, content: msg.content } }
          : e,
      );

      return { ...state, entries };
    }
    case 'clear_chat':
      return { ...state, entries: [] };
    default:
      return state;
  }
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
