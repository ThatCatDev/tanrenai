// Bridge to the VS Code extension host via postMessage.

import type { WebviewInbound, WebviewOutbound } from '../src/protocol';

interface VSCodeApi {
  postMessage(msg: unknown): void;
  getState(): unknown;
  setState(state: unknown): unknown;
}

declare const acquireVsCodeApi: () => VSCodeApi;

const api = acquireVsCodeApi();

export function send(msg: WebviewInbound): void {
  api.postMessage(msg);
}

export function onMessage(handler: (msg: WebviewOutbound) => void): () => void {
  const listener = (event: MessageEvent): void => {
    const msg = event.data as WebviewOutbound;
    if (!msg || typeof msg.type !== 'string') {
      return;
    }
    handler(msg);
  };
  window.addEventListener('message', listener);

  return () => window.removeEventListener('message', listener);
}

/**
 * Persisted state survives webview remount (sidebar hide → show). Used to
 * paint the chat shell immediately instead of flashing the idle/connecting
 * panel while the host pushes the current state. Only stable visuals
 * (connection + mode) are persisted — chat entries are replayed by the
 * controller's transcript on remount.
 */
export interface PersistedShell {
  connection: import('../src/protocol').ConnectionState;
  mode: import('../src/protocol').Mode;
}

export function getPersistedShell(): PersistedShell | undefined {
  const raw = api.getState() as PersistedShell | undefined;
  if (!raw || typeof raw !== 'object' || !('connection' in raw) || !('mode' in raw)) {
    return undefined;
  }

  return raw;
}

export function setPersistedShell(shell: PersistedShell): void {
  api.setState(shell);
}
