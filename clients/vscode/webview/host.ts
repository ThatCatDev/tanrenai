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

// Register the message listener at module load — BEFORE Preact mounts —
// so messages the host posts while the bundle is still booting (state,
// transcript replay, mode) are not lost. They're buffered until a real
// handler attaches via onMessage(); then drained in order.
let activeHandler: ((msg: WebviewOutbound) => void) | undefined;
let buffered: WebviewOutbound[] = [];

window.addEventListener('message', (event: MessageEvent) => {
  const msg = event.data as WebviewOutbound;
  if (!msg || typeof msg.type !== 'string') {
    return;
  }
  if (activeHandler) {
    activeHandler(msg);
  } else {
    buffered.push(msg);
  }
});

export function onMessage(handler: (msg: WebviewOutbound) => void): () => void {
  activeHandler = handler;
  // Flush anything that arrived while the App was mounting.
  for (const msg of buffered) {
    handler(msg);
  }
  buffered = [];

  return () => {
    if (activeHandler === handler) {
      activeHandler = undefined;
    }
  };
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
