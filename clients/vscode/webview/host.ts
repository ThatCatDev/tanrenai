// Bridge to the VS Code extension host via postMessage.

import type { WebviewInbound, WebviewOutbound } from '../src/protocol';

interface VSCodeApi {
  postMessage(msg: unknown): void;
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
