/**
 * E2E for the webview: mounts <App /> into happy-dom, posts the same
 * sequence of messages the host would, and asserts the rendered DOM. This
 * exercises the path that broke ("chat disappears on tab switch") at the
 * actual component level, not just the unit reducer.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { WebviewOutbound } from '../../src/protocol';

// Stub acquireVsCodeApi BEFORE importing host/App, since host.ts captures it
// at module load.
const stubVsCodeApi = () => {
  const state: { value: unknown } = { value: undefined };
  const api = {
    postMessage: vi.fn(),
    getState: () => state.value,
    setState: vi.fn().mockImplementation((v: unknown) => {
      state.value = v;

      return v;
    }),
  };
  (globalThis as { acquireVsCodeApi?: () => typeof api }).acquireVsCodeApi = () => api;

  return api;
};

const post = (msg: WebviewOutbound) => {
  window.dispatchEvent(new MessageEvent('message', { data: msg }));
};

// Wait for Preact's microtask-scheduled re-render to flush.
const flush = () => new Promise<void>((r) => setTimeout(r, 0));

let root: HTMLDivElement;

beforeEach(() => {
  stubVsCodeApi();
  vi.resetModules();
  document.body.innerHTML = '';
  root = document.createElement('div');
  document.body.appendChild(root);
});

afterEach(() => {
  delete (globalThis as { acquireVsCodeApi?: unknown }).acquireVsCodeApi;
});

async function mount(): Promise<void> {
  // Dynamic import so the freshly stubbed acquireVsCodeApi is captured.
  const preact = await import('preact');
  const { App } = await import('../../webview/App');
  preact.render(<App />, root);
  await flush();
}

describe('App E2E', () => {
  it('renders the connecting status with progress lines', async () => {
    await mount();
    post({
      type: 'state',
      state: {
        status: 'connecting',
        progress: [
          { message: 'Allocating GPU…', level: 'info' },
          { message: 'Downloading model (40%)…', level: 'info' },
        ],
      },
    });
    await flush();
    expect(document.body.textContent).toContain('Connecting');
    expect(document.body.textContent).toContain('Allocating GPU');
    expect(document.body.textContent).toContain('Downloading model (40%)');
  });

  it('shows the chat surface once connected and renders streaming content', async () => {
    await mount();
    post({
      type: 'state',
      state: { status: 'connected', model: 'Qwen3.6', toolCount: 5 },
    });
    post({ type: 'mode', mode: 'agent' });
    await flush();

    expect(document.body.textContent).toContain('Qwen3.6');
    expect(document.body.querySelector('textarea')).not.toBeNull();

    // Simulate a turn.
    post({ type: 'turn_start' });
    post({ type: 'message_start', role: 'user', id: 'u1' });
    post({ type: 'message_delta', id: 'u1', text: 'hello' });
    post({ type: 'message_end', id: 'u1' });
    post({ type: 'message_start', role: 'assistant', id: 'a1' });
    post({ type: 'message_delta', id: 'a1', text: 'Hi! ', channel: 'content' });
    post({ type: 'message_delta', id: 'a1', text: 'How can I help?', channel: 'content' });
    post({ type: 'message_end', id: 'a1' });
    post({ type: 'turn_end', ok: true });
    await flush();

    expect(document.body.textContent).toContain('hello');
    expect(document.body.textContent).toContain('Hi! How can I help?');
  });

  it('shows the activity indicator while a turn is in progress and removes it after turn_end', async () => {
    await mount();
    post({
      type: 'state',
      state: { status: 'connected', model: 'X', toolCount: 0 },
    });
    post({ type: 'turn_start' });
    post({ type: 'message_start', role: 'assistant', id: 'a1' });
    post({ type: 'message_delta', id: 'a1', text: 'x', channel: 'reasoning' });
    await flush();
    expect(document.body.querySelector('.activity')).not.toBeNull();
    expect(document.body.textContent).toContain('thinking');

    post({ type: 'message_delta', id: 'a1', text: 'a', channel: 'content' });
    await flush();
    expect(document.body.textContent).toContain('generating');

    post({ type: 'message_end', id: 'a1' });
    post({ type: 'turn_end', ok: true });
    await flush();
    expect(document.body.querySelector('.activity')).toBeNull();
  });

  it('shows tool spinner while a tool is awaiting result', async () => {
    await mount();
    post({
      type: 'state',
      state: { status: 'connected', model: 'X', toolCount: 1 },
    });
    post({ type: 'turn_start' });
    post({
      type: 'tool_call',
      id: 't1',
      name: 'file_read',
      arguments: '{"path":"foo"}',
      intercepted: true,
    });
    await flush();

    expect(document.body.querySelector('.tool.running')).not.toBeNull();
    expect(document.body.querySelector('.tool-spinner')).not.toBeNull();
    expect(document.body.textContent).toContain('running file_read');

    post({ type: 'tool_result', id: 't1', ok: true, content: 'data' });
    post({ type: 'turn_end', ok: true });
    await flush();

    expect(document.body.querySelector('.tool.running')).toBeNull();
  });

  it('seeds the chat shell from persisted state so a remount paints instantly', async () => {
    // Simulate the persisted shell that a previous mount would have written
    // via setState. This is what the host's getState() returns on remount.
    const api = (globalThis as { acquireVsCodeApi: () => { setState: (v: unknown) => void } })
      .acquireVsCodeApi();
    api.setState({
      connection: { status: 'connected', model: 'M', toolCount: 2 },
      mode: 'agent',
    });

    // First render — no live state messages yet.
    await mount();

    // Without the host posting any state, the chat shell is already painted
    // because init() read getPersistedShell. No "Connecting…" flash.
    expect(document.body.querySelector('textarea')).not.toBeNull();
    expect(document.body.textContent).toContain('M');
    expect(document.body.textContent).not.toContain('Connecting');

    // Replay a user message (controller's onMounted does this) and confirm
    // it lands in the chat.
    post({ type: 'message_start', role: 'user', id: 'u1' });
    post({ type: 'message_delta', id: 'u1', text: 'hello world' });
    post({ type: 'message_end', id: 'u1' });
    await flush();
    expect(document.body.textContent).toContain('hello world');
  });

  it('clears the chat on a clear_chat event', async () => {
    await mount();
    post({
      type: 'state',
      state: { status: 'connected', model: 'X', toolCount: 0 },
    });
    post({ type: 'message_start', role: 'user', id: 'u1' });
    post({ type: 'message_delta', id: 'u1', text: 'aaa' });
    post({ type: 'message_end', id: 'u1' });
    await flush();
    expect(document.body.textContent).toContain('aaa');

    post({ type: 'clear_chat' });
    await flush();
    expect(document.body.textContent).not.toContain('aaa');
  });
});
