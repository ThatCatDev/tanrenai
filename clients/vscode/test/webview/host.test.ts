import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

// host.ts captures a window message listener at module-load time. Tests
// reload the module per case so we get a fresh listener and clean state.
const stubVsCodeApi = () => {
  const api = {
    postMessage: vi.fn(),
    getState: vi.fn().mockReturnValue(undefined),
    setState: vi.fn().mockImplementation((v: unknown) => v),
  };
  (globalThis as { acquireVsCodeApi?: () => typeof api }).acquireVsCodeApi = () => api;

  return api;
};

const post = (msg: unknown) => {
  window.dispatchEvent(new MessageEvent('message', { data: msg }));
};

describe('host message buffering', () => {
  beforeEach(() => {
    stubVsCodeApi();
    vi.resetModules();
  });

  afterEach(() => {
    delete (globalThis as { acquireVsCodeApi?: unknown }).acquireVsCodeApi;
  });

  it('replays messages that arrived before onMessage attaches', async () => {
    const { onMessage } = await import('../../webview/host');

    // Simulate the host posting messages while the bundle is still booting.
    post({ type: 'state', state: { status: 'connecting', progress: [] } });
    post({ type: 'mode', mode: 'agent' });
    post({ type: 'message_start', role: 'user', id: 'u1' });

    const seen: unknown[] = [];
    onMessage((m) => seen.push(m));

    expect(seen).toHaveLength(3);
    expect((seen[0] as { type: string }).type).toBe('state');
    expect((seen[1] as { type: string }).type).toBe('mode');
    expect((seen[2] as { type: string }).type).toBe('message_start');
  });

  it('delivers live messages directly to the active handler', async () => {
    const { onMessage } = await import('../../webview/host');

    const seen: unknown[] = [];
    onMessage((m) => seen.push(m));

    post({ type: 'turn_start' });
    post({ type: 'turn_end', ok: true });

    expect(seen).toHaveLength(2);
    expect((seen[0] as { type: string }).type).toBe('turn_start');
  });

  it('ignores malformed messages', async () => {
    const { onMessage } = await import('../../webview/host');

    const seen: unknown[] = [];
    onMessage((m) => seen.push(m));

    post(null);
    post('a string');
    post({ noType: true });

    expect(seen).toHaveLength(0);
  });

  it('disposer detaches the handler so subsequent messages do not re-fire', async () => {
    const { onMessage } = await import('../../webview/host');

    const seen: unknown[] = [];
    const dispose = onMessage((m) => seen.push(m));

    post({ type: 'turn_start' });
    expect(seen).toHaveLength(1);

    dispose();
    post({ type: 'turn_end', ok: true });
    // After dispose, the handler should not be called again.
    expect(seen).toHaveLength(1);
  });
});
