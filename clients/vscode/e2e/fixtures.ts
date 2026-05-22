import { test as base, expect, type Page } from '@playwright/test';

/**
 * Test fixtures + helpers built around the dev shell's message-injection
 * harness (window.__pushMsg / window.__sentMessages — see webview/dev.html).
 *
 * Tests get a `webview` fixture pre-loaded with the connected shell, and
 * helpers to push fake `WebviewOutbound` messages or read what the UI
 * posted back. Lets specs read like a script: "push these messages, click
 * this button, assert this DOM".
 */

export interface Webview {
  /** Send a single `WebviewOutbound` message into the webview as if the
   *  host posted it. Type is `unknown` to keep tests dependency-free —
   *  the real protocol type lives in `src/protocol.ts` but pulling it
   *  here would couple e2e to source layout. Tests use literals. */
  push(msg: unknown): Promise<void>;
  /** Send a sequence of messages with an optional inter-message delay.
   *  Use a non-zero delay when the spec needs to observe live streaming
   *  (e.g. token_rate updates during a long generation). */
  pushSequence(msgs: unknown[], delayMs?: number): Promise<void>;
  /** Read everything the webview has posted back to the host (clicks,
   *  composer sends, approval decisions). Acts as the assertion surface
   *  for "did the click do the right thing?". */
  sentMessages(): Promise<unknown[]>;
  /** Wipe the sent-messages log between assertions. */
  clearSent(): Promise<void>;
  /** Direct access for the rare case a fixture method doesn't cover. */
  page: Page;
}

export const test = base.extend<{ webview: Webview }>({
  webview: async ({ page }, run) => {
    // Suppress the dev shell's auto-connect so each test starts from a
    // known idle state. Specs that need a connected shell call
    // `webview.push({ type: 'state', state: { status: 'connected', ... } })`
    // themselves.
    //
    // Also clear any persisted shell from a previous test. localStorage
    // is per-origin and survives page navigation, so without this a
    // test that left "connected" persisted would pre-paint the chat
    // shell for the next test and trip "starts idle" assertions.
    await page.goto('/?nostate=1');
    await page.evaluate(() => {
      try {
        localStorage.removeItem('__dev_vscode_state');
      } catch { /* private mode — fine */ }
    });
    await page.reload();
    // Wait until the bundle has wired up the message listener.
    await page.waitForFunction(() => typeof (window as any).__pushMsg === 'function');

    const wv: Webview = {
      page,
      push: (msg) =>
        page.evaluate((m) => (window as any).__pushMsg(m), msg),
      pushSequence: (msgs, delayMs = 0) =>
        page.evaluate(
          ([list, d]) => (window as any).__pushSequence(list, d),
          [msgs, delayMs] as const,
        ),
      sentMessages: () =>
        page.evaluate(() => (window as any).__sentMessages.slice()),
      clearSent: () =>
        page.evaluate(() => {
          (window as any).__sentMessages.length = 0;
        }),
    };
    await run(wv);
  },
});

export { expect };

/** Convenience: push the minimum set of messages to land on the connected
 *  chat shell. Most specs that aren't testing connection states want to
 *  start here. */
export async function connectShell(
  wv: Webview,
  opts: { model?: string; toolCount?: number; mode?: 'chat' | 'agent' | 'swarm' } = {},
) {
  const model = opts.model ?? 'test-model';
  const toolCount = opts.toolCount ?? 5;
  const mode = opts.mode ?? 'agent';
  await wv.push({ type: 'state', state: { status: 'connected', model, toolCount } });
  await wv.push({ type: 'mode', mode });
  // Wait for the header to render so subsequent assertions don't race
  // the initial paint.
  await wv.page.getByText(model).waitFor();
}
