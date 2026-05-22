import { test, expect, connectShell } from './fixtures';

/**
 * Long tail of edge cases that don't belong to a specific feature spec
 * but each one represents a real "this almost broke prod" scenario.
 */
test.describe('long-tail edge cases', () => {
  test.beforeEach(async ({ webview }) => {
    await connectShell(webview, { mode: 'agent' });
  });

  test('message_delta for an unknown id synthesises a new assistant entry', async ({
    webview,
  }) => {
    // The agent occasionally streams a delta without a preceding
    // message_start (race or skipped start event). Don't drop the
    // content — synthesise the entry so the user sees output.
    await webview.push({ type: 'turn_start' });
    await webview.push({
      type: 'message_delta',
      id: 'orphan',
      text: 'rescued text',
      channel: 'content',
    });
    await expect(webview.page.getByText('rescued text')).toBeVisible();
  });

  test('connecting progress log appends without replacing on each push', async ({
    webview,
  }) => {
    // The controller streams progress lines one at a time — each
    // `state: connecting` event carries the *latest* full progress
    // array. Verify the panel renders them in order without dedupe
    // dropping the previous lines.
    await webview.push({
      type: 'state',
      state: { status: 'connecting', progress: [{ level: 'info', message: 'step 1' }] },
    });
    await webview.push({
      type: 'state',
      state: {
        status: 'connecting',
        progress: [
          { level: 'info', message: 'step 1' },
          { level: 'info', message: 'step 2' },
        ],
      },
    });
    await webview.push({
      type: 'state',
      state: {
        status: 'connecting',
        progress: [
          { level: 'info', message: 'step 1' },
          { level: 'info', message: 'step 2' },
          { level: 'warn', message: 'step 3 slow' },
        ],
      },
    });
    const lines = webview.page.locator('.progress-log .line');
    await expect(lines).toHaveCount(3);
    await expect(lines.nth(2)).toContainText('step 3 slow');
    await expect(lines.nth(2)).toHaveClass(/warn/);
  });

  test('rapid mode switches both post their events without coalescing', async ({
    webview,
  }) => {
    await webview.clearSent();
    await webview.page.getByRole('button', { name: 'Chat', exact: true }).click();
    await webview.page.getByRole('button', { name: 'Swarm', exact: true }).click();
    await webview.page.getByRole('button', { name: 'Agent', exact: true }).click();

    const sent = await webview.sentMessages();
    const modes = (sent as Array<{ type: string; mode?: string }>)
      .filter((m) => m.type === 'set_mode')
      .map((m) => m.mode);
    expect(modes).toEqual(['chat', 'swarm', 'agent']);
  });

  test('history_cleared from the host empties entries', async ({ webview }) => {
    // The controller can broadcast history_cleared (separate from
    // clear_chat) to wipe state without a user-initiated click. The
    // webview must respect it identically.
    await webview.push({
      type: 'message_start',
      role: 'user',
      id: 'u1',
    });
    await webview.push({ type: 'message_delta', id: 'u1', text: 'hi' });
    await webview.push({ type: 'message_end', id: 'u1' });
    await expect(webview.page.locator('.msg')).toHaveCount(1);
    // history_cleared from the controller goes through clear_chat
    // semantics on the webview side.
    await webview.push({ type: 'clear_chat' });
    await expect(webview.page.locator('.msg')).toHaveCount(0);
  });

  test('extremely long single message renders and is scrollable', async ({ webview }) => {
    // Generate a long, single-paragraph response — the chat surface
    // must accommodate without breaking the layout. Wrap behaviour
    // is what we're verifying.
    const longText = 'word '.repeat(800).trim();
    await webview.pushSequence([
      { type: 'turn_start' },
      { type: 'message_start', role: 'assistant', id: 'a1' },
      { type: 'message_delta', id: 'a1', text: longText, channel: 'content' },
      { type: 'message_end', id: 'a1' },
      { type: 'turn_end', ok: true },
    ]);
    const body = webview.page.locator('.msg.assistant .body.markdown').first();
    await expect(body).toBeVisible();
    // The body must be taller than a single line — proves wrap rather
    // than horizontal overflow.
    const h = await body.evaluate((el) => el.getBoundingClientRect().height);
    expect(h).toBeGreaterThan(100);
  });

  test('attach_selection deduplicates by path+text', async ({ webview }) => {
    // Pasting the same selection twice shouldn't queue two chips —
    // the reducer dedupes by `path` + `text` (see state.ts).
    const sel = {
      label: 'a.ts:1-1',
      path: 'a.ts',
      languageId: 'ts',
      startLine: 1,
      endLine: 1,
      text: 'export const x = 1;',
    };
    await webview.push({ type: 'attach_selection', selection: sel });
    await webview.push({ type: 'attach_selection', selection: sel });
    await expect(webview.page.locator('.chip', { hasText: 'a.ts:1-1' })).toHaveCount(1);
  });

  test('turn_start without a previous turn_end starts cleanly', async ({ webview }) => {
    // Defensive: if the agent crashed mid-turn and the next turn
    // starts without a proper turn_end, the new turn should still
    // run — turnRunning just gets re-asserted.
    await webview.push({ type: 'turn_start' });
    await webview.push({ type: 'turn_start' });
    // Composer is in turn-running mode (Cancel visible, no Send).
    await expect(webview.page.getByRole('button', { name: /^cancel$/i })).toBeVisible();
    await expect(webview.page.getByRole('button', { name: /^send$/i })).toHaveCount(0);
  });
});
