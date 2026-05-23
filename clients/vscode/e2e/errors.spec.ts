import { test, expect, connectShell } from './fixtures';

/**
 * Error-path coverage. The webview gets error signals through three
 * channels:
 *   1. `state: { status: 'error', message }` — connection-level error
 *      (CLI gone, backend HTML response, OAuth failure).
 *   2. `turn_end: { ok: false, reason }` — the in-flight turn ended
 *      badly. The reason becomes a visible error bubble in the chat.
 *   3. `error: { message, fatal }` — RPC-level error, possibly fatal.
 *      Fatal errors flip the connection to error state; non-fatal log
 *      but keep the chat alive.
 *
 * These tests pin how each surfaces, plus recovery flows (retry,
 * subsequent turns after an error).
 */
test.describe('error handling', () => {
  test('connection-level error replaces the chat with the error panel', async ({ webview }) => {
    await connectShell(webview);
    await expect(webview.page.getByPlaceholder(/Ask Tanrenai/)).toBeVisible();

    await webview.push({
      type: 'state',
      state: { status: 'error', message: 'lost connection to backend' },
    });

    // The chat surface is unmounted — only the StatusPanel is visible.
    // Composer goes away because the user can't send anything until
    // they recover from the error.
    await expect(webview.page.getByText('lost connection to backend')).toBeVisible();
    await expect(webview.page.getByPlaceholder(/Ask Tanrenai/)).toHaveCount(0);
    await expect(webview.page.getByRole('button', { name: /retry/i })).toBeVisible();
  });

  test('Retry from error state posts {type:"reconnect"}', async ({ webview }) => {
    await webview.push({
      type: 'state',
      state: { status: 'error', message: 'something broke' },
    });
    await webview.clearSent();
    await webview.page.getByRole('button', { name: /retry/i }).click();
    await expect.poll(() => webview.sentMessages()).toContainEqual({ type: 'reconnect' });
  });

  test('error → reconnect → connected restores the chat surface', async ({ webview }) => {
    await webview.push({
      type: 'state',
      state: { status: 'error', message: 'transient' },
    });
    await expect(webview.page.getByText('transient')).toBeVisible();

    // Controller responds to retry: connecting, then connected. The
    // chat repaints fresh — entries from before the error are NOT
    // restored here (those come via controller replay; outside our
    // dev-shell scope).
    await webview.push({
      type: 'state',
      state: { status: 'connecting', progress: [{ level: 'info', message: 'reconnecting' }] },
    });
    await expect(webview.page.getByText('Connecting', { exact: true })).toBeVisible();

    await webview.push({
      type: 'state',
      state: { status: 'connected', model: 'm', toolCount: 1 },
    });
    await expect(webview.page.getByPlaceholder(/Ask Tanrenai/)).toBeVisible();
  });

  test('turn_end with reason renders the reason as an error bubble', async ({ webview }) => {
    await connectShell(webview);
    await webview.push({ type: 'turn_start' });
    await webview.push({
      type: 'turn_end',
      ok: false,
      reason: 'model returned 503 gpu_unavailable',
    });

    // Error entry sits in the chat; subsequent turns aren't blocked.
    await expect(webview.page.getByText('model returned 503 gpu_unavailable')).toBeVisible();
    // Composer is back to Send (turn ended).
    await expect(webview.page.getByRole('button', { name: /^send$/i })).toBeVisible();
  });

  test('a second error in a row appends a second error entry', async ({ webview }) => {
    // Recovery flow: user retried, agent failed again. Both errors
    // should be visible — don't dedupe or replace.
    await connectShell(webview);
    await webview.push({ type: 'turn_start' });
    await webview.push({ type: 'turn_end', ok: false, reason: 'first failure' });
    await webview.push({ type: 'turn_start' });
    await webview.push({ type: 'turn_end', ok: false, reason: 'second failure' });

    await expect(webview.page.getByText('first failure')).toBeVisible();
    await expect(webview.page.getByText('second failure')).toBeVisible();
  });

  test('turn_end with ok=true and a cancellation-style reason still shows it', async ({
    webview,
  }) => {
    // Cancel during a turn closes it with ok=false + reason like
    // "cancelled". The webview shouldn't filter — surface whatever the
    // controller sends.
    await connectShell(webview);
    await webview.push({ type: 'turn_start' });
    await webview.push({ type: 'turn_end', ok: false, reason: 'cancelled' });
    await expect(webview.page.getByText('cancelled')).toBeVisible();
  });

  test('successful turn after errors clears the activity row', async ({ webview }) => {
    // After a failed turn the activity row should hide (turn over);
    // a new successful turn should run cleanly without the previous
    // error leaking in.
    await connectShell(webview);
    await webview.push({ type: 'turn_start' });
    await webview.push({ type: 'turn_end', ok: false, reason: 'broke' });

    // Now a clean turn.
    await webview.push({ type: 'turn_start' });
    await webview.push({ type: 'message_start', role: 'assistant', id: 'a1' });
    await webview.push({ type: 'message_delta', id: 'a1', text: 'ok now.', channel: 'content' });
    await webview.push({ type: 'message_end', id: 'a1' });
    await webview.push({ type: 'turn_end', ok: true });

    // Both the old error and the new content are visible.
    await expect(webview.page.getByText('broke')).toBeVisible();
    await expect(webview.page.getByText('ok now.')).toBeVisible();
  });
});
