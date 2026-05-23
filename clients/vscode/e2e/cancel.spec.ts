import { test, expect, connectShell } from './fixtures';

/**
 * Cancellation paths — Cancel during a running turn, Cancel during
 * connect. Both buttons replace the Send / Sign In CTAs while their
 * respective operation is in flight; clicking either must post the
 * matching event so the controller can tear down the in-progress work.
 */
test.describe('cancellation', () => {
  test('Cancel button replaces Send during a running turn and posts {type:"cancel"}', async ({
    webview,
  }) => {
    await connectShell(webview);
    await webview.push({ type: 'turn_start' });
    await webview.push({ type: 'message_start', role: 'assistant', id: 'a1' });
    await webview.push({ type: 'message_delta', id: 'a1', text: 'thinking…', channel: 'reasoning' });

    // Composer transitions Send → Cancel while turnRunning is true —
    // this is the only stop-the-bleeding affordance during a long turn,
    // so it must be present and post the right event.
    const cancel = webview.page.getByRole('button', { name: /^cancel$/i });
    await expect(cancel).toBeVisible();
    await webview.clearSent();
    await cancel.click();
    await expect.poll(() => webview.sentMessages()).toContainEqual({ type: 'cancel' });
  });

  test('Cancel during connecting posts {type:"cancel_connect"}', async ({ webview }) => {
    await webview.push({
      type: 'state',
      state: {
        status: 'connecting',
        progress: [{ level: 'info', message: 'searching' }],
      },
    });
    await webview.clearSent();
    await webview.page.getByRole('button', { name: /^cancel$/i }).click();
    await expect.poll(() => webview.sentMessages()).toContainEqual({ type: 'cancel_connect' });
  });

  test('after turn_end the Cancel button reverts to Send', async ({ webview }) => {
    await connectShell(webview);
    await webview.push({ type: 'turn_start' });
    await expect(webview.page.getByRole('button', { name: /^cancel$/i })).toBeVisible();
    await webview.push({ type: 'turn_end', ok: true });
    // Send is back (disabled because composer is empty, but present).
    await expect(webview.page.getByRole('button', { name: /^send$/i })).toBeVisible();
    await expect(webview.page.getByRole('button', { name: /^cancel$/i })).toHaveCount(0);
  });
});
