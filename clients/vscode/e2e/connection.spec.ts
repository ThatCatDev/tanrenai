import { test, expect } from './fixtures';

test.describe('connection states', () => {
  test('idle on first load (no state, no persisted shell)', async ({ webview }) => {
    await webview.page.evaluate(() => (window as any).__clearPersistedState?.());
    await webview.page.goto('/?nostate=1');
    // The idle StatusPanel says "Initialising…". A turn shouldn't start
    // from this state — composer should be absent because the chat
    // surface only mounts on `connected`.
    await expect(webview.page.getByText('Initialising…')).toBeVisible();
    await expect(webview.page.getByPlaceholder(/Ask|Brief/)).toHaveCount(0);
  });

  test('no_credentials shows Sign in CTA', async ({ webview }) => {
    await webview.push({ type: 'state', state: { status: 'no_credentials' } });
    const btn = webview.page.getByRole('button', { name: /sign in/i });
    await expect(btn).toBeVisible();

    await btn.click();
    // Clicking Sign in must post a `login` event so the extension can
    // hand off to VS Code's OAuth flow — regression would mean the
    // button mounts but does nothing.
    await expect.poll(() => webview.sentMessages()).toContainEqual({ type: 'login' });
  });

  test('connecting shows spinner + progress log entries in order', async ({ webview }) => {
    await webview.push({
      type: 'state',
      state: {
        status: 'connecting',
        progress: [
          { level: 'info', message: 'Searching for GPU' },
          { level: 'info', message: 'Allocating H100' },
          { level: 'warn', message: 'Pool tight — may take 30s' },
        ],
      },
    });
    await expect(webview.page.getByText('Connecting')).toBeVisible();
    // All three lines should be visible AND in source order — the
    // status panel auto-scrolls the bottom into view, so a regression
    // that reverses the order would land them off-screen.
    const lines = webview.page.locator('.progress-log .line');
    await expect(lines).toHaveCount(3);
    await expect(lines.nth(0)).toContainText('Searching for GPU');
    await expect(lines.nth(2)).toContainText('Pool tight');
  });

  test('error shows message + Retry', async ({ webview }) => {
    await webview.push({
      type: 'state',
      state: { status: 'error', message: 'Could not find the tanrenai CLI' },
    });
    await expect(webview.page.getByText('Could not find the tanrenai CLI')).toBeVisible();

    await webview.page.getByRole('button', { name: /retry/i }).click();
    await expect.poll(() => webview.sentMessages()).toContainEqual({ type: 'reconnect' });
  });

  test('connected mounts the chat surface', async ({ webview }) => {
    await webview.push({
      type: 'state',
      state: { status: 'connected', model: 'test-model', toolCount: 7 },
    });
    await webview.push({ type: 'mode', mode: 'agent' });
    // Header shows model + tool count; composer is mounted.
    await expect(webview.page.getByText('test-model')).toBeVisible();
    await expect(webview.page.getByText(/7 tools/)).toBeVisible();
    await expect(webview.page.getByPlaceholder(/Ask Tanrenai/)).toBeVisible();
  });
});
