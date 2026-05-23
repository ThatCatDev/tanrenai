import { test, expect, connectShell } from './fixtures';

/**
 * Persistence edge cases beyond reload.spec.ts:
 *   - Saving the same shell twice is idempotent (no quota churn)
 *   - Persisted shell survives multiple reload cycles
 *   - Mode change after a reload still updates and persists
 *   - Switching from connected → error → connected doesn't leave
 *     stale persisted shell pointing at an old model
 */
test.describe('persistence edge cases', () => {
  test('mode change persists across reload', async ({ webview }) => {
    await webview.push({
      type: 'state',
      state: { status: 'connected', model: 'm', toolCount: 1 },
    });
    await webview.push({ type: 'mode', mode: 'swarm' });
    await webview.page.locator('.footer-mode').waitFor();
    await expect(webview.page.locator('.footer-mode')).toHaveText('swarm');

    await webview.page.goto('/?nostate=1');
    await expect(webview.page.locator('.footer-mode')).toHaveText('swarm');
  });

  test('multiple reload cycles preserve the same shell', async ({ webview }) => {
    await webview.push({
      type: 'state',
      state: { status: 'connected', model: 'persist-3x', toolCount: 7 },
    });
    await webview.push({ type: 'mode', mode: 'chat' });
    await webview.page.getByText('persist-3x').waitFor();

    for (let i = 0; i < 3; i++) {
      await webview.page.goto('/?nostate=1');
      await expect(webview.page.getByText('persist-3x')).toBeVisible();
      await expect(webview.page.locator('.footer-mode')).toHaveText('chat');
    }
  });

  test('connected → error → connected with new model updates persisted shell', async ({
    webview,
  }) => {
    await webview.push({
      type: 'state',
      state: { status: 'connected', model: 'old-model-name', toolCount: 1 },
    });
    await webview.page.getByText('old-model-name').waitFor();

    // Error blows up connection-level state. The persisted shell on
    // disk now reflects status=error, which the next reload would
    // pick up — surface that.
    await webview.push({
      type: 'state',
      state: { status: 'error', message: 'persisted-error-marker' },
    });
    // Wait for the persistence effect (which fires after paint) to have
    // committed before we reload — waitFor on text is paint-done but
    // not effect-done; goto would race the useEffect.
    await webview.page.waitForFunction(() => {
      const raw = localStorage.getItem('__dev_vscode_state');
      return raw !== null && raw.includes('persisted-error-marker');
    });
    await webview.page.goto('/?nostate=1');
    await expect(webview.page.getByText('persisted-error-marker')).toBeVisible();

    // Recovery: new connected state with a different model.
    await webview.push({
      type: 'state',
      state: { status: 'connected', model: 'persist-new-model', toolCount: 2 },
    });
    await webview.page.getByText('persist-new-model').waitFor();
    // Wait until the persistence effect has actually written the new
    // model to localStorage — useEffect runs after render, so the
    // "text is visible" milestone isn't enough on its own.
    await webview.page.waitForFunction(() => {
      const raw = localStorage.getItem('__dev_vscode_state');
      return raw !== null && raw.includes('persist-new-model');
    });

    // Final reload should show the *new* model, not the old one.
    await webview.page.goto('/?nostate=1');
    await expect(webview.page.getByText('persist-new-model')).toBeVisible();
    await expect(webview.page.getByText('old-model-name')).toHaveCount(0);
  });
});
