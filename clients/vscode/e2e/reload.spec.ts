import { test, expect } from './fixtures';

test.describe('view reload / remount', () => {
  test('persisted shell repaints connected state on reload (no idle flash)', async ({
    webview,
  }) => {
    // Set up a connected shell, let the persistence effect fire.
    await webview.push({
      type: 'state',
      state: { status: 'connected', model: 'persist-test', toolCount: 4 },
    });
    await webview.push({ type: 'mode', mode: 'swarm' });
    // Wait until the chat surface has actually rendered — that's the
    // signal that the persistence effect ran.
    await webview.page.getByText('persist-test').waitFor();

    // Simulate the webview being hidden + shown again. Use ?nostate=1
    // so the dev shell's auto-connect doesn't help us — only the
    // persisted shell can paint the connected UI on this navigation.
    await webview.page.goto('/?nostate=1');

    // Connected shell repaints with no idle/connecting flash.
    await expect(webview.page.getByText('persist-test')).toBeVisible();
    await expect(webview.page.locator('.footer-mode')).toHaveText('swarm');
    // Chat entries are NOT persisted — the controller is responsible
    // for replaying the transcript. Verify entries are empty so we
    // catch the day that contract changes accidentally.
    await expect(webview.page.locator('.user, .assistant')).toHaveCount(0);
  });

  test('reload without persisted shell stays idle', async ({ webview }) => {
    await webview.page.evaluate(() => (window as any).__clearPersistedState?.());
    await webview.page.goto('/?nostate=1');
    await expect(webview.page.getByText('Initialising…')).toBeVisible();
  });
});
