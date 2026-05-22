import { test, expect, connectShell } from './fixtures';

test.describe('mode switching', () => {
  test.beforeEach(async ({ webview }) => {
    await connectShell(webview, { mode: 'agent' });
  });

  test('clicking a mode tab posts set_mode with the chosen value', async ({ webview }) => {
    await webview.clearSent();
    await webview.page.getByRole('button', { name: 'Chat', exact: true }).click();
    await expect.poll(() => webview.sentMessages()).toContainEqual({
      type: 'set_mode',
      mode: 'chat',
    });

    await webview.page.getByRole('button', { name: 'Swarm', exact: true }).click();
    await expect.poll(() => webview.sentMessages()).toContainEqual({
      type: 'set_mode',
      mode: 'swarm',
    });
  });

  test('footer reflects the active mode', async ({ webview }) => {
    // The footer is the always-visible mode cue; tests for placeholder
    // text live in composer.spec.ts.
    await webview.push({ type: 'mode', mode: 'chat' });
    await expect(webview.page.locator('.footer-mode')).toHaveText('chat');

    await webview.push({ type: 'mode', mode: 'swarm' });
    await expect(webview.page.locator('.footer-mode')).toHaveText('swarm');

    await webview.push({ type: 'mode', mode: 'agent' });
    await expect(webview.page.locator('.footer-mode')).toHaveText('agent');
  });
});
