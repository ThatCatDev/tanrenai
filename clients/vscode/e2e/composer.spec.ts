import { test, expect, connectShell } from './fixtures';

test.describe('composer', () => {
  test.beforeEach(async ({ webview }) => {
    await connectShell(webview, { mode: 'agent' });
  });

  test('Send is disabled when empty, enables once typed', async ({ webview }) => {
    const send = webview.page.getByRole('button', { name: 'Send' });
    await expect(send).toBeDisabled();
    await webview.page.getByPlaceholder(/Ask Tanrenai/).fill('hi');
    await expect(send).toBeEnabled();
  });

  test('Send posts a user message with the typed content', async ({ webview }) => {
    await webview.page.getByPlaceholder(/Ask Tanrenai/).fill('hello world');
    await webview.page.getByRole('button', { name: 'Send' }).click();
    await expect.poll(() => webview.sentMessages()).toContainEqual({
      type: 'send',
      content: 'hello world',
    });
  });

  test('Enter submits when not holding Shift', async ({ webview }) => {
    const input = webview.page.getByPlaceholder(/Ask Tanrenai/);
    await input.fill('go');
    await input.press('Enter');
    await expect.poll(() => webview.sentMessages()).toContainEqual({
      type: 'send',
      content: 'go',
    });
  });

  test('placeholder reflects the active mode', async ({ webview }) => {
    // Mode is wired to the placeholder so users understand what's going
    // out — "Brief the swarm…" reads very differently from "Ask Tanrenai…"
    // and is the only visible mode cue inside the composer.
    await expect(webview.page.getByPlaceholder('Ask Tanrenai…')).toBeVisible();

    await webview.push({ type: 'mode', mode: 'chat' });
    await expect(webview.page.getByPlaceholder(/Chat with Tanrenai/)).toBeVisible();

    await webview.push({ type: 'mode', mode: 'swarm' });
    await expect(webview.page.getByPlaceholder(/Brief the swarm/)).toBeVisible();
  });
});
