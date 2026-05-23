import { test, expect, connectShell } from './fixtures';

/**
 * Footer menu items map 1:1 to WebviewInbound events the controller
 * handles. Each menu click is a contract — a regression that renamed
 * `clear_chat` to `clearChat` would break the controller's switch
 * dispatch silently, no console error. These tests pin the wire format
 * and the menu-dismiss behavior.
 */
test.describe('footer menu', () => {
  test.beforeEach(async ({ webview }) => {
    await connectShell(webview);
  });

  // Each [label, event] pair: clicking the menu item must post exactly
  // that event. Three-dot trigger opens the menu; click closes it.
  const items: Array<[RegExp, { type: string }]> = [
    [/choose model/i, { type: 'pick_model' }],
    [/^reconnect$/i, { type: 'reconnect' }],
    [/clear chat/i, { type: 'clear_chat' }],
    [/gpu status/i, { type: 'show_gpu_status' }],
    [/^stop gpu$/i, { type: 'stop_gpu' }],
    [/destroy gpu/i, { type: 'destroy_gpu' }],
  ];

  for (const [label, expected] of items) {
    test(`"${label.source}" posts ${JSON.stringify(expected)}`, async ({ webview }) => {
      await webview.page.locator('.footer-trigger').click();
      await webview.clearSent();
      await webview.page.getByRole('menuitem', { name: label }).click();

      await expect.poll(() => webview.sentMessages()).toContainEqual(expected);
      // Menu auto-dismisses after action.
      await expect(webview.page.locator('.footer-menu-panel')).toHaveCount(0);
    });
  }

  test('"Destroy GPU…" item carries the danger style', async ({ webview }) => {
    // Visual cue is the only thing protecting the user from misclicking
    // a destructive operation in a long flat menu. Keep the class on.
    await webview.page.locator('.footer-trigger').click();
    const destroy = webview.page.getByRole('menuitem', { name: /destroy gpu/i });
    await expect(destroy).toHaveClass(/menu-danger/);
  });

  test('clicking outside the menu dismisses it', async ({ webview }) => {
    await webview.page.locator('.footer-trigger').click();
    await expect(webview.page.locator('.footer-menu-panel')).toBeVisible();

    // Click the chat area — a place that's not the menu and not the trigger.
    await webview.page.locator('#root').click({ position: { x: 200, y: 200 } });
    await expect(webview.page.locator('.footer-menu-panel')).toHaveCount(0);
  });
});
