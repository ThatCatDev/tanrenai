import { test, expect, connectShell } from './fixtures';

/**
 * Model-switching from the webview's perspective. The actual picker UI
 * is `vscode.window.showQuickPick` (native VS Code chrome — not in the
 * webview, so untestable here). The webview's surface is:
 *   1. "Choose model" menu click posts {type:"pick_model"}.
 *   2. After the user picks, the controller updates the setting and
 *      reconnects → webview receives a fresh `state` with the new model
 *      in the connected payload, then renders the new name in the header.
 *
 * This file pins both halves. The full quick-pick interaction needs
 * vscode-extension-tester.
 */
test.describe('model switching', () => {
  test.beforeEach(async ({ webview }) => {
    await connectShell(webview, { model: 'old-model-Q4_K_M', toolCount: 5 });
  });

  test('Choose model menu item posts {type:"pick_model"}', async ({ webview }) => {
    await webview.page.locator('.footer-trigger').click();
    await webview.clearSent();
    await webview.page.getByRole('menuitem', { name: /choose model/i }).click();
    await expect.poll(() => webview.sentMessages()).toContainEqual({ type: 'pick_model' });
  });

  test('header model link updates when a new connected state arrives', async ({ webview }) => {
    await expect(webview.page.getByText('old-model-Q4_K_M')).toBeVisible();

    // Simulate the controller's post-pick reconnect: brief connecting,
    // then connected with the new name. The webview must repaint the
    // header without requiring a full reload.
    await webview.push({
      type: 'state',
      state: { status: 'connecting', progress: [{ level: 'info', message: 'switching model' }] },
    });
    await expect(webview.page.getByText('Connecting')).toBeVisible();

    await webview.push({
      type: 'state',
      state: { status: 'connected', model: 'new-model-Q8_0', toolCount: 9 },
    });
    await expect(webview.page.getByText('new-model-Q8_0')).toBeVisible();
    await expect(webview.page.getByText('old-model-Q4_K_M')).toHaveCount(0);
    await expect(webview.page.getByText(/9 tools/)).toBeVisible();
  });

  test('clicking the model link in the header also posts {type:"pick_model"}', async ({
    webview,
  }) => {
    // The header model name is itself an inline "change model" affordance.
    // Hidden behind a link styling, but it's the highest-traffic CTA
    // for users who already know which model they want.
    await webview.clearSent();
    await webview.page.getByText('old-model-Q4_K_M').click();
    await expect.poll(() => webview.sentMessages()).toContainEqual({ type: 'pick_model' });
  });
});
