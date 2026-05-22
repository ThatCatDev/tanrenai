import { test, expect, connectShell } from './fixtures';

/**
 * Sign-in / sign-out happen through native VS Code chrome (OAuth flow,
 * notifications). The webview's job is to surface the right CTA in the
 * right state and post `login` / `logout` events so the controller can
 * hand off — these tests pin those event posts. Anything past the post
 * lives in `vscode-extension-tester`-land, not here.
 */
test.describe('auth', () => {
  test('Sign In CTA in the status panel posts {type:"login"}', async ({ webview }) => {
    await webview.push({ type: 'state', state: { status: 'no_credentials' } });
    await webview.page.getByRole('button', { name: /sign in/i }).click();
    await expect.poll(() => webview.sentMessages()).toContainEqual({ type: 'login' });
  });

  test('footer menu shows "Sign out" when signed in (connected state)', async ({ webview }) => {
    // Connected state implies signedIn=true in App.tsx's renderRoot.
    await connectShell(webview);
    await webview.page.locator('.footer-trigger').click();
    await expect(webview.page.getByRole('menuitem', { name: /sign out/i })).toBeVisible();
    await expect(webview.page.getByRole('menuitem', { name: /^sign in$/i })).toHaveCount(0);
  });

  test('Sign Out menu item posts {type:"logout"} and closes the menu', async ({ webview }) => {
    await connectShell(webview);
    await webview.page.locator('.footer-trigger').click();
    await webview.clearSent();
    await webview.page.getByRole('menuitem', { name: /sign out/i }).click();

    await expect.poll(() => webview.sentMessages()).toContainEqual({ type: 'logout' });
    // Menu dismisses after action so users aren't left with a half-open
    // overlay covering the chat area.
    await expect(webview.page.locator('.footer-menu-panel')).toHaveCount(0);
  });

  test('footer menu shows "Sign in" when not signed in', async ({ webview }) => {
    // no_credentials state explicitly flips signedIn=false in App.tsx.
    await webview.push({ type: 'state', state: { status: 'no_credentials' } });
    // The footer is visible in the disconnected shell too — that's where
    // the menu lives.
    await webview.page.locator('.footer-trigger').click();
    await expect(webview.page.getByRole('menuitem', { name: /^sign in$/i })).toBeVisible();
    await expect(webview.page.getByRole('menuitem', { name: /sign out/i })).toHaveCount(0);
  });

  test('Sign In menu item posts {type:"login"}', async ({ webview }) => {
    await webview.push({ type: 'state', state: { status: 'no_credentials' } });
    await webview.page.locator('.footer-trigger').click();
    await webview.clearSent();
    await webview.page.getByRole('menuitem', { name: /^sign in$/i }).click();
    await expect.poll(() => webview.sentMessages()).toContainEqual({ type: 'login' });
  });
});
