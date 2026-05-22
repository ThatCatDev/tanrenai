import { test, expect, connectShell } from './fixtures';

/**
 * Header surface — model link, tool count badge, mode tabs, Clear button.
 * Each of these is a visible always-mounted control once connected;
 * regressions here are the kind that an entire team would file on
 * day one ("I can't find the model picker anymore").
 */
test.describe('header', () => {
  test.beforeEach(async ({ webview }) => {
    await connectShell(webview, { model: 'foo-model', toolCount: 5 });
  });

  test('shows model name + tool count', async ({ webview }) => {
    await expect(webview.page.getByText('foo-model')).toBeVisible();
    await expect(webview.page.getByText(/5 tools/)).toBeVisible();
  });

  test('tool count updates when a fresh connected state arrives', async ({ webview }) => {
    await expect(webview.page.getByText(/5 tools/)).toBeVisible();
    await webview.push({
      type: 'state',
      state: { status: 'connected', model: 'foo-model', toolCount: 12 },
    });
    await expect(webview.page.getByText(/12 tools/)).toBeVisible();
    await expect(webview.page.getByText(/5 tools/)).toHaveCount(0);
  });

  test('Clear button in the header posts {type:"clear_chat"}', async ({ webview }) => {
    // Distinct from the footer-menu Clear chat item — both surfaces
    // post the same event so users can clear from wherever their eye
    // happens to land. Regression on either is a silent UX downgrade.
    await webview.clearSent();
    await webview.page.locator('.header').getByRole('button', { name: /^clear$/i }).click();
    await expect.poll(() => webview.sentMessages()).toContainEqual({ type: 'clear_chat' });
  });

  test('active mode tab has the active styling', async ({ webview }) => {
    // The active tab is the only visible signal of which mode the user
    // is in, aside from the footer mode chip. The styling difference
    // must be tied to the actual mode state.
    await webview.push({ type: 'mode', mode: 'swarm' });
    const swarmTab = webview.page.getByRole('button', { name: 'Swarm', exact: true });
    const agentTab = webview.page.getByRole('button', { name: 'Agent', exact: true });
    // Active tab carries a class — exact name is implementation
    // detail; we just need it to differ from the inactive ones.
    const swarmClass = (await swarmTab.getAttribute('class')) ?? '';
    const agentClass = (await agentTab.getAttribute('class')) ?? '';
    expect(swarmClass).not.toBe(agentClass);
  });

  test('model link in the header is a clickable affordance', async ({ webview }) => {
    // Distinct from the footer "Choose model" menu — the inline link
    // is the highest-traffic CTA for power users. Both must work.
    await webview.clearSent();
    await webview.page.getByText('foo-model').click();
    await expect.poll(() => webview.sentMessages()).toContainEqual({ type: 'pick_model' });
  });
});
