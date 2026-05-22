import { test, expect, connectShell } from './fixtures';

/**
 * Multiple-approval queuing — the agent can request approval for tool A,
 * the user delays, the agent (or a parallel path) requests approval for
 * tool B. Both must be visible, individually resolvable, and the
 * decisions must carry the right ids so the controller routes each
 * decision back to the right pending RPC.
 */
test.describe('multiple approvals', () => {
  test.beforeEach(async ({ webview }) => {
    await connectShell(webview, { mode: 'agent' });
  });

  test('two simultaneous approvals both render', async ({ webview }) => {
    await webview.push({ type: 'turn_start' });
    await webview.push({
      type: 'approval_required',
      id: 'a1',
      name: 'shell_exec',
      arguments: '{"cmd":"ls"}',
    });
    await webview.push({
      type: 'approval_required',
      id: 'a2',
      name: 'file_write',
      arguments: '{"path":"x"}',
    });

    await expect(webview.page.locator('.approval')).toHaveCount(2);
  });

  test('resolving one approval leaves the other pending', async ({ webview }) => {
    await webview.push({ type: 'turn_start' });
    await webview.push({
      type: 'approval_required',
      id: 'a1',
      name: 'shell_exec',
      arguments: '{"cmd":"ls"}',
    });
    await webview.push({
      type: 'approval_required',
      id: 'a2',
      name: 'file_write',
      arguments: '{"path":"x"}',
    });

    // Approve the first by id-scoped lookup.
    const first = webview.page.locator('.approval', { hasText: 'shell_exec' });
    const second = webview.page.locator('.approval', { hasText: 'file_write' });
    await first.getByRole('button', { name: /allow once/i }).click();

    // Server confirms.
    await webview.push({ type: 'approval_resolved', id: 'a1' });
    await expect(first).toHaveClass(/resolved/);

    // Second still pending — buttons still present.
    await expect(second).not.toHaveClass(/resolved/);
    await expect(second.getByRole('button', { name: /allow once/i })).toBeVisible();
  });

  test('each decision posts with its own id', async ({ webview }) => {
    await webview.push({ type: 'turn_start' });
    await webview.push({
      type: 'approval_required',
      id: 'a1',
      name: 'shell_exec',
      arguments: '{"cmd":"a"}',
    });
    await webview.push({
      type: 'approval_required',
      id: 'a2',
      name: 'shell_exec',
      arguments: '{"cmd":"b"}',
    });

    await webview.clearSent();
    // Same tool name, different args — disambiguate by args text so
    // we click the right one. Regression that confused ids would
    // send the wrong decision back.
    await webview.page
      .locator('.approval', { hasText: '{"cmd":"a"}' })
      .getByRole('button', { name: /deny/i })
      .click();
    await webview.page
      .locator('.approval', { hasText: '{"cmd":"b"}' })
      .getByRole('button', { name: /allow once/i })
      .click();

    const sent = await webview.sentMessages();
    expect(sent).toContainEqual({ type: 'approval_decision', id: 'a1', action: 'deny' });
    expect(sent).toContainEqual({ type: 'approval_decision', id: 'a2', action: 'allow' });
  });
});
