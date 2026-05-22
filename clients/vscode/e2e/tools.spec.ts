import { test, expect, connectShell } from './fixtures';

test.describe('tool calls + approvals', () => {
  test.beforeEach(async ({ webview }) => {
    await connectShell(webview, { mode: 'agent' });
  });

  test('tool call renders with name + args, result collapses behind a toggle', async ({ webview }) => {
    await webview.pushSequence([
      { type: 'turn_start' },
      { type: 'iteration_start', iteration: 1, maxIterations: 8 },
      {
        type: 'tool_call',
        id: 't1',
        name: 'file_read',
        arguments: '{"path":"src/foo.ts"}',
        intercepted: false,
      },
      {
        type: 'tool_result',
        id: 't1',
        ok: true,
        content: 'export function foo() {}',
      },
    ]);

    // Tool name in a code span; args visible nearby.
    await expect(webview.page.getByText('file_read').first()).toBeVisible();
    await expect(webview.page.getByText('{"path":"src/foo.ts"}')).toBeVisible();
    // Result is collapsed by default — the ▸ triangle is the visible cue.
    const resultToggle = webview.page.getByText(/Result/);
    await expect(resultToggle).toBeVisible();
  });

  test('approval prompt shows three explicit choices and posts the decision', async ({ webview }) => {
    await webview.push({ type: 'turn_start' });
    await webview.push({
      type: 'approval_required',
      id: 'a1',
      name: 'shell_exec',
      arguments: '{"cmd":"rm -rf /tmp/x"}',
    });

    // All three buttons must be present and distinct — auto-approving
    // (or missing Deny entirely) would be a security regression.
    const allow = webview.page.getByRole('button', { name: /allow once/i });
    const always = webview.page.getByRole('button', { name: /always/i });
    const deny = webview.page.getByRole('button', { name: /deny/i });
    await expect(allow).toBeVisible();
    await expect(always).toBeVisible();
    await expect(deny).toBeVisible();

    await webview.clearSent();
    await deny.click();
    await expect.poll(() => webview.sentMessages()).toContainEqual({
      type: 'approval_decision',
      id: 'a1',
      action: 'deny',
    });
  });

  test('activity bar surfaces awaiting-approval state', async ({ webview }) => {
    await webview.push({ type: 'turn_start' });
    await webview.push({ type: 'iteration_start', iteration: 1, maxIterations: 8 });
    await webview.push({
      type: 'approval_required',
      id: 'a1',
      name: 'shell_exec',
      arguments: '{"cmd":"x"}',
    });
    // Approval pending is the most blocking state — it must beat any
    // streaming/thinking activity in the bar so the user sees the
    // prompt isn't waiting on the model, it's waiting on them.
    await expect(webview.page.getByText(/awaiting approval/)).toBeVisible();
    await expect(webview.page.getByText(/shell_exec/).first()).toBeVisible();
  });
});
